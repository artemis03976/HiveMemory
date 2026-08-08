"""本地工作队列的通用数据契约。

本模块只描述运行时能够理解的工作项、状态和只读快照，不包含任何
Patchouli、Alice 或 server 领域模型。业务层需要把这些快照投影为自己的
task、job 或 receipt，不能直接把通用模型作为公共 API 返回。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType

from hivememory.system.runtime.work_queue.cancellation import WorkCancellationToken


class WorkState(str, Enum):
    """通用 work item 的生命周期状态。"""

    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


TERMINAL_WORK_STATES = frozenset(
    {
        WorkState.SUCCEEDED,
        WorkState.FAILED,
        WorkState.DEAD_LETTER,
        WorkState.CANCELLED,
    }
)

# 状态机只冻结通用运行时允许的直接迁移；是否允许取消仍由 lane policy 决定。
WORK_STATE_TRANSITIONS: Mapping[WorkState, frozenset[WorkState]] = MappingProxyType(
    {
        WorkState.QUEUED: frozenset({WorkState.RUNNING, WorkState.CANCELLED}),
        WorkState.RUNNING: frozenset(
            {
                WorkState.SUCCEEDED,
                WorkState.RETRY_WAIT,
                WorkState.FAILED,
                WorkState.DEAD_LETTER,
                WorkState.CANCELLED,
            }
        ),
        WorkState.RETRY_WAIT: frozenset({WorkState.QUEUED, WorkState.CANCELLED}),
        WorkState.SUCCEEDED: frozenset(),
        WorkState.FAILED: frozenset(),
        WorkState.DEAD_LETTER: frozenset(),
        WorkState.CANCELLED: frozenset(),
    }
)


def can_transition_work_state(current: WorkState, target: WorkState) -> bool:
    """判断两个通用状态之间是否存在直接迁移。"""

    return target in WORK_STATE_TRANSITIONS[current]


def _require_non_blank(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True)
class WorkItem[PayloadT]:
    """进入通用运行时的不可变工作信封。

    ``payload`` 对 runtime 保持不透明。冻结 dataclass 只保证信封字段不可改写，
    业务 adapter 仍需在 enqueue 前提供稳定快照或可序列化值。
    """

    work_id: str
    lane: str
    kind: str
    schema_version: int
    payload: PayloadT
    ordering_key: str | None = None
    priority: int = 0
    correlation_id: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.work_id, field_name="work_id")
        _require_non_blank(self.lane, field_name="lane")
        _require_non_blank(self.kind, field_name="kind")
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        for field_name in ("ordering_key", "correlation_id", "idempotency_key"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_blank(value, field_name=field_name)


@dataclass(frozen=True)
class WorkErrorSnapshot:
    """允许写入 store 与观测事件的脱敏错误快照。"""

    error_class: str
    message: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.error_class, field_name="error_class")


@dataclass(frozen=True)
class WorkRecord[PayloadT]:
    """由 store/runtime 持有的工作状态真相快照。"""

    item: WorkItem[PayloadT]
    state: WorkState
    attempt_count: int
    enqueued_at: datetime
    available_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    lease_until: datetime | None = None
    last_error: WorkErrorSnapshot | None = None
    result_ref: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_count < 0:
            raise ValueError("attempt_count must not be negative")

    @property
    def work_id(self) -> str:
        return self.item.work_id

    @property
    def lane(self) -> str:
        return self.item.lane


@dataclass(frozen=True)
class WorkExecutionContext:
    """handler 单次执行可见的通用上下文，不包含业务服务实例。"""

    work_id: str
    lane: str
    kind: str
    schema_version: int
    attempt_count: int
    correlation_id: str | None = None
    idempotency_key: str | None = None
    cancellation: WorkCancellationToken = field(default_factory=WorkCancellationToken)

    def __post_init__(self) -> None:
        _require_non_blank(self.work_id, field_name="work_id")
        _require_non_blank(self.lane, field_name="lane")
        _require_non_blank(self.kind, field_name="kind")
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be at least 1")


@dataclass(frozen=True)
class WorkReceipt:
    """runtime 接受 work item 后返回的接收凭证。

    receipt 只确认工作项已进入运行时状态真相源，不表示 handler 已执行成功，
    也不能替代 ``InteractionSubmissionReceipt``、``MemoryGenerationTask`` 或
    ``RuntimeJob`` 等领域投影。
    """

    work_id: str
    lane: str
    state: WorkState
    enqueued_at: datetime

    def __post_init__(self) -> None:
        _require_non_blank(self.work_id, field_name="work_id")
        _require_non_blank(self.lane, field_name="lane")


@dataclass(frozen=True)
class WorkLaneSnapshot:
    """store 在某一时刻按 lane 聚合的状态计数。"""

    lane: str
    queued: int = 0
    running: int = 0
    retry_wait: int = 0
    succeeded: int = 0
    failed: int = 0
    dead_letter: int = 0
    cancelled: int = 0

    @property
    def pending(self) -> int:
        return self.queued + self.retry_wait

    @property
    def active(self) -> int:
        return self.pending + self.running

    @property
    def terminal(self) -> int:
        return self.succeeded + self.failed + self.dead_letter + self.cancelled


@dataclass(frozen=True)
class WorkLaneShutdownSummary:
    """单条 lane 在 stop 返回时的 drain 结果。"""

    lane: str
    queued: int
    retry_wait: int
    running: int
    cancelled_during_shutdown: int = 0
    drain_timed_out: bool = False
    in_memory_loss_risk: int = 0

    @property
    def pending(self) -> int:
        return self.queued + self.retry_wait


@dataclass(frozen=True)
class WorkQueueShutdownSummary:
    """runtime stop 后按 lane 汇总的未完成工作快照。"""

    lanes: tuple[WorkLaneShutdownSummary, ...]
    already_stopped: bool = False

    @property
    def queued(self) -> int:
        return sum(lane.queued for lane in self.lanes)

    @property
    def retry_wait(self) -> int:
        return sum(lane.retry_wait for lane in self.lanes)

    @property
    def pending(self) -> int:
        return sum(lane.pending for lane in self.lanes)

    @property
    def running(self) -> int:
        return sum(lane.running for lane in self.lanes)

    @property
    def cancelled_during_shutdown(self) -> int:
        return sum(lane.cancelled_during_shutdown for lane in self.lanes)

    @property
    def in_memory_loss_risk(self) -> int:
        return sum(lane.in_memory_loss_risk for lane in self.lanes)


__all__ = [
    "TERMINAL_WORK_STATES",
    "WORK_STATE_TRANSITIONS",
    "WorkErrorSnapshot",
    "WorkExecutionContext",
    "WorkItem",
    "WorkLaneShutdownSummary",
    "WorkLaneSnapshot",
    "WorkQueueShutdownSummary",
    "WorkReceipt",
    "WorkRecord",
    "WorkState",
    "can_transition_work_state",
]
