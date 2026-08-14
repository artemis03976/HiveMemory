"""Patchouli 记忆生成任务定义与对外快照。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from hivememory.core.models import LogicalBlock, PendingAtomSettlement
from hivememory.engines.generation.models import GenerationRequest
from hivememory.system.runtime.work_queue import TaskOutcome, WorkState


class MemoryGenerationTaskStatus(str, Enum):
    """对外记忆生成任务的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_WORK_STATE_TO_TASK_STATUS = {
    WorkState.QUEUED: MemoryGenerationTaskStatus.PENDING,
    WorkState.RETRY_WAIT: MemoryGenerationTaskStatus.PENDING,
    WorkState.RUNNING: MemoryGenerationTaskStatus.RUNNING,
    WorkState.SUCCEEDED: MemoryGenerationTaskStatus.COMPLETED,
    WorkState.CANCELLED: MemoryGenerationTaskStatus.CANCELLED,
    WorkState.FAILED: MemoryGenerationTaskStatus.FAILED,
    WorkState.DEAD_LETTER: MemoryGenerationTaskStatus.FAILED,
}


class MemoryGenerationSource(str, Enum):
    """触发记忆生成的领域操作来源。"""

    WRITE = "WRITE"
    UPDATE = "UPDATE"
    ARCHIVE = "ARCHIVE"

    @property
    def creation_artifact_intent(
        self,
    ) -> Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"]:
        """映射新建记忆制品使用的来源意图。"""

        if self == MemoryGenerationSource.ARCHIVE:
            return "ARCHIVE"
        if self == MemoryGenerationSource.WRITE:
            return "WRITE"
        return "SYSTEM"

    @property
    def version_update_source(
        self,
    ) -> Literal["UPDATE"]:
        """映射记忆版本更新使用的来源类型。"""

        return "UPDATE"


@dataclass(frozen=True)
class InteractionArtifactInput:
    """传递给记忆生成数据面的原始交互数据。"""

    topic_id: str
    topic_title: str = ""
    topic_summary: str = ""
    blocks: tuple[LogicalBlock, ...] = ()


@dataclass(frozen=True)
class MemoryGenerationTaskSpec:
    """记忆生成控制面与数据面共享的规范化输入。"""

    topic_id: str
    label: str
    source: MemoryGenerationSource
    request: GenerationRequest
    interaction_input: InteractionArtifactInput | None = None
    intent_id: str | None = None
    pending_alias: str | None = None


@dataclass(frozen=True)
class MemoryGenerationResult:
    """生成数据面完成持久化后返回给控制面的领域事实。

    Engine 的 ``GenerationOutcome`` 只在 Familiar 内参与 compute、artifact 与
    persist 流水线；控制面只需要最终 canonical identity 和可选的
    ``PendingAtom`` 结算事实。
    """

    canonical_alias: str | None = None
    canonical_uuid: str | None = None
    settlement: PendingAtomSettlement | None = None


@dataclass(frozen=True)
class MemoryGenerationTask:
    """单个记忆生成任务的对外只读快照。

    快照创建后不会原地更新。调用方需要通过控制器重新查询以获取新状态，不能
    把曾经取得的实例视为可观察的运行时句柄。
    """

    task_id: str
    topic_id: str
    label: str
    source: MemoryGenerationSource
    pending_alias: str | None = None
    status: MemoryGenerationTaskStatus = MemoryGenerationTaskStatus.PENDING
    canonical_alias: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested: bool = False
    cancel_reason: str | None = None

    @classmethod
    def from_spec(
        cls,
        task_id: str,
        spec: MemoryGenerationTaskSpec,
        *,
        created_at: datetime,
    ) -> MemoryGenerationTask:
        """从已接纳的任务规范创建对外初始快照。"""

        return cls(
            task_id=task_id,
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            pending_alias=spec.pending_alias,
            created_at=created_at,
        )

    @classmethod
    def from_outcome(
        cls,
        created: MemoryGenerationTask,
        outcome: TaskOutcome[tuple[MemoryGenerationResult, ...]],
        *,
        expose_terminal: bool,
    ) -> MemoryGenerationTask:
        """将通用任务结果投影为最新的只读领域快照。

        ``expose_terminal`` 为假时，即使队列已经快速结束，也只暴露最后一个可见
        的非终态；领域终态由 finalize 完成关联副作用后再对外发布。
        """

        record = outcome.record
        status = _WORK_STATE_TO_TASK_STATUS[record.state]
        if not expose_terminal and status in {
            MemoryGenerationTaskStatus.COMPLETED,
            MemoryGenerationTaskStatus.CANCELLED,
            MemoryGenerationTaskStatus.FAILED,
        }:
            status = (
                MemoryGenerationTaskStatus.RUNNING
                if record.started_at is not None
                else MemoryGenerationTaskStatus.PENDING
            )

        cancelled = expose_terminal and record.state == WorkState.CANCELLED
        failed = expose_terminal and record.state in {
            WorkState.FAILED,
            WorkState.DEAD_LETTER,
        }
        cancel_reason = outcome.cancel_reason
        if cancelled and cancel_reason is None:
            cancel_reason = "runtime_cancelled"

        return replace(
            created,
            status=status,
            canonical_alias=(
                cls._select_canonical_alias(
                    outcome.result or (),
                    pending_alias=created.pending_alias,
                )
                if expose_terminal and record.state == WorkState.SUCCEEDED
                else None
            ),
            error=(outcome.error or "memory generation work failed") if failed else None,
            started_at=record.started_at,
            finished_at=record.finished_at if expose_terminal else None,
            cancel_requested=cancel_reason is not None,
            cancel_reason=cancel_reason,
        )

    def as_failed(
        self,
        error: str,
        *,
        finished_at: datetime | None = None,
    ) -> MemoryGenerationTask:
        """从当前快照派生失败快照。"""

        return replace(
            self,
            status=MemoryGenerationTaskStatus.FAILED,
            error=error,
            finished_at=finished_at,
        )

    def with_cancel_request(self, reason: str) -> MemoryGenerationTask:
        """从当前快照派生已收到取消请求的快照。"""

        return replace(
            self,
            cancel_requested=True,
            cancel_reason=reason,
        )

    @staticmethod
    def _select_canonical_alias(
        results: tuple[MemoryGenerationResult, ...],
        *,
        pending_alias: str | None,
    ) -> str | None:
        """优先选择与 pending alias 对应的 canonical alias。"""

        candidates = results
        if pending_alias:
            matched = tuple(
                result
                for result in results
                if result.settlement is not None
                and result.settlement.pending_alias == pending_alias
            )
            if matched:
                candidates = matched
        for result in candidates:
            if result.settlement is not None and result.settlement.canonical_alias:
                return result.settlement.canonical_alias
            if result.canonical_alias:
                return result.canonical_alias
        return None

    @property
    def cancelled(self) -> bool:
        """判断任务是否已收到取消请求或已经进入取消终态。"""

        return (
            self.cancel_requested
            or self.status == MemoryGenerationTaskStatus.CANCELLED
        )


def memory_task_to_payload(
    memory_task: MemoryGenerationTask,
    *,
    reason: str | None = None,
) -> dict[str, object]:
    """将单个任务序列化为稳定的事件载荷快照。"""

    cancelled = memory_task.status == MemoryGenerationTaskStatus.CANCELLED
    return {
        "task_id": memory_task.task_id,
        "topic_id": memory_task.topic_id,
        "label": memory_task.label,
        "source": memory_task.source.value,
        "pending_alias": memory_task.pending_alias,
        "status": memory_task.status.value,
        "canonical_alias": memory_task.canonical_alias,
        "error": memory_task.error,
        "created_at": memory_task.created_at.isoformat(),
        "started_at": (
            memory_task.started_at.isoformat()
            if memory_task.started_at is not None
            else None
        ),
        "finished_at": (
            memory_task.finished_at.isoformat()
            if memory_task.finished_at is not None
            else None
        ),
        "cancel_requested": memory_task.cancel_requested,
        "cancelled": cancelled,
        "reason": reason or memory_task.cancel_reason,
    }


__all__ = [
    "InteractionArtifactInput",
    "MemoryGenerationResult",
    "MemoryGenerationSource",
    "MemoryGenerationTask",
    "MemoryGenerationTaskSpec",
    "MemoryGenerationTaskStatus",
    "memory_task_to_payload",
]
