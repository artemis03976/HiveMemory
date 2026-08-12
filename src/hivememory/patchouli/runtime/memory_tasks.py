"""Patchouli 记忆生成任务定义与对外快照。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from hivememory.core.models import LogicalBlock, PendingAtomSettlement
from hivememory.engines.generation.models import DuplicateDecision, GenerationRequest
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
    MERGE = "MERGE"
    SPLIT = "SPLIT"

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
    ) -> Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"]:
        """映射记忆版本更新使用的来源类型。"""

        if self == MemoryGenerationSource.MERGE:
            return "MERGE"
        if self == MemoryGenerationSource.SPLIT:
            return "SYSTEM_REWRITE"
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
class MemoryGenerationWork:
    """单个记忆生成任务的不可变、可入队定义。

    该类型保留领域可读的任务结构；真正入队时由业务适配器将其编码为运行时
    私有的不可变 ``WorkItem``，因此它不需要继承队列内部模型。
    """

    task_id: str
    spec: MemoryGenerationTaskSpec

    @property
    def topic_id(self) -> str:
        return self.spec.topic_id

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def source(self) -> MemoryGenerationSource:
        return self.spec.source

    @property
    def intent_id(self) -> str | None:
        return self.spec.intent_id

    @property
    def pending_alias(self) -> str | None:
        return self.spec.pending_alias


class MemoryGenerationResult(BaseModel):
    """由 Patchouli 管理的记忆生成执行结果视图。"""

    intent_id: str | None = None
    pending_alias: str | None = None
    atom: Any | None = None
    canonical_alias: str | None = None
    canonical_uuid: str | None = None
    duplicate_decision: DuplicateDecision | None = None
    memory_before_snapshot: Any | None = None
    changelog: str | None = None
    settlement: PendingAtomSettlement | None = None
    message: str | None = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


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
    def from_work(
        cls,
        work: MemoryGenerationWork,
        *,
        created_at: datetime,
    ) -> MemoryGenerationTask:
        """从不可变工作定义创建任务接纳快照。"""

        return cls(
            task_id=work.task_id,
            topic_id=work.topic_id,
            label=work.label,
            source=work.source,
            pending_alias=work.pending_alias,
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
                if result.pending_alias == pending_alias
                or (
                    result.settlement is not None
                    and result.settlement.pending_alias == pending_alias
                )
            )
            if matched:
                candidates = matched
        for result in candidates:
            if result.settlement is not None and result.settlement.canonical_alias:
                return result.settlement.canonical_alias
            if result.canonical_alias:
                return result.canonical_alias
            if result.atom is not None:
                get_alias = getattr(result.atom, "get_alias", None)
                if callable(get_alias) and (alias := get_alias()):
                    return alias
        return None

    @property
    def cancelled(self) -> bool:
        """判断任务是否已收到取消请求或已经进入取消终态。"""

        return (
            self.cancel_requested
            or self.status == MemoryGenerationTaskStatus.CANCELLED
        )


@dataclass(frozen=True)
class MemoryGenerationTaskWaitResult:
    """等待单个记忆生成任务时返回的结果快照。"""

    task_id: str
    found: bool
    timed_out: bool = False
    status: MemoryGenerationTaskStatus | None = None
    canonical_alias: str | None = None
    error: str | None = None

    @classmethod
    def from_task(
        cls,
        memory_task: MemoryGenerationTask,
        *,
        timed_out: bool = False,
    ) -> MemoryGenerationTaskWaitResult:
        """从已找到的领域快照构造等待结果。"""

        return cls(
            task_id=memory_task.task_id,
            found=True,
            timed_out=timed_out,
            status=memory_task.status,
            canonical_alias=memory_task.canonical_alias,
            error=memory_task.error,
        )

    @classmethod
    def not_found(cls, task_id: str) -> MemoryGenerationTaskWaitResult:
        """构造任务不存在时的等待结果。"""

        return cls(task_id=task_id, found=False)


@dataclass(frozen=True)
class MemoryGenerationTaskWaitSummary:
    """等待多个记忆生成任务时返回的聚合结果。"""

    requested: int
    found: int
    missing: int
    completed: int
    failed: int
    cancelled: int
    pending: int
    running: int
    timed_out: int
    results: tuple[MemoryGenerationTaskWaitResult, ...]

    @classmethod
    def from_results(
        cls,
        results: list[MemoryGenerationTaskWaitResult],
    ) -> MemoryGenerationTaskWaitSummary:
        """按等待结果的领域状态生成聚合计数。"""

        found = [result for result in results if result.found]
        return cls(
            requested=len(results),
            found=len(found),
            missing=sum(1 for result in results if not result.found),
            completed=sum(
                result.status == MemoryGenerationTaskStatus.COMPLETED
                for result in found
            ),
            failed=sum(
                result.status == MemoryGenerationTaskStatus.FAILED
                for result in found
            ),
            cancelled=sum(
                result.status == MemoryGenerationTaskStatus.CANCELLED
                for result in found
            ),
            pending=sum(
                result.status == MemoryGenerationTaskStatus.PENDING
                for result in found
            ),
            running=sum(
                result.status == MemoryGenerationTaskStatus.RUNNING
                for result in found
            ),
            timed_out=sum(result.timed_out for result in results),
            results=tuple(results),
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
    "MemoryGenerationTaskWaitResult",
    "MemoryGenerationTaskWaitSummary",
    "MemoryGenerationWork",
    "memory_task_to_payload",
]
