"""Patchouli 记忆生成控制面模型与任务快照投影。

对外只读快照 ``MemoryGenerationTask`` 及其状态/来源枚举定义在公共契约
``patchouli.contracts.memory_tasks``（A2 §8.2），此处转发导出以兼容既有
引用；本模块保留控制面与数据面共享的输入/结果模型，以及由通用 work queue
结果投影任务快照的工厂函数。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from hivememory.components.work_queue import (
    TaskOutcome,
    WorkState,
)
from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    PendingAtomSettlement,
    TopicAssetBinding,
    WorkspaceIdentity,
)
from hivememory.engines.generation.models import GenerationRequest
from hivememory.patchouli.contracts.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)

_WORK_STATE_TO_TASK_STATUS = {
    WorkState.QUEUED: MemoryGenerationTaskStatus.PENDING,
    WorkState.RETRY_WAIT: MemoryGenerationTaskStatus.PENDING,
    WorkState.RUNNING: MemoryGenerationTaskStatus.RUNNING,
    WorkState.SUCCEEDED: MemoryGenerationTaskStatus.COMPLETED,
    WorkState.CANCELLED: MemoryGenerationTaskStatus.CANCELLED,
    WorkState.FAILED: MemoryGenerationTaskStatus.FAILED,
    WorkState.DEAD_LETTER: MemoryGenerationTaskStatus.FAILED,
}


@dataclass(frozen=True)
class InteractionArtifactInput:
    """传递给记忆生成数据面的原始交互数据。"""

    topic_id: str
    topic_title: str = ""
    topic_summary: str = ""
    blocks: tuple[LogicalBlock, ...] = ()
    # settle 前冻结的 Topic 真实使用资产关系；进入 queue 后不再依赖短期 buffer 实体。
    asset_bindings: tuple[TopicAssetBinding, ...] = ()


@dataclass(frozen=True)
class MemoryGenerationTaskSpec:
    """记忆生成控制面与数据面共享的规范化输入。

    ``belong_to`` 与 ``from_actor`` 分别保存归属与发起者；来源记录由
    Patchouli 内部生成链维护，不在任务规范中重复保存。
    """

    belong_to: WorkspaceIdentity
    from_actor: ActorIdentity
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


def memory_task_from_spec(
    task_id: str,
    spec: MemoryGenerationTaskSpec,
    *,
    created_at: datetime,
) -> MemoryGenerationTask:
    """从已接纳的任务规范创建对外初始快照。"""

    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=spec.topic_id,
        label=spec.label,
        source=spec.source,
        pending_alias=spec.pending_alias,
        belong_to=spec.belong_to,
        from_actor=spec.from_actor,
        created_at=created_at,
    )


def memory_task_from_outcome(
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
            _select_canonical_alias(
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
            if result.settlement is not None and result.settlement.pending_alias == pending_alias
        )
        if matched:
            candidates = matched
    for result in candidates:
        if result.settlement is not None and result.settlement.canonical_alias:
            return result.settlement.canonical_alias
        if result.canonical_alias:
            return result.canonical_alias
    return None


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
            memory_task.started_at.isoformat() if memory_task.started_at is not None else None
        ),
        "finished_at": (
            memory_task.finished_at.isoformat() if memory_task.finished_at is not None else None
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
    "memory_task_from_outcome",
    "memory_task_from_spec",
    "memory_task_to_payload",
]
