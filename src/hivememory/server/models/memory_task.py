"""记忆任务 API 模型。"""

from __future__ import annotations

from pydantic import BaseModel

from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)


class MemoryTaskResponse(BaseModel):
    """面向 ``MemoryGenerationTask`` REST API 的稳定对外投影。"""

    task_id: str
    topic_id: str
    label: str
    source: str
    pending_alias: str | None = None
    status: str
    canonical_alias: str | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    cancelled: bool = False
    reason: str | None = None

    @classmethod
    def from_domain(
        cls,
        memory_task: MemoryGenerationTask,
        *,
        reason: str | None = None,
    ) -> MemoryTaskResponse:
        """从不可变领域快照构造稳定的 REST 响应。"""

        cancel_requested = memory_task.cancel_requested
        cancelled = memory_task.status == MemoryGenerationTaskStatus.CANCELLED
        return cls(
            task_id=memory_task.task_id,
            topic_id=memory_task.topic_id,
            label=memory_task.label,
            source=memory_task.source.value,
            pending_alias=memory_task.pending_alias,
            status=memory_task.status.value,
            canonical_alias=memory_task.canonical_alias,
            error=memory_task.error,
            created_at=memory_task.created_at.isoformat(),
            started_at=(
                memory_task.started_at.isoformat()
                if memory_task.started_at is not None
                else None
            ),
            finished_at=(
                memory_task.finished_at.isoformat()
                if memory_task.finished_at is not None
                else None
            ),
            cancel_requested=cancel_requested,
            cancelled=cancelled,
            reason=reason or memory_task.cancel_reason,
        )


class MemoryTaskListResponse(BaseModel):
    """记忆任务列表响应。"""

    tasks: list[MemoryTaskResponse]


__all__ = ["MemoryTaskListResponse", "MemoryTaskResponse"]
