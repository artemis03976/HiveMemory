"""记忆生成任务的可观测事件发布。"""

from __future__ import annotations

from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
    memory_task_to_payload,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.publisher import RuntimeEventPublisher

_TERMINAL_EVENT_TYPES = {
    MemoryGenerationTaskStatus.COMPLETED: RuntimeEventType.MEMORY_TASK_COMPLETED,
    MemoryGenerationTaskStatus.CANCELLED: RuntimeEventType.MEMORY_TASK_CANCELLED,
    MemoryGenerationTaskStatus.FAILED: RuntimeEventType.MEMORY_TASK_FAILED,
}


class MemoryTaskEventEmitter:
    """集中投影并发布 ``memory.task.*`` 可观测事件。

    Emitter 不修改任务状态，也不发布 ``PendingAtom`` 功能事件；任务生命周期的
    发生时机仍由 Controller 显式决定。
    """

    def __init__(self, publisher: RuntimeEventPublisher) -> None:
        self._publisher = publisher.scoped(
            subsystem="patchouli",
            source="patchouli.memory_generation",
            component="memory_task",
        )

    def created(self, snapshot: MemoryGenerationTask) -> None:
        """发布任务已创建事件，使前端建立对应的任务视图。"""

        self._emit(
            RuntimeEventType.MEMORY_TASK_CREATED,
            snapshot,
            message="Memory generation task created.",
        )

    def running(self, snapshot: MemoryGenerationTask) -> None:
        """发布任务已进入运行状态的事件。"""

        self._emit(RuntimeEventType.MEMORY_TASK_STATUS, snapshot)

    def terminal(
        self,
        snapshot: MemoryGenerationTask,
        *,
        reason: str | None = None,
    ) -> None:
        """按领域终态发布完成、取消或失败事件。"""

        event_type = _TERMINAL_EVENT_TYPES.get(snapshot.status)
        if event_type is None:
            raise ValueError(
                f"Memory task status is not terminal: {snapshot.status.value}"
            )
        self._emit(event_type, snapshot, reason=reason)

    def cancel_requested(
        self,
        snapshot: MemoryGenerationTask,
        *,
        reason: str,
    ) -> None:
        """发布任务取消请求已被接纳的事件。"""

        self._emit(
            RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED,
            snapshot,
            reason=reason,
        )

    def _emit(
        self,
        event_type: RuntimeEventType,
        snapshot: MemoryGenerationTask,
        *,
        reason: str | None = None,
        message: str | None = None,
    ) -> None:
        self._publisher.bind(
            task_type="background",
            task_id=snapshot.task_id,
            topic_id=snapshot.topic_id,
        ).emit(
            event_type,
            status=snapshot.status.value,
            severity=(
                "error"
                if event_type == RuntimeEventType.MEMORY_TASK_FAILED
                else "info"
            ),
            reason=reason,
            message=message,
            data=memory_task_to_payload(snapshot, reason=reason),
        )


__all__ = ["MemoryTaskEventEmitter"]
