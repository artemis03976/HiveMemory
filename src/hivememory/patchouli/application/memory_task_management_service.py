from __future__ import annotations

from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask
from hivememory.patchouli.services.memory_generation_tasks import MemoryGenerationTaskController


class MemoryTaskManagementService:
    """Patchouli application service for public memory task APIs."""

    def __init__(self, *, task_controller: MemoryGenerationTaskController) -> None:
        self._task_controller = task_controller

    async def list_memory_tasks(self) -> list[MemoryGenerationTask]:
        return self._task_controller.list_tasks()

    async def get_memory_task(self, task_id: str) -> MemoryGenerationTask | None:
        return self._task_controller.get_task(task_id)

    async def cancel_memory_task(self, task_id: str) -> bool:
        return self._task_controller.cancel_task(task_id)


__all__ = ["MemoryTaskManagementService"]
