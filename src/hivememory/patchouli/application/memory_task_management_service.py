from __future__ import annotations

from typing import Any

from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask


class MemoryTaskManagementService:
    """Patchouli application service for public memory task APIs."""

    def __init__(self, *, librarian_core: Any) -> None:
        self._librarian_core = librarian_core

    async def list_memory_tasks(self) -> list[MemoryGenerationTask]:
        return self._librarian_core.list_tasks()

    async def get_memory_task(self, task_id: str) -> MemoryGenerationTask | None:
        return self._librarian_core.get_task(task_id)

    async def cancel_memory_task(self, task_id: str) -> bool:
        return self._librarian_core.cancel_task(task_id)


__all__ = ["MemoryTaskManagementService"]
