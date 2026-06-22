from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class MemoryTaskManagementService:
    """Patchouli application service for public memory task APIs."""

    def __init__(self, *, bus: "PatchouliBus") -> None:
        # Public use-case 层只通过 local bus 访问任务控制面，避免直接持有 controller。
        self._bus = bus

    async def list_memory_tasks(self) -> list[MemoryGenerationTask]:
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_LIST)

    async def get_memory_task(self, task_id: str) -> MemoryGenerationTask | None:
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_GET, task_id)

    async def cancel_memory_task(self, task_id: str) -> bool:
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, task_id)


__all__ = ["MemoryTaskManagementService"]
