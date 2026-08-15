"""Application facade for Patchouli-owned memory generation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class MemoryTaskApplicationService:
    """
    Top-level API facade for memory task query and cancellation.

    MemoryGenerationTask 的生命周期仍由 Patchouli 拥有；顶层 service 只通过
    GlobalSystemBus 请求 Patchouli 公开 API，保持 system/application service 与
    其它子系统能力访问方式一致。
    """

    def __init__(self, global_bus: "GlobalSystemBus") -> None:
        self._global_bus = global_bus

    async def list_memory_tasks(self) -> list[MemoryGenerationTask]:
        return await self._global_bus.request(GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST)

    async def get_memory_task(self, task_id: str) -> MemoryGenerationTask | None:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
            task_id,
        )

    async def cancel_memory_task(self, task_id: str) -> bool:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            task_id,
        )


__all__ = ["MemoryTaskApplicationService"]
