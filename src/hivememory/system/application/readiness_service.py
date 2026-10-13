from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus


class SystemReadinessService:
    """系统级模型就绪服务。"""

    def __init__(self, global_bus: GlobalSystemBus) -> None:
        self._global_bus = global_bus

    async def warmup_models(self) -> None:
        await self._global_bus.request(PatchouliRoutes.WARMUP_MODELS)

    async def is_models_ready(self) -> bool:
        return await self._global_bus.request(PatchouliRoutes.MODELS_READY)

    async def readiness(self) -> dict[str, bool | str]:
        ready = await self.is_models_ready()
        return {
            "status": "ready" if ready else "warming_up",
            "models_ready": ready,
        }
