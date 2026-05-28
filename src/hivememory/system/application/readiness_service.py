from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class SystemReadinessService:
    """System-level model readiness service."""

    def __init__(self, global_bus: "GlobalSystemBus") -> None:
        self._global_bus = global_bus

    async def warmup_models(self) -> None:
        await self._global_bus.request(GlobalRoutes.PATCHOULI_WARMUP_MODELS)

    async def is_models_ready(self) -> bool:
        return await self._global_bus.request(GlobalRoutes.PATCHOULI_MODELS_READY)

    async def readiness(self) -> dict[str, bool | str]:
        ready = await self.is_models_ready()
        return {
            "status": "ready" if ready else "warming_up",
            "models_ready": ready,
        }
