from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class ModelReadinessService:
    """Patchouli model readiness public API."""

    def __init__(self, bus: "PatchouliBus") -> None:
        # 模型就绪 public API 只编排 runtime primitive，不直接依赖 PatchouliRuntime。
        self._bus = bus

    async def warmup_models(self) -> None:
        await self._bus.request(PatchouliLocalRoutes.RUNTIME_MODELS_WARMUP)

    async def is_models_ready(self) -> bool:
        return await self._bus.request(PatchouliLocalRoutes.RUNTIME_MODELS_READY)
