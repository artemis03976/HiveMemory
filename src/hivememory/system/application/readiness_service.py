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

    async def readiness(self) -> dict[str, bool | str | list[str]]:
        """模型就绪状态：已就绪、仍在加载，或有模型加载失败（带失败的模型名）。

        加载失败时模型保持未就绪，下一次使用时重新加载；桌面端据此提示用户重开应用重试，
        而不是一直等待。
        """
        if await self.is_models_ready():
            return {"status": "ready", "models_ready": True}
        failed: list[str] = await self._global_bus.request(PatchouliRoutes.MODELS_FAILED)
        if failed:
            return {"status": "model_load_failed", "models_ready": False, "failed_models": failed}
        return {"status": "warming_up", "models_ready": False}
