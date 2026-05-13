from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hivememory.patchouli.system import PatchouliSystem


class PatchouliSubsystemAdapter:
    """将 PatchouliSystem 以子系统形式接入顶层 registry。"""

    name = "patchouli"

    def __init__(self, patchouli: PatchouliSystem) -> None:
        self._patchouli = patchouli

    async def start(self) -> None:
        self._patchouli.start_scheduler()

    async def stop(self) -> None:
        await self._patchouli.shutdown_drain()

    async def health(self) -> dict[str, Any]:
        models_ready = self._patchouli.kernel.is_models_ready()
        return {
            "status": "ok" if models_ready else "warming_up",
            "models_ready": models_ready,
        }
