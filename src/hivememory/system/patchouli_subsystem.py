from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus
    from hivememory.patchouli.system import PatchouliSystem
    from hivememory.system.runtime.bridge import SubsystemBridge


class PatchouliSubsystemAdapter:
    """将 PatchouliSystem 以子系统形式接入顶层 registry。"""

    name = "patchouli"

    def __init__(
        self,
        patchouli: PatchouliSystem,
        local_bus: PatchouliBus | None = None,
        bridge: SubsystemBridge | None = None,
    ) -> None:
        self._patchouli = patchouli
        self._local_bus = local_bus
        self._bridge = bridge
        self._local_routes_registered = False
        self._bridge_mounted = False

    async def start(self) -> None:
        if self._local_bus and not self._local_routes_registered:
            self._register_local_routes()
            self._local_routes_registered = True
        if self._bridge and not self._bridge_mounted:
            self._bridge.mount()
            self._bridge_mounted = True
        self._patchouli.start_scheduler()

    async def stop(self) -> None:
        await self._patchouli.shutdown_drain()
        if self._bridge and self._bridge_mounted:
            self._bridge.unmount()
            self._bridge_mounted = False
        if self._local_bus and self._local_routes_registered:
            self._unregister_local_routes()
            self._local_routes_registered = False

    async def health(self) -> dict[str, Any]:
        models_ready = self._patchouli.kernel.is_models_ready()
        return {
            "status": "ok" if models_ready else "warming_up",
            "models_ready": models_ready,
        }

    def _register_local_routes(self) -> None:
        self._local_bus.register(
            "kernel.submit_interaction",
            self._patchouli.kernel.submit_interaction,
        )
        self._local_bus.register(
            "passive.analyze_and_retrieve",
            self._patchouli.analyze_and_retrieve,
        )

    def _unregister_local_routes(self) -> None:
        self._local_bus.unregister("kernel.submit_interaction")
        self._local_bus.unregister("passive.analyze_and_retrieve")
