import logging
from typing import Any, Awaitable, Callable

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.bus import AliceBus
from hivememory.system.runtime.bus.bridge import SubsystemBridge
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)


class AliceBridge(SubsystemBridge):
    """Alice 子系统桥接器 — 将 Alice 公开能力挂载到全局总线。"""

    ROUTE_MAP: dict[str, str] = {
        AliceRoutes.RUN_AGENT: AliceRoutes.RUN_AGENT,
    }

    def __init__(self, local_bus: AliceBus, global_bus: GlobalSystemBus) -> None:
        self._local = local_bus
        self._global = global_bus
        self._mounted_routes: list[str] = []

    def mount_public_routes(self) -> None:
        for global_route, local_route in self.ROUTE_MAP.items():
            forwarder = self._make_forwarder(local_route)
            self._global.register(global_route, forwarder)
            self._mounted_routes.append(global_route)
            logger.debug(f"AliceBridge: mounted {global_route}")

    def mount_event_bridges(self) -> None:
        pass

    def unmount(self) -> None:
        for route in self._mounted_routes:
            self._global.unregister(route)
        self._mounted_routes.clear()

    def _make_forwarder(self, local_route: str) -> Callable[..., Awaitable[Any]]:
        async def _forward(*args: Any, **kwargs: Any) -> Any:
            return await self._local.request(local_route, *args, **kwargs)
        return _forward
