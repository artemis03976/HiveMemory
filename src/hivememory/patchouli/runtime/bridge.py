"""
PatchouliBridge — 连接 PatchouliBus 与 GlobalSystemBus

职责:
    - 将 Patchouli 公开路由注册到 GlobalSystemBus (转发代理)
    - 订阅 PatchouliBus 域事件并转发到 GlobalSystemBus
    - unmount 时清理所有注册与订阅
"""

import logging
from typing import Any, Awaitable, Callable

from hivememory.patchouli.contracts.domain_events import PatchouliEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.system.runtime.bridge import SubsystemBridge
from hivememory.system.runtime.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)


class PatchouliBridge(SubsystemBridge):
    """Patchouli 子系统桥接器 — 将公开能力挂载到全局总线。"""

    ROUTE_MAP: dict[str, str] = {
        PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE: "passive.analyze_and_retrieve",
        PatchouliRoutes.SUBMIT_INTERACTION: "kernel.submit_interaction",
    }

    BRIDGED_EVENTS: list[str] = [
        PatchouliEvents.MEMORY_GENERATED,
        PatchouliEvents.TOPIC_EVICTED,
        PatchouliEvents.OBSERVER_SESSION_FLUSHED,
    ]

    def __init__(
        self,
        local_bus: PatchouliBus,
        global_bus: GlobalSystemBus,
    ) -> None:
        self._local = local_bus
        self._global = global_bus
        self._mounted_routes: list[str] = []
        self._mounted_event_bridges: list[tuple[str, Callable[..., Awaitable[None]]]] = []

    def mount_public_routes(self) -> None:
        for global_route, local_route in self.ROUTE_MAP.items():
            forwarder = self._make_forwarder(local_route)
            self._global.register(global_route, forwarder)
            self._mounted_routes.append(global_route)
            logger.debug(f"PatchouliBridge: mounted {global_route} -> {local_route}")

    def mount_event_bridges(self) -> None:
        for event_name in self.BRIDGED_EVENTS:
            bridge_cb = self._make_event_bridge(event_name)
            self._local.subscribe(event_name, bridge_cb)
            self._mounted_event_bridges.append((event_name, bridge_cb))

    def unmount(self) -> None:
        for route in self._mounted_routes:
            self._global.unregister(route)
        for event_name, cb in self._mounted_event_bridges:
            self._local.unsubscribe(event_name, cb)
        self._mounted_routes.clear()
        self._mounted_event_bridges.clear()

    def _make_forwarder(self, local_route: str) -> Callable[..., Awaitable[Any]]:
        async def _forward(*args: Any, **kwargs: Any) -> Any:
            return await self._local.request(local_route, *args, **kwargs)
        return _forward

    def _make_event_bridge(self, event_name: str) -> Callable[..., Awaitable[None]]:
        async def _bridge(*args: Any, **kwargs: Any) -> None:
            await self._global.publish(event_name, *args, **kwargs)
        return _bridge
