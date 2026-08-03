"""Alice cross-system bus bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.bus import AliceBus
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

if TYPE_CHECKING:
    from hivememory.alice.application import AgentRunService
    from hivememory.alice.runtime.core import AliceRuntime


@dataclass(frozen=True)
class AlicePublicApi:
    """Public Alice API surface mounted by AliceBridge."""

    agent: AgentRunService


class AliceBridge:
    """Bridge Alice local capabilities to system-level buses.

    职责：
        - 公开路由：将 Alice 的 run_agent / run_agent_stream 挂载到全局总线
        - 路由代理：在本地总线上挂载 Patchouli 公开路由代理（本地请求转发到全局总线）
        - 事件桥接：订阅全局 PendingAtom 事件并转发给运行时处理器
    """

    #: 本地总线上代理的 Patchouli 公开路由
    _PROXY_ROUTES = (
        GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
        GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
        GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
    )

    def __init__(
        self,
        *,
        local_bus: AliceBus | None = None,
        runtime: AliceRuntime,
        public_api: AlicePublicApi,
        global_bus: GlobalSystemBus | None = None,
    ) -> None:
        if local_bus is None:
            raise ValueError("AliceBridge requires an AliceBus")
        self._local_bus = local_bus
        self._runtime = runtime
        self._public_api = public_api
        self._global_bus = global_bus
        self._public_routes_registered = False
        self._route_proxies_registered = False
        self._global_events_registered = False

    @property
    def public_routes_registered(self) -> bool:
        return self._public_routes_registered

    @property
    def route_proxies_registered(self) -> bool:
        return self._route_proxies_registered

    @property
    def global_events_registered(self) -> bool:
        return self._global_events_registered

    def mount(self) -> None:
        if self._global_bus is None:
            return

        if not self._route_proxies_registered:
            self._register_route_proxies()
            self._route_proxies_registered = True

        if not self._global_events_registered:
            self._register_global_event_bridges()
            self._global_events_registered = True

        if not self._public_routes_registered:
            self._register_public_routes()
            self._public_routes_registered = True

    def unmount(self) -> None:
        if self._global_bus is None:
            return

        if self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False

        if self._global_events_registered:
            self._unregister_global_event_bridges()
            self._global_events_registered = False

        if self._route_proxies_registered:
            self._unregister_route_proxies()
            self._route_proxies_registered = False

    # ========== 公开路由（Alice → 全局总线） ==========

    def _register_public_routes(self) -> None:
        if self._global_bus is None:
            return
        self._global_bus.register(
            AliceRoutes.RUN_AGENT,
            self._public_api.agent.run_agent,
        )
        self._global_bus.register(
            AliceRoutes.RUN_AGENT_STREAM,
            self._run_agent_stream_route,
        )

    def _unregister_public_routes(self) -> None:
        if self._global_bus is None:
            return
        self._global_bus.unregister(AliceRoutes.RUN_AGENT)
        self._global_bus.unregister(AliceRoutes.RUN_AGENT_STREAM)

    async def _run_agent_stream_route(self, *args: Any, **kwargs: Any) -> Any:
        """为 AsyncSystemBus 适配流式 handler，返回 async generator 对象。"""
        return self._public_api.agent.run_agent_stream(*args, **kwargs)

    # ========== 路由代理（本地总线 → 全局总线） ==========

    def _register_route_proxies(self) -> None:
        for route in self._PROXY_ROUTES:
            self._local_bus.register(route, self._make_route_proxy(route))

    def _unregister_route_proxies(self) -> None:
        for route in self._PROXY_ROUTES:
            self._local_bus.unregister(route)

    def _make_route_proxy(self, route: str):
        async def _proxy(*args: Any, **kwargs: Any) -> Any:
            if self._global_bus is None:
                raise KeyError(route)
            return await self._global_bus.request(route, *args, **kwargs)

        return _proxy

    # ========== 全局事件桥接（全局总线 → 运行时） ==========

    def _global_event_bindings(self) -> list[tuple[str, Any]]:
        return [
            (
                GlobalEvents.PENDING_ATOM_SETTLED,
                self._runtime.on_pending_atom_settled,
            ),
            (
                GlobalEvents.PENDING_ATOM_FAILED,
                self._runtime.on_pending_atom_failed,
            ),
            (
                GlobalEvents.PENDING_ATOM_CANCELLED,
                self._runtime.on_pending_atom_cancelled,
            ),
        ]

    def _register_global_event_bridges(self) -> None:
        if self._global_bus is None:
            return
        for event, handler in self._global_event_bindings():
            self._global_bus.subscribe(event, handler)

    def _unregister_global_event_bridges(self) -> None:
        if self._global_bus is None:
            return
        for event, handler in self._global_event_bindings():
            self._global_bus.unsubscribe(event, handler)


__all__ = ["AliceBridge", "AlicePublicApi"]
