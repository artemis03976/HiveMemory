"""Alice 跨系统总线桥接。

负责把 Alice 本地能力桥接到系统级总线：注册公开路由、在本地总线上代理
Patchouli 公开能力（见 docs/alice/orchestration.md §1）。
"""

from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.bus import AliceBus
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.alice.application import AgentRunService
    from hivememory.alice.runtime.core import AliceRuntime


@dataclass(frozen=True)
class AlicePublicApi:
    """由 AliceBridge 挂载的 Alice 公开 API 面。"""

    agent: AgentRunService


class AliceBridge:
    """把 Alice 本地能力桥接到系统级总线。

    职责：
        - 公开路由：将 Alice 的统一执行入口 run_agent 挂载到全局总线
        - 路由代理：在本地总线上挂载 Patchouli 公开路由代理（本地请求转发到全局总线）
    """

    #: 本地总线上代理的 Patchouli 公开路由
    _PROXY_ROUTES = (
        GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
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

    @property
    def public_routes_registered(self) -> bool:
        return self._public_routes_registered

    def mount(self) -> None:
        if self._global_bus is None:
            return

        if not self._route_proxies_registered:
            self._register_route_proxies()
            self._route_proxies_registered = True

        if not self._public_routes_registered:
            self._register_public_routes()
            self._public_routes_registered = True

    def unmount(self) -> None:
        if self._global_bus is None:
            return

        if self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False

        if self._route_proxies_registered:
            self._unregister_route_proxies()
            self._route_proxies_registered = False

    # ========== 公开路由（Alice → 全局总线） ==========

    def _register_public_routes(self) -> None:
        if self._global_bus is None:
            return
        self._global_bus.register(
            AliceRoutes.RUN_AGENT,
            self._run_agent_route,
        )

    def _unregister_public_routes(self) -> None:
        if self._global_bus is None:
            return
        self._global_bus.unregister(AliceRoutes.RUN_AGENT)

    async def _run_agent_route(self, *args: Any, **kwargs: Any) -> Any:
        """统一执行路由 handler：非流式等待执行结果，流式返回事件生成器对象。"""
        call = self._public_api.agent.run_agent(*args, **kwargs)
        if isinstance(call, Coroutine):
            return await call
        return call

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


__all__ = ["AliceBridge", "AlicePublicApi"]
