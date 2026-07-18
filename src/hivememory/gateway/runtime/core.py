"""GatewayRuntime：Gateway 子系统运行时聚合根。"""

from __future__ import annotations

from typing import Any

from hivememory.engines.gateway.interceptors import create_interceptor
from hivememory.engines.gateway.topic_router import TopicRouterEngine
from hivememory.gateway.analysis import UserQueryAnalysisResolver
from hivememory.gateway.commands import (
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.gateway.context import GatewayContextProvider
from hivememory.gateway.contracts.local_routes import GatewayLocalRoutes
from hivememory.gateway.runtime.bus import GatewayBus
from hivememory.gateway.runtime.route_bindings import build_gateway_route_bindings
from hivememory.gateway.workflow import GatewayWorkflow
from hivememory.gateway.workflow.topology import build_gateway_workflow
from hivememory.system.config import SystemGatewayConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink


class GatewayRuntime:
    """
    Gateway 子系统运行时。

    Runtime 只持有 Gateway 本地总线、workflow 及其运行期依赖。
    Engine、Provider、Resolver 的具体装配由后续阶段逐步补齐。
    """

    def __init__(
        self,
        *,
        config: SystemGatewayConfig,
        global_bus: GlobalSystemBus,
        runtime_events: RuntimeEventSink | None = None,
        workflow: GatewayWorkflow | None = None,
        context_provider: GatewayContextProvider | None = None,
        topic_router: TopicRouterEngine | None = None,
        analysis_resolver: UserQueryAnalysisResolver | None = None,
    ) -> None:
        self.config = config
        self.global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        self._local_bus = GatewayBus()
        self._local_routes_registered = False

        registry = create_builtin_command_registry(config.commands.builtin)
        interceptor = create_interceptor(config.interceptor, registry)
        command_dispatcher = SystemCommandDispatcher(
            registry,
            global_bus=global_bus,
            debug_enabled=config.commands.enable_debug_commands,
            expose_listing=config.commands.expose_listing,
        )
        self.workflow = workflow or build_gateway_workflow(
            interceptor=interceptor,
            command_dispatcher=command_dispatcher,
            context_provider=context_provider,
            topic_router=topic_router,
            analysis_resolver=analysis_resolver,
            context_config=config.context_preparation,
            topic_router_config=config.topic_router,
            analysis_config=config.user_query_analysis,
            runtime_events=self._runtime_events,
        )

    @property
    def local_bus(self) -> GatewayBus:
        return self._local_bus

    @property
    def local_routes_registered(self) -> bool:
        return self._local_routes_registered

    def mount_local_routes(self, service: Any) -> None:
        """幂等挂载 Gateway 本地路由。"""

        if self._local_routes_registered:
            return

        for route, handler in build_gateway_route_bindings(service):
            self._local_bus.register(route, handler)
        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        """幂等卸载 Gateway 本地路由。"""

        if not self._local_routes_registered:
            return

        for route in GatewayLocalRoutes.ALL:
            self._local_bus.unregister(route)
        self._local_routes_registered = False

    def health(self) -> dict[str, Any]:
        """返回 Gateway runtime 的最小健康状态。"""

        return {
            "local_routes_registered": self._local_routes_registered,
            "workflow_ready": self.workflow is not None,
        }


__all__ = ["GatewayRuntime"]
