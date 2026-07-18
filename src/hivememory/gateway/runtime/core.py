"""GatewayRuntime：Gateway 子系统运行时聚合根。"""

from __future__ import annotations

from typing import Any

from hivememory.gateway.contracts.local_routes import GatewayLocalRoutes
from hivememory.gateway.runtime.bus import GatewayBus
from hivememory.gateway.runtime.route_bindings import build_gateway_route_bindings
from hivememory.gateway.workflow import GatewayWorkflow
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
    ) -> None:
        self.config = config
        self.global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        self._local_bus = GatewayBus()
        self._local_routes_registered = False

        self.workflow = workflow or GatewayWorkflow(
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
