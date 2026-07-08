"""GatewayRuntime：Gateway 子系统运行时聚合根。"""

from __future__ import annotations

from typing import Any

from hivememory.engines.gateway.context_router import ContextRouterEngine
from hivememory.engines.gateway.intent_classifier import IntentClassifierEngine
from hivememory.engines.gateway.memory_value_judge import MemoryValueJudgeEngine
from hivememory.engines.gateway.retrieval_strategy import RetrievalStrategyEngine
from hivememory.gateway.commands import (
    CommandRegistry,
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.gateway.context import GatewayContextBuilder
from hivememory.gateway.contracts.local_routes import GatewayLocalRoutes
from hivememory.gateway.pipeline import GatewayPipeline
from hivememory.gateway.runtime.bus import GatewayBus
from hivememory.gateway.runtime.route_bindings import build_gateway_route_bindings
from hivememory.gateway.stages.s0_command import CommandInterceptorStage
from hivememory.system.config import SystemGatewayConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink


class GatewayRuntime:
    """
    Gateway 子系统运行时。

    Phase 3B 固定装配边界：本地总线、Context Hydration、空 Pipeline、
    command interceptor 与轻量 engine 原语。完整 S1-S5 装配留给后续阶段。
    """

    def __init__(
        self,
        *,
        config: SystemGatewayConfig,
        global_bus: GlobalSystemBus | None = None,
        runtime_events: RuntimeEventSink | None = None,
        context_builder: GatewayContextBuilder | None = None,
        pipeline: GatewayPipeline | None = None,
    ) -> None:
        self.config = config
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        self._local_bus = GatewayBus()
        self._local_routes_registered = False

        self.context_builder = context_builder or GatewayContextBuilder()
        self.pipeline = pipeline or GatewayPipeline()

        self.command_registry = self._build_command_registry(config)
        self.command_dispatcher = self._build_command_dispatcher(
            config,
            global_bus=global_bus,
        )
        self.command_interceptor = CommandInterceptorStage(self.command_registry)

        # Phase 3B 只持有原语实例，不把它们接入空 Pipeline。
        self.intent_classifier = IntentClassifierEngine()
        self.context_router = ContextRouterEngine()
        self.memory_value_judge = MemoryValueJudgeEngine()
        self.retrieval_strategy = RetrievalStrategyEngine()

    def _build_command_registry(
        self,
        config: SystemGatewayConfig,
    ) -> CommandRegistry | None:
        """根据 Gateway 配置构造 S0 指令注册表。"""

        command_config = config.commands
        if not command_config.enabled:
            return None
        return create_builtin_command_registry(command_config.builtin)

    def _build_command_dispatcher(
        self,
        config: SystemGatewayConfig,
        *,
        global_bus: GlobalSystemBus | None,
    ) -> SystemCommandDispatcher | None:
        """根据 Gateway 配置构造 S0 指令执行器。"""

        if self.command_registry is None:
            return None

        command_config = config.commands
        return SystemCommandDispatcher(
            self.command_registry,
            global_bus=global_bus,
            debug_enabled=command_config.enable_debug_commands,
            expose_listing=command_config.expose_listing,
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
            "pipeline_stage_count": len(self.pipeline.stages),
            "command_registry_enabled": self.command_registry is not None,
            "command_dispatcher_enabled": self.command_dispatcher is not None,
            "engines": {
                "intent_classifier": self.intent_classifier is not None,
                "context_router": self.context_router is not None,
                "memory_value_judge": self.memory_value_judge is not None,
                "retrieval_strategy": self.retrieval_strategy is not None,
            },
        }


__all__ = ["GatewayRuntime"]
