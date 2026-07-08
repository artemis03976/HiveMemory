"""GatewaySystem 装配工厂。"""

from __future__ import annotations

from hivememory.gateway.commands import (
    CommandRegistry,
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.system import GatewaySystem
from hivememory.system.config import LLMConfig, SystemGatewayConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


def build_gateway_system(
    config: SystemGatewayConfig,
    llm_config: LLMConfig | None = None,
    *,
    global_bus: GlobalSystemBus | None = None,
    command_registry: CommandRegistry | None = None,
) -> GatewaySystem:
    """
    构造 Phase 3 Gateway 子系统。

    llm_config 保留为后续 engine 原语装配入口；Phase 3A 不再构造旧
    GatewayEngine 主路径。
    """

    _ = llm_config
    command_config = config.commands
    active_command_registry = (
        command_registry
        if command_registry is not None
        else (
            create_builtin_command_registry(command_config.builtin)
            if command_config.enabled
            else None
        )
    )
    command_dispatcher = (
        SystemCommandDispatcher(
            active_command_registry,
            global_bus=global_bus,
            debug_enabled=command_config.enable_debug_commands,
            expose_listing=command_config.expose_listing,
        )
        if active_command_registry is not None
        else None
    )

    runtime = GatewayRuntime(
        command_registry=active_command_registry,
        command_dispatcher=command_dispatcher,
    )
    return GatewaySystem(runtime=runtime, global_bus=global_bus)


__all__ = ["build_gateway_system"]
