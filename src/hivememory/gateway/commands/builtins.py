from __future__ import annotations

from hivememory.gateway.commands.models import (
    CommandCategory,
    CommandDefinition,
    CommandPermissionPolicy,
    CommandRouteTarget,
    CommandRouteTargetKind,
)
from hivememory.gateway.commands.registry import CommandRegistry


def register_builtin_commands(
    registry: CommandRegistry,
    builtin_overrides: dict[str, bool] | None = None,
) -> CommandRegistry:
    """向指定 registry 注册内置系统指令。"""

    for definition in _builtin_definitions():
        if builtin_overrides and builtin_overrides.get(definition.command_id) is False:
            continue
        registry.register(definition)
    return registry


def create_builtin_command_registry(
    builtin_overrides: dict[str, bool] | None = None,
) -> CommandRegistry:
    """创建包含内置系统指令的默认 registry。"""

    return register_builtin_commands(
        CommandRegistry(),
        builtin_overrides=builtin_overrides,
    )


def _builtin_definitions() -> list[CommandDefinition]:
    """内置指令只声明能力与路由目标，不在 registry 阶段执行副作用。"""

    public = CommandPermissionPolicy(visibility="public")
    return [
        CommandDefinition(
            command_id="system.help",
            category=CommandCategory.SYSTEM,
            primary_name="/help",
            aliases=["/start"],
            summary="显示可用系统指令。",
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.LOCAL_HANDLER,
                name="system.help",
            ),
            permission=public,
            priority=10,
        ),
        CommandDefinition(
            command_id="system.commands",
            category=CommandCategory.SYSTEM,
            primary_name="/commands",
            summary="列出系统指令。",
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.LOCAL_HANDLER,
                name="system.commands",
            ),
            permission=public,
            priority=20,
        ),
        CommandDefinition(
            command_id="system.clear",
            category=CommandCategory.SYSTEM,
            primary_name="/clear",
            aliases=["/reset", "/restart"],
            summary="清空当前聊天客户端状态。",
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.CLIENT_ACTION,
                name="clear_chat",
            ),
            permission=public,
            priority=30,
        ),
        CommandDefinition(
            command_id="runtime.status",
            category=CommandCategory.RUNTIME,
            primary_name="/status",
            summary="显示运行时状态。",
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.LOCAL_HANDLER,
                name="runtime.status",
            ),
            permission=public,
            priority=40,
        ),
    ]


__all__ = [
    "create_builtin_command_registry",
    "register_builtin_commands",
]
