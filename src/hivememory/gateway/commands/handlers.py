from __future__ import annotations

from collections.abc import Iterable

from hivememory.core.models import ActorIdentity
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
)
from hivememory.gateway.commands.models import (
    CommandDefinition,
    CommandParseResult,
)
from hivememory.gateway.commands.registry import CommandRegistry


def handle_help(
    *,
    command: CommandParseResult,
    registry: CommandRegistry,
    identity: ActorIdentity,
    debug_enabled: bool = False,
    expose_listing: bool = True,
) -> CommandExecutionResult:
    """返回可见系统指令的简短帮助文本。"""

    visible = (
        _visible_definitions(registry, identity=identity, debug_enabled=debug_enabled)
        if expose_listing
        else []
    )
    lines = [f"{definition.primary_name} - {definition.summary}" for definition in visible]
    message = "可用系统指令：\n" + "\n".join(lines) if lines else "当前没有可用系统指令。"
    return CommandExecutionResult(
        command_id=command.command_id or "system.help",
        status=CommandExecutionStatus.COMPLETED,
        message=message,
        data={"commands": [_definition_payload(definition) for definition in visible]},
    )


def handle_commands(
    *,
    command: CommandParseResult,
    registry: CommandRegistry,
    identity: ActorIdentity,
    debug_enabled: bool = False,
    expose_listing: bool = True,
) -> CommandExecutionResult:
    """返回结构化系统指令列表。"""

    visible = (
        _visible_definitions(registry, identity=identity, debug_enabled=debug_enabled)
        if expose_listing
        else []
    )
    return CommandExecutionResult(
        command_id=command.command_id or "system.commands",
        status=CommandExecutionStatus.COMPLETED,
        message=f"共 {len(visible)} 条可用系统指令。",
        data={"commands": [_definition_payload(definition) for definition in visible]},
    )


def handle_status(
    *,
    command: CommandParseResult,
    registry: CommandRegistry,
    identity: ActorIdentity,
    debug_enabled: bool = False,
    expose_listing: bool = True,
) -> CommandExecutionResult:
    """返回 system gateway 的最小运行摘要。"""

    visible_count = len(_visible_definitions(registry, identity=identity, debug_enabled=debug_enabled))
    return CommandExecutionResult(
        command_id=command.command_id or "runtime.status",
        status=CommandExecutionStatus.COMPLETED,
        message="System Gateway 正常运行。",
        data={
            "gateway": "ok",
            "commands_visible": visible_count,
            "debug_enabled": debug_enabled,
        },
    )


def _visible_definitions(
    registry: CommandRegistry,
    *,
    identity: ActorIdentity,
    debug_enabled: bool,
) -> list[CommandDefinition]:
    return [
        definition
        for definition in registry.list(include_hidden=False)
        if _is_visible(definition, identity=identity, debug_enabled=debug_enabled)
    ]


def _is_visible(
    definition: CommandDefinition,
    *,
    identity: ActorIdentity,
    debug_enabled: bool,
) -> bool:
    permission = definition.permission
    if permission.visibility == "debug" and not debug_enabled:
        return False
    if permission.visibility == "admin" and not _in_allowlist(identity, permission.allowed_user_ids, permission.allowed_agent_ids):
        return False
    return True


def _in_allowlist(
    identity: ActorIdentity,
    allowed_user_ids: Iterable[str] | None,
    allowed_agent_ids: Iterable[str] | None,
) -> bool:
    user_allowed = allowed_user_ids is not None and identity.user_id in allowed_user_ids
    agent_allowed = allowed_agent_ids is not None and identity.agent_id in allowed_agent_ids
    return user_allowed or agent_allowed


def _definition_payload(definition: CommandDefinition) -> dict[str, object]:
    return {
        "command_id": definition.command_id,
        "category": definition.category,
        "primary_name": definition.primary_name,
        "aliases": list(definition.aliases),
        "summary": definition.summary,
        "description": definition.description,
        "route_target": definition.route_target.model_dump(),
    }


__all__ = [
    "handle_commands",
    "handle_help",
    "handle_status",
]
