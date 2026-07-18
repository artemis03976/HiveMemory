from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
)
from hivememory.gateway.commands.handlers import (
    handle_commands,
    handle_help,
    handle_status,
)
from hivememory.gateway.commands.models import (
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
    CommandRouteTargetKind,
)
from hivememory.gateway.commands.registry import CommandRegistry

logger = logging.getLogger(__name__)

CommandHandler = Callable[..., CommandExecutionResult | Awaitable[CommandExecutionResult]]


class SystemCommandDispatcher:
    """系统指令统一执行入口。

    Dispatcher 是唯一允许产生指令副作用的组件。Gateway S0 只负责识别与
    透传，不调用 handler。
    """

    def __init__(
        self,
        registry: CommandRegistry,
        *,
        global_bus: Any | None = None,
        debug_enabled: bool = False,
        expose_listing: bool = True,
        handlers: dict[str, CommandHandler] | None = None,
    ) -> None:
        self.registry = registry
        self.global_bus = global_bus
        self.debug_enabled = debug_enabled
        self.expose_listing = expose_listing
        self._handlers: dict[str, CommandHandler] = {
            "system.help": handle_help,
            "system.commands": handle_commands,
            "runtime.status": handle_status,
        }
        if handlers:
            self._handlers.update(handlers)

    async def execute(
        self,
        command: CommandParseResult | None,
        *,
        identity: Identity | None = None,
    ) -> CommandExecutionResult:
        """执行已解析的系统指令。"""

        identity = identity or Identity()

        if command is None:
            return self._rejected(
                command_id="unknown",
                message="未提供系统指令。",
                error_code="command.missing",
            )

        if command.parse_status != CommandParseStatus.MATCHED:
            return self._rejected(
                command_id=command.command_id or command.name or "unknown",
                message=command.error or "系统指令解析失败。",
                error_code=f"command.parse.{command.parse_status}",
            )

        if not command.command_id:
            return self._rejected(
                command_id=command.name or "unknown",
                message="系统指令缺少 command_id。",
                error_code="command.missing_id",
            )

        definition = self.registry.get(command.command_id)
        if definition is None:
            return self._rejected(
                command_id=command.command_id,
                message=f"系统指令未注册：{command.command_id}",
                error_code="command.not_registered",
            )

        permission_result = self._check_permission(definition, identity)
        if permission_result is not None:
            return permission_result

        try:
            return await self._dispatch(definition, command, identity)
        except Exception as exc:  # pragma: no cover - tested through handler failure
            logger.error("System command dispatch failed: %s", exc, exc_info=True)
            return CommandExecutionResult(
                command_id=definition.command_id,
                status=CommandExecutionStatus.FAILED,
                message=f"系统指令执行失败：{exc}",
                error_code="command.failed",
            )

    async def _dispatch(
        self,
        definition: CommandDefinition,
        command: CommandParseResult,
        identity: Identity,
    ) -> CommandExecutionResult:
        target = definition.route_target
        target_kind = target.kind

        if target_kind == CommandRouteTargetKind.LOCAL_HANDLER:
            return await self._dispatch_local_handler(definition, command, identity)

        if target_kind == CommandRouteTargetKind.CLIENT_ACTION:
            return CommandExecutionResult(
                command_id=definition.command_id,
                status=CommandExecutionStatus.COMPLETED,
                message=definition.summary,
                client_action={"type": target.name, **target.payload_template},
            )

        if target_kind == CommandRouteTargetKind.GLOBAL_ROUTE:
            return await self._dispatch_global_route(definition, command, identity)

        return CommandExecutionResult(
            command_id=definition.command_id,
            status=CommandExecutionStatus.NOT_IMPLEMENTED,
            message=f"系统指令路由尚未实现：{target.kind}",
            error_code="command.route.not_implemented",
        )

    async def _dispatch_local_handler(
        self,
        definition: CommandDefinition,
        command: CommandParseResult,
        identity: Identity,
    ) -> CommandExecutionResult:
        handler = self._handlers.get(definition.route_target.name)
        if handler is None:
            return CommandExecutionResult(
                command_id=definition.command_id,
                status=CommandExecutionStatus.NOT_IMPLEMENTED,
                message=f"系统指令 handler 未实现：{definition.route_target.name}",
                error_code="command.handler.not_implemented",
            )

        result = handler(
            command=command,
            registry=self.registry,
            identity=identity,
            debug_enabled=self.debug_enabled,
            expose_listing=self.expose_listing,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _dispatch_global_route(
        self,
        definition: CommandDefinition,
        command: CommandParseResult,
        identity: Identity,
    ) -> CommandExecutionResult:
        if self.global_bus is None:
            return CommandExecutionResult(
                command_id=definition.command_id,
                status=CommandExecutionStatus.NOT_IMPLEMENTED,
                message="未配置 GlobalSystemBus，无法执行 global_route 指令。",
                error_code="command.global_route.no_bus",
            )

        payload = dict(definition.route_target.payload_template)
        response = await self.global_bus.request(
            definition.route_target.name,
            command=command,
            identity=identity,
            args=command.args,
            **payload,
        )
        if isinstance(response, CommandExecutionResult):
            return response
        return CommandExecutionResult(
            command_id=definition.command_id,
            status=CommandExecutionStatus.COMPLETED,
            message="系统指令已通过全局路由执行。",
            data={"response": response},
        )

    def _check_permission(
        self,
        definition: CommandDefinition,
        identity: Identity,
    ) -> CommandExecutionResult | None:
        permission = definition.permission

        if permission.allowed_user_ids is not None and identity.user_id not in permission.allowed_user_ids:
            return self._permission_rejected(definition, "当前用户无权执行该系统指令。")
        if permission.allowed_agent_ids is not None and identity.agent_id not in permission.allowed_agent_ids:
            return self._permission_rejected(definition, "当前 Agent 无权执行该系统指令。")
        if permission.visibility == "debug" and not self.debug_enabled:
            return self._permission_rejected(definition, "Debug 指令未启用。")
        if permission.visibility == "admin" and not (
            _is_allowed(identity.user_id, permission.allowed_user_ids)
            or _is_allowed(identity.agent_id, permission.allowed_agent_ids)
        ):
            return self._permission_rejected(definition, "Admin 指令默认不可执行。")
        if permission.destructive or permission.requires_confirmation:
            return CommandExecutionResult(
                command_id=definition.command_id,
                status=CommandExecutionStatus.REQUIRES_CONFIRMATION,
                message="该系统指令需要确认后才能执行。",
                error_code="command.requires_confirmation",
            )
        return None

    @staticmethod
    def _rejected(
        *,
        command_id: str,
        message: str,
        error_code: str,
    ) -> CommandExecutionResult:
        return CommandExecutionResult(
            command_id=command_id,
            status=CommandExecutionStatus.REJECTED,
            message=message,
            error_code=error_code,
        )

    @staticmethod
    def _permission_rejected(
        definition: CommandDefinition,
        message: str,
    ) -> CommandExecutionResult:
        return CommandExecutionResult(
            command_id=definition.command_id,
            status=CommandExecutionStatus.REJECTED,
            message=message,
            error_code="command.permission_denied",
        )


def _is_allowed(value: str, allowed: tuple[str, ...] | None) -> bool:
    return allowed is not None and value in allowed


__all__ = ["CommandHandler", "SystemCommandDispatcher"]
