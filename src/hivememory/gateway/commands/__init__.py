"""Phase 3 Gateway 命令系统兼容导出。"""

from hivememory.system.gateway.commands import (
    CommandCategory,
    CommandDefinition,
    CommandExecutionResult,
    CommandExecutionStatus,
    CommandHandler,
    CommandParseResult,
    CommandParseStatus,
    CommandPermissionPolicy,
    CommandRegistry,
    CommandRouteTarget,
    CommandRouteTargetKind,
    SystemCommandDispatcher,
    create_builtin_command_registry,
    register_builtin_commands,
)

__all__ = [
    "CommandCategory",
    "CommandDefinition",
    "CommandExecutionResult",
    "CommandExecutionStatus",
    "CommandHandler",
    "CommandParseResult",
    "CommandParseStatus",
    "CommandPermissionPolicy",
    "CommandRegistry",
    "CommandRouteTarget",
    "CommandRouteTargetKind",
    "SystemCommandDispatcher",
    "create_builtin_command_registry",
    "register_builtin_commands",
]
