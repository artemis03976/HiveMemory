"""系统指令内置 handler 测试。"""

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import CommandExecutionStatus
from hivememory.gateway.commands.builtins import create_builtin_command_registry
from hivememory.gateway.commands.handlers import (
    handle_commands,
    handle_help,
    handle_status,
)
from hivememory.gateway.commands.models import (
    CommandCategory,
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
    CommandPermissionPolicy,
    CommandRouteTarget,
    CommandRouteTargetKind,
)


def _definition(
    *,
    command_id: str,
    primary_name: str,
    visibility: str = "public",
    allowed_user_ids: tuple[str, ...] | None = None,
    allowed_agent_ids: tuple[str, ...] | None = None,
) -> CommandDefinition:
    return CommandDefinition(
        command_id=command_id,
        category=CommandCategory.SYSTEM,
        primary_name=primary_name,
        summary="summary text",
        description="desc",
        permission=CommandPermissionPolicy(
            visibility=visibility,
            allowed_user_ids=allowed_user_ids,
            allowed_agent_ids=allowed_agent_ids,
        ),
        route_target=CommandRouteTarget(
            kind=CommandRouteTargetKind.LOCAL_HANDLER,
            name=command_id,
        ),
    )


def _registry():
    registry = create_builtin_command_registry()
    registry.register(_definition(command_id="dbg.cmd", primary_name="/dbg", visibility="debug"))
    registry.register(
        _definition(
            command_id="adm.cmd",
            primary_name="/adm",
            visibility="admin",
            allowed_user_ids=("admin_u",),
        )
    )
    return registry


def _parse_result(command_id: str = "system.help") -> CommandParseResult:
    return CommandParseResult(
        command_id=command_id,
        raw_input="/help",
        parse_status=CommandParseStatus.MATCHED,
    )


class TestHandleHelp:
    def test_lists_visible_public_commands(self):
        result = handle_help(
            command=_parse_result(),
            registry=_registry(),
            identity=Identity(),
        )
        assert result.status == CommandExecutionStatus.COMPLETED
        assert "/help" in result.message
        assert "/commands" in result.message
        # debug/admin 指令默认不可见
        assert "/dbg" not in result.message
        assert "/adm" not in result.message
        assert len(result.data["commands"]) == 4

    def test_debug_enabled_shows_debug_commands(self):
        result = handle_help(
            command=_parse_result(),
            registry=_registry(),
            identity=Identity(),
            debug_enabled=True,
        )
        assert "/dbg" in result.message
        assert len(result.data["commands"]) == 5

    def test_expose_listing_false_hides_all(self):
        result = handle_help(
            command=_parse_result(),
            registry=_registry(),
            identity=Identity(),
            expose_listing=False,
        )
        assert result.data["commands"] == ()
        assert "当前没有可用系统指令" in result.message

    def test_admin_visible_to_allowed_user(self):
        result = handle_help(
            command=_parse_result(),
            registry=_registry(),
            identity=Identity(user_id="admin_u"),
        )
        assert "/adm" in result.message


class TestHandleCommands:
    def test_reports_visible_count(self):
        result = handle_commands(
            command=_parse_result(command_id="system.commands"),
            registry=_registry(),
            identity=Identity(),
        )
        assert result.status == CommandExecutionStatus.COMPLETED
        assert result.message == "共 4 条可用系统指令。"
        ids = {entry["command_id"] for entry in result.data["commands"]}
        assert ids == {"system.help", "system.commands", "system.clear", "runtime.status"}

    def test_debug_flag_included_in_payload(self):
        result = handle_commands(
            command=_parse_result(),
            registry=_registry(),
            identity=Identity(),
            debug_enabled=True,
        )
        assert "dbg.cmd" in {entry["command_id"] for entry in result.data["commands"]}

    def test_expose_listing_false(self):
        result = handle_commands(
            command=_parse_result(),
            registry=_registry(),
            identity=Identity(),
            expose_listing=False,
        )
        assert result.message == "共 0 条可用系统指令。"


class TestHandleStatus:
    def test_reports_runtime_summary(self):
        result = handle_status(
            command=_parse_result(command_id="runtime.status"),
            registry=_registry(),
            identity=Identity(),
        )
        assert result.status == CommandExecutionStatus.COMPLETED
        assert result.data["gateway"] == "ok"
        assert result.data["commands_visible"] == 4
        assert result.data["debug_enabled"] is False
