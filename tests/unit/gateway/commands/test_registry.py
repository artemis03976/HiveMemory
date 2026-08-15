"""系统指令注册表与内置指令测试。"""

import pytest

from hivememory.gateway.commands.builtins import (
    create_builtin_command_registry,
    register_builtin_commands,
)
from hivememory.gateway.commands.models import (
    CommandCategory,
    CommandDefinition,
    CommandParseStatus,
    CommandRouteTarget,
    CommandRouteTargetKind,
)
from hivememory.gateway.commands.registry import CommandRegistry


def _definition(
    *,
    command_id: str,
    primary_name: str,
    aliases: tuple[str, ...] = (),
    priority: int = 100,
    enabled: bool = True,
    hidden: bool = False,
) -> CommandDefinition:
    return CommandDefinition(
        command_id=command_id,
        category=CommandCategory.SYSTEM,
        primary_name=primary_name,
        aliases=aliases,
        summary="test",
        route_target=CommandRouteTarget(
            kind=CommandRouteTargetKind.LOCAL_HANDLER,
            name=command_id,
        ),
        priority=priority,
        enabled=enabled,
        hidden=hidden,
    )


class TestCommandRegistry:
    def test_register_and_get(self):
        registry = CommandRegistry()
        definition = _definition(command_id="cmd.a", primary_name="/a")
        registry.register(definition)
        assert registry.get("cmd.a") is definition
        assert registry.get("missing") is None

    def test_duplicate_command_id_rejected(self):
        registry = CommandRegistry()
        registry.register(_definition(command_id="cmd.a", primary_name="/a"))
        with pytest.raises(ValueError, match="Duplicate command id"):
            registry.register(_definition(command_id="cmd.a", primary_name="/b"))

    def test_duplicate_alias_within_definition_rejected(self):
        registry = CommandRegistry()
        with pytest.raises(ValueError, match="Duplicate alias in command definition"):
            registry.register(
                _definition(command_id="cmd.a", primary_name="/a", aliases=("/a",))
            )

    def test_duplicate_alias_across_definitions_rejected(self):
        registry = CommandRegistry()
        registry.register(_definition(command_id="cmd.a", primary_name="/a"))
        with pytest.raises(ValueError, match="Duplicate command alias"):
            registry.register(_definition(command_id="cmd.b", primary_name="/b", aliases=("/a",)))

    def test_register_rejects_non_slash_name(self):
        registry = CommandRegistry()
        with pytest.raises(ValueError, match="must start with '/'"):
            registry.register(_definition(command_id="cmd.a", primary_name="a"))

    def test_list_sorted_and_filters_hidden(self):
        registry = CommandRegistry()
        registry.register(_definition(command_id="cmd.b", primary_name="/b", priority=50))
        registry.register(_definition(command_id="cmd.a", primary_name="/a", priority=10))
        registry.register(_definition(command_id="cmd.hidden", primary_name="/h", hidden=True))
        names = [definition.primary_name for definition in registry.list()]
        assert names == ["/a", "/b"]
        all_names = [definition.primary_name for definition in registry.list(include_hidden=True)]
        assert all_names == ["/a", "/b", "/h"]


class TestCommandMatch:
    def _registry(self) -> CommandRegistry:
        registry = CommandRegistry()
        registry.register(_definition(command_id="cmd.help", primary_name="/help", aliases=("/start",)))
        registry.register(
            _definition(command_id="cmd.sys", primary_name="/sys", priority=20)
        )
        registry.register(
            _definition(command_id="cmd.mult", primary_name="/multi word")
        )
        registry.register(_definition(command_id="cmd.off", primary_name="/off", enabled=False))
        return registry

    def test_blank_input_returns_none(self):
        assert self._registry().match("   ") is None

    def test_non_slash_returns_none(self):
        assert self._registry().match("hello world") is None

    def test_unknown_slash_returns_unknown(self):
        result = self._registry().match("/nope")
        assert result is not None
        assert result.parse_status == CommandParseStatus.UNKNOWN
        assert "Unknown system command" in result.error

    def test_basic_match_with_args(self):
        result = self._registry().match("/help --verbose")
        assert result is not None
        assert result.command_id == "cmd.help"
        assert result.parse_status == CommandParseStatus.MATCHED
        assert result.args["verbose"] is True

    def test_alias_match(self):
        result = self._registry().match("/start")
        assert result.command_id == "cmd.help"
        assert result.matched_alias == "/start"

    def test_multi_word_alias(self):
        result = self._registry().match("/multi word extra")
        assert result.command_id == "cmd.mult"
        assert result.args["_positional"] == ("extra",)

    def test_disabled_command_not_matched(self):
        result = self._registry().match("/off")
        assert result.parse_status == CommandParseStatus.UNKNOWN

    def test_prefix_matching_prefers_longer_alias(self):
        registry = CommandRegistry()
        registry.register(_definition(command_id="cmd.short", primary_name="/run"))
        registry.register(_definition(command_id="cmd.long", primary_name="/run fast"))
        result = registry.match("/run fast")
        # 长 alias 优先（更具体）
        assert result.command_id == "cmd.long"

    def test_tokenize_error_returns_invalid_args(self):
        result = self._registry().match('/help content="unclosed')
        assert result.parse_status == CommandParseStatus.INVALID_ARGS


class TestBuiltinCommands:
    def test_registers_all_builtins(self):
        registry = create_builtin_command_registry()
        ids = {definition.command_id for definition in registry.list()}
        assert ids == {"system.help", "system.commands", "system.clear", "runtime.status"}

    def test_alias_resolution(self):
        registry = create_builtin_command_registry()
        assert registry.match("/help").command_id == "system.help"
        assert registry.match("/start").command_id == "system.help"
        assert registry.match("/reset").command_id == "system.clear"

    def test_override_disables_command(self):
        registry = create_builtin_command_registry(
            builtin_overrides={"system.help": False}
        )
        ids = {definition.command_id for definition in registry.list()}
        assert "system.help" not in ids

    def test_register_returns_registry(self):
        registry = CommandRegistry()
        assert register_builtin_commands(registry) is registry
