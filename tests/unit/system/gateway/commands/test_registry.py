import pytest

from hivememory.gateway.commands import (
    CommandCategory,
    CommandDefinition,
    CommandParseStatus,
    CommandRegistry,
    CommandRouteTarget,
    create_builtin_command_registry,
)


def _definition(
    command_id: str = "system.test",
    primary_name: str = "/test",
    **kwargs,
) -> CommandDefinition:
    defaults = dict(
        category=CommandCategory.SYSTEM,
        summary="Test command",
        route_target=CommandRouteTarget(name=command_id),
    )
    defaults.update(kwargs)
    return CommandDefinition(
        command_id=command_id,
        primary_name=primary_name,
        **defaults,
    )


def test_register_get_and_list():
    registry = CommandRegistry()
    definition = _definition()

    registry.register(definition)

    assert registry.get("system.test") is definition
    assert registry.list() == [definition]


def test_register_rejects_duplicate_command_id():
    registry = CommandRegistry()
    registry.register(_definition())

    with pytest.raises(ValueError, match="Duplicate command id"):
        registry.register(_definition(primary_name="/other"))


def test_register_rejects_duplicate_alias():
    registry = CommandRegistry()
    registry.register(_definition(command_id="system.one", primary_name="/one", aliases=["/same"]))

    with pytest.raises(ValueError, match="Duplicate command alias"):
        registry.register(_definition(command_id="system.two", primary_name="/same"))


def test_list_hides_hidden_by_default():
    registry = CommandRegistry()
    visible = _definition(command_id="system.visible", primary_name="/visible")
    hidden = _definition(command_id="system.hidden", primary_name="/hidden", hidden=True)
    registry.register(hidden)
    registry.register(visible)

    assert registry.list() == [visible]
    assert registry.list(include_hidden=True) == [hidden, visible]


def test_match_long_command_name_wins():
    registry = CommandRegistry()
    registry.register(_definition(command_id="system.models", primary_name="/models", priority=10))
    registry.register(_definition(command_id="system.models.ready", primary_name="/models ready", priority=100))

    result = registry.match("/models ready --all")

    assert result.parse_status == CommandParseStatus.MATCHED
    assert result.command_id == "system.models.ready"
    assert result.args["all"] is True


def test_match_unknown_slash_command_returns_unknown():
    result = CommandRegistry().match("/missing")

    assert result.parse_status == CommandParseStatus.UNKNOWN
    assert result.name == "/missing"


def test_match_non_command_returns_none():
    assert CommandRegistry().match("hello") is None


def test_builtin_registry_matches_legacy_commands():
    registry = create_builtin_command_registry()

    for command in ["/help", "/commands", "/clear", "/status", "/reset", "/start", "/restart"]:
        result = registry.match(command)
        assert result.parse_status == CommandParseStatus.MATCHED
        assert result.command_id is not None


def test_builtin_registry_supports_disable_override():
    registry = create_builtin_command_registry({"system.clear": False})

    result = registry.match("/clear")

    assert result.parse_status == CommandParseStatus.UNKNOWN
