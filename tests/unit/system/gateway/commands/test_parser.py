import pytest

from hivememory.gateway.commands.models import (
    CommandCategory,
    CommandDefinition,
    CommandParseStatus,
    CommandRouteTarget,
)
from hivememory.gateway.commands.parser import (
    CommandParseError,
    build_parse_result,
    parse_command_args,
    tokenize_command,
)


def _definition(**kwargs) -> CommandDefinition:
    defaults = dict(
        command_id="system.test",
        category=CommandCategory.SYSTEM,
        primary_name="/test",
        summary="Test command",
        route_target=CommandRouteTarget(name="system.test"),
    )
    defaults.update(kwargs)
    return CommandDefinition(**defaults)


def test_tokenize_command_supports_quoted_strings():
    assert tokenize_command('/help --topic "system commands"') == [
        "/help",
        "--topic",
        "system commands",
    ]


def test_tokenize_command_rejects_unclosed_quote():
    with pytest.raises(CommandParseError):
        tokenize_command('/help "unterminated')


def test_parse_command_args_supports_key_value_equals_and_flags():
    args = parse_command_args(["topic", "--name", "gateway", "--debug", "--limit=5"])

    assert args["_positional"] == ["topic"]
    assert args["name"] == "gateway"
    assert args["debug"] is True
    assert args["limit"] == "5"


def test_build_parse_result_returns_invalid_for_missing_required_arg():
    definition = _definition(
        argument_schema={
            "required": ["topic"],
            "properties": {"topic": {"type": "string"}},
        }
    )

    result = build_parse_result(
        definition=definition,
        raw_input="/test",
        tokens=["/test"],
        matched_alias="/test",
        arg_tokens=[],
    )

    assert result.parse_status == CommandParseStatus.INVALID_ARGS
    assert "topic" in result.error


def test_build_parse_result_returns_matched_args():
    result = build_parse_result(
        definition=_definition(),
        raw_input="/test --name gateway",
        tokens=["/test", "--name", "gateway"],
        matched_alias="/test",
        arg_tokens=["--name", "gateway"],
    )

    assert result.parse_status == CommandParseStatus.MATCHED
    assert result.args["name"] == "gateway"
    assert result.command_id == "system.test"
