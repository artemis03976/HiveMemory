"""系统指令参数解析测试。"""

import pytest

from hivememory.gateway.commands.models import (
    CommandCategory,
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
    CommandRouteTarget,
    CommandRouteTargetKind,
)
from hivememory.gateway.commands.parser import (
    CommandParseError,
    build_parse_result,
    parse_command_args,
    tokenize_command,
    validate_command_args,
)


class TestTokenizeCommand:
    def test_splits_by_whitespace(self):
        assert tokenize_command("/help --verbose") == ["/help", "--verbose"]

    def test_respects_quotes(self):
        assert tokenize_command('/write title="fix cors"') == [
            "/write",
            "title=fix cors",
        ]

    def test_unclosed_quote_raises(self):
        with pytest.raises(CommandParseError):
            tokenize_command('/write content="unclosed')


class TestParseCommandArgs:
    def test_positional_only(self):
        assert parse_command_args(["a", "b"]) == {"_positional": ["a", "b"]}

    def test_positional_plus_options(self):
        args = parse_command_args(["query", "--limit", "5", "--verbose"])
        assert args == {"_positional": ["query"], "limit": "5", "verbose": True}

    def test_option_equals_value(self):
        assert parse_command_args(["--key=value"]) == {"key": "value"}

    def test_flag_without_value(self):
        assert parse_command_args(["--flag"]) == {"flag": True}

    def test_option_then_option_means_flag(self):
        assert parse_command_args(["--a", "--b"]) == {"a": True, "b": True}

    def test_empty_option_name_raises(self):
        with pytest.raises(CommandParseError, match="Empty option name"):
            parse_command_args(["--=value"])

    def test_bare_double_dash_is_positional(self):
        assert parse_command_args(["--"]) == {"_positional": ["--"]}


class TestValidateCommandArgs:
    def test_empty_schema_passes(self):
        assert validate_command_args({"a": 1}, {}) is None

    def test_missing_required(self):
        error = validate_command_args({"b": 1}, {"required": ["a"]})
        assert error == "Missing required argument: a"

    def test_type_boolean(self):
        schema = {"properties": {"flag": {"type": "boolean"}}}
        assert validate_command_args({"flag": True}, schema) is None
        assert validate_command_args({"flag": "yes"}, schema) is not None

    def test_type_string(self):
        schema = {"properties": {"name": {"type": "string"}}}
        assert validate_command_args({"name": "x"}, schema) is None
        assert validate_command_args({"name": 1}, schema) is not None

    def test_type_integer(self):
        schema = {"properties": {"n": {"type": "integer"}}}
        assert validate_command_args({"n": "5"}, schema) is None
        assert validate_command_args({"n": "abc"}, schema) is not None
        # bool 不是 integer
        assert validate_command_args({"n": True}, schema) is not None

    def test_type_number(self):
        schema = {"properties": {"x": {"type": "number"}}}
        assert validate_command_args({"x": "3.14"}, schema) is None
        assert validate_command_args({"x": "abc"}, schema) is not None

    def test_type_array(self):
        schema = {"properties": {"items": {"type": "array"}}}
        assert validate_command_args({"items": []}, schema) is None
        assert validate_command_args({"items": "x"}, schema) is not None

    def test_unknown_type_passes(self):
        assert validate_command_args({"x": object()}, {"properties": {"x": {"type": "any"}}}) is None


class TestBuildParseResult:
    def _definition(self) -> CommandDefinition:
        return CommandDefinition(
            command_id="test.cmd",
            category=CommandCategory.SYSTEM,
            primary_name="/test",
            summary="test",
            argument_schema={"required": ["q"]},
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.LOCAL_HANDLER,
                name="test.cmd",
            ),
        )

    def _definition_without_required(self) -> CommandDefinition:
        return self._definition().model_copy(update={"argument_schema": {}})

    def test_matched(self):
        definition = self._definition_without_required()
        result = build_parse_result(
            definition=definition,
            raw_input="/test q",
            tokens=["/test", "q"],
            matched_alias="/test",
            arg_tokens=["q"],
        )
        assert result.parse_status == CommandParseStatus.MATCHED
        assert result.args["_positional"] == ("q",)

    def test_invalid_args_on_parse_error(self):
        definition = self._definition()
        result = build_parse_result(
            definition=definition,
            raw_input="/test --=x",
            tokens=["/test", "--=x"],
            matched_alias="/test",
            arg_tokens=["--=x"],
        )
        assert result.parse_status == CommandParseStatus.INVALID_ARGS
        assert "Empty option name" in result.error

    def test_invalid_args_on_validation_error(self):
        definition = self._definition()
        result = build_parse_result(
            definition=definition,
            raw_input="/test",
            tokens=["/test"],
            matched_alias="/test",
            arg_tokens=[],
        )
        assert result.parse_status == CommandParseStatus.INVALID_ARGS
        assert result.error == "Missing required argument: q"
