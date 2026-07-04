from __future__ import annotations

import shlex
from typing import Any

from hivememory.system.gateway.commands.models import (
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
)


class CommandParseError(ValueError):
    """轻量命令行解析错误。"""
    pass


def tokenize_command(raw_input: str) -> list[str]:
    """按命令行风格切分输入，仅支持引号，不提供 shell expansion。"""

    try:
        return shlex.split(raw_input, posix=True)
    except ValueError as exc:
        raise CommandParseError(str(exc)) from exc


def parse_command_args(tokens: list[str]) -> dict[str, Any]:
    """解析参数 token，支持位置参数、--key value、--key=value 和 --flag。"""

    args: dict[str, Any] = {}
    positional: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and len(token) > 2:
            key_value = token[2:]
            if "=" in key_value:
                key, value = key_value.split("=", 1)
                if not key:
                    raise CommandParseError("Empty option name")
                args[key] = value
            else:
                key = key_value
                if not key:
                    raise CommandParseError("Empty option name")
                next_index = index + 1
                if next_index < len(tokens) and not tokens[next_index].startswith("--"):
                    args[key] = tokens[next_index]
                    index = next_index
                else:
                    args[key] = True
        else:
            positional.append(token)
        index += 1

    if positional:
        args["_positional"] = positional
    return args


def validate_command_args(
    args: dict[str, Any],
    argument_schema: dict[str, Any],
) -> str | None:
    """执行最小 schema 校验，避免 Phase 2.0 引入完整 JSON Schema 依赖。"""

    if not argument_schema:
        return None

    required = argument_schema.get("required", [])
    for key in required:
        if key not in args:
            return f"Missing required argument: {key}"

    properties = argument_schema.get("properties", {})
    for key, schema in properties.items():
        if key not in args:
            continue
        expected_type = schema.get("type") if isinstance(schema, dict) else None
        if expected_type and not _matches_type(args[key], expected_type):
            return f"Invalid type for argument '{key}': expected {expected_type}"

    return None


def build_parse_result(
    definition: CommandDefinition,
    raw_input: str,
    tokens: list[str],
    matched_alias: str,
    arg_tokens: list[str],
) -> CommandParseResult:
    """把匹配到的指令 definition 与参数 token 合成为结构化解析结果。"""

    try:
        args = parse_command_args(arg_tokens)
    except CommandParseError as exc:
        return CommandParseResult(
            command_id=definition.command_id,
            raw_input=raw_input,
            name=definition.primary_name,
            tokens=tokens,
            matched_alias=matched_alias,
            parse_status=CommandParseStatus.INVALID_ARGS,
            error=str(exc),
        )

    validation_error = validate_command_args(args, definition.argument_schema)
    if validation_error is not None:
        return CommandParseResult(
            command_id=definition.command_id,
            raw_input=raw_input,
            name=definition.primary_name,
            args=args,
            tokens=tokens,
            matched_alias=matched_alias,
            parse_status=CommandParseStatus.INVALID_ARGS,
            error=validation_error,
        )

    return CommandParseResult(
        command_id=definition.command_id,
        raw_input=raw_input,
        name=definition.primary_name,
        args=args,
        tokens=tokens,
        matched_alias=matched_alias,
        parse_status=CommandParseStatus.MATCHED,
    )


def _matches_type(value: Any, expected_type: str) -> bool:
    """最小类型检查；数值类型保持字符串输入，但要求可转换。"""

    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        if isinstance(value, bool):
            return False
        try:
            int(value)
        except (TypeError, ValueError):
            return False
        return True
    if expected_type == "number":
        if isinstance(value, bool):
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True
    if expected_type == "array":
        return isinstance(value, list)
    return True


__all__ = [
    "CommandParseError",
    "build_parse_result",
    "parse_command_args",
    "tokenize_command",
    "validate_command_args",
]
