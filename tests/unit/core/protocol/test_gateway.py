"""Gateway 公开结果模型：命令解析结果对外报告的命令标识。"""

from __future__ import annotations

import pytest

from hivememory.core.protocol.gateway import CommandParseResult, CommandParseStatus


@pytest.mark.parametrize(
    ("command_id", "name", "expected"),
    [
        ("system.help", "/help", "system.help"),
        (None, "/no_such_command", "/no_such_command"),
        (None, "", "unknown"),
    ],
)
def test_effective_command_id_falls_back_from_id_to_name_to_unknown(
    command_id: str | None, name: str, expected: str
) -> None:
    """解析成功取 command_id；未匹配时取解析出的命令名；切分失败连命令名都没有时为 unknown。"""
    parse = CommandParseResult(
        command_id=command_id,
        raw_input=name or '/x "unterminated',
        name=name,
        parse_status=(
            CommandParseStatus.MATCHED if command_id else CommandParseStatus.INVALID_ARGS
        ),
    )

    assert parse.effective_command_id == expected
