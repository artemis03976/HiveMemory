"""任务进程命令终态转换测试。

被测边界：``command_terminal`` 把 Gateway 的命令解析结果转换为用户可见的
命令终态（v0.7.0 命令只解析不执行：解析成功返回"暂不可用"，解析失败拒绝）。
"""

from __future__ import annotations

import pytest

from hivememory.core.protocol.gateway import (
    CommandExecutionStatus,
    CommandParseResult,
    CommandParseStatus,
)
from hivememory.workspace.process.command_terminal import command_terminal


def _parse_result(
    *,
    parse_status: CommandParseStatus,
    error: str | None = None,
) -> CommandParseResult:
    """构造典型解析产物：解析成功携带 command_id，失败时为空。"""

    matched = parse_status == CommandParseStatus.MATCHED
    return CommandParseResult(
        command_id="system.help" if matched else None,
        raw_input="/help",
        name="/help",
        tokens=["/help"],
        matched_alias="/help" if matched else None,
        parse_status=parse_status,
        error=error,
    )


def test_matched_parse_result_becomes_not_implemented_and_unavailable() -> None:
    """解析成功的命令是命令自身终态：not_implemented、command.unavailable，id 保留。"""

    result = command_terminal(_parse_result(parse_status=CommandParseStatus.MATCHED))

    assert result.command_id == "system.help"
    assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
    assert result.error_code == "command.unavailable"
    assert result.client_action is None


@pytest.mark.parametrize(
    ("parse_status", "expected_error_code"),
    [
        (CommandParseStatus.UNKNOWN, "command.parse.unknown"),
        (CommandParseStatus.INVALID_ARGS, "command.parse.invalid_args"),
        (CommandParseStatus.AMBIGUOUS, "command.parse.ambiguous"),
    ],
)
def test_failed_parse_result_is_rejected_with_parse_error_code(
    parse_status: CommandParseStatus,
    expected_error_code: str,
) -> None:
    """解析失败的命令被拒绝：错误码为 command.parse.<解析状态>，文案取解析错误。"""

    result = command_terminal(
        _parse_result(
            parse_status=parse_status,
            error="Unknown system command: /help --bad",
        )
    )

    assert result.status == CommandExecutionStatus.REJECTED
    assert result.error_code == expected_error_code
    assert result.message == "Unknown system command: /help --bad"
    # 解析失败时 command_id 为空，回退到解析出的命令名。
    assert result.command_id == "/help"
    assert result.client_action is None
