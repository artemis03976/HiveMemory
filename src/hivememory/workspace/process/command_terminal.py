"""任务进程的命令终态转换。

命令解析与执行解耦后（v0.7.0 命令只解析），Gateway 的命令结果只携带
解析产物，命令不执行：进程在 Gateway 阶段短路时，由本模块把解析结果
转换为用户可见的命令终态（``CommandExecutionResult``），不进入
prepare、CPU 分配与 Actor 执行。命令在进程内何时、由谁运行尚未决定，
随命令系统接回时重新设计。
"""

from __future__ import annotations

from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    CommandParseResult,
    CommandParseStatus,
)


def command_terminal(parse_result: CommandParseResult) -> CommandExecutionResult:
    """把 Gateway 的命令解析结果转换为命令终态（纯函数）。

    - 解析成功：命令暂不可用，状态 ``not_implemented``、错误码
      ``command.unavailable``；这是命令自身的终态，不是进程失败；
    - 解析失败（未知、参数无效、有歧义）：拒绝，状态 ``rejected``、
      错误码 ``command.parse.<解析状态>``，文案取解析错误；
    - 终态不携带客户端动作：``/clear`` 的 ``clear_chat`` 随执行一起后置。
    """

    # 解析失败时 command_id 为空，回退到解析出的命令名。
    command_id = parse_result.command_id or parse_result.name or "unknown"
    if parse_result.parse_status == CommandParseStatus.MATCHED:
        return CommandExecutionResult(
            command_id=command_id,
            status=CommandExecutionStatus.NOT_IMPLEMENTED,
            message=f"系统指令 {parse_result.name} 暂不可用。",
            error_code="command.unavailable",
        )
    return CommandExecutionResult(
        command_id=command_id,
        status=CommandExecutionStatus.REJECTED,
        message=parse_result.error or "系统指令解析失败。",
        error_code=f"command.parse.{parse_result.parse_status}",
    )


__all__ = ["command_terminal"]
