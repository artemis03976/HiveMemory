"""用户代码工具的参数绑定、沙箱执行与 MTP 响应。"""

from hivememory.core.mtp import MTPResponse, MTPResponseStatus


def execute_user_tool(code: str, args: dict[str, str], *, timeout_seconds: int) -> MTPResponse:
    """
    在受限沙箱中执行用户工具

    用户工具通过 params 字典接收 MTP 参数。
    工具代码中可通过 params["key"] 访问传入的参数。

    Args:
        code: 工具代码 (来自 CODE_SNIPPET 记忆原子的 payload.content)
        args: MTP 指令参数 (key-value pairs)

    Returns:
        MTPResponse: 执行结果
    """
    from hivememory.agent_runtime.tools import execute_sandboxed

    result = execute_sandboxed(
        code,
        namespace_extras={"params": dict(args)},
        timeout_seconds=timeout_seconds,
    )

    return MTPResponse(status=MTPResponseStatus.SUCCESS, content=result.content)
