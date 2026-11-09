"""本插件四个记忆工具的调用在回流时的写法（回声标记）。

插件声明的 MCP server 提供检索、读取、写入与更新 HiveMemory 记忆四个工具。
这些调用回流时归一化为 HiveMemory 能力调用，只写动词与作用对象，不上传调用名：
检索为 ``SEARCH``，查询留在参数中；读取为 ``READ``，参数 ``aliases`` 移出参数、
作为作用对象；写入为 ``WRITE``，没有作用对象；更新为 ``UPDATE``，参数 ``alias``
移出参数、作为唯一的作用对象。检索与读取取回的是已有记忆，写入与更新的内容
已经由显式写入保存，HiveMemory 生成时都不会把它们再提取一遍。写入与更新的
参数 ``turn`` 是 ``PreToolUse`` hook 补上的轮次信息，不上传。

工具名与参数名和 HiveMemory 的 MCP server 一致，改名时同时修改。各 harness 的
读取器负责认出哪些调用来自本插件的 server，再交给 :func:`tool_call_fields`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 写入与更新工具中由 PreToolUse hook 填写的轮次信息参数。
TURN_PARAM = "turn"


@dataclass(frozen=True)
class MemoryTool:
    """一个记忆工具对应的 HiveMemory 能力：动词与作用对象所在的参数。

    ``single_target`` 表示作用对象参数是单个 alias，否则是 alias 列表。
    """

    verb: str
    target_param: str | None = None
    single_target: bool = False

    def targets(self, value: Any) -> list[str] | None:
        """参数中的作用对象；形状不对时为 ``None``，MCP server 已以工具错误拒绝这次调用。"""
        if self.single_target:
            return [value] if isinstance(value, str) and value else None
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return value
        return None


# 记忆工具名到 HiveMemory 能力。
MEMORY_TOOLS = {
    "memory_search": MemoryTool("SEARCH"),
    "memory_read": MemoryTool("READ", target_param="aliases"),
    "memory_write": MemoryTool("WRITE"),
    "memory_update": MemoryTool("UPDATE", target_param="alias", single_target=True),
}


def tool_call_fields(
    tool_name: str, memory_tool: str | None, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """工具调用事件中描述调用本身的字段：工具名与参数，或能力动词、作用对象与参数。

    ``memory_tool`` 是本插件记忆工具的名字，调用不来自本插件时为 ``None``。模型
    传错作用对象时按普通工具写出：MCP server 已以工具错误拒绝这次调用，而
    服务端要求读取与更新带作用对象，否则会拒绝整轮。
    """
    args = dict(tool_args)
    spec = MEMORY_TOOLS.get(memory_tool) if memory_tool is not None else None
    if spec is None:
        return {"tool_name": tool_name, "tool_args": args}
    # 轮次信息是连接器自己补上的，不属于这次调用的内容。
    args.pop(TURN_PARAM, None)
    targets = None
    if spec.target_param:
        targets = spec.targets(args.get(spec.target_param))
        if targets is None:
            return {"tool_name": tool_name, "tool_args": args}
        del args[spec.target_param]
    return {"verb": spec.verb, "targets": targets, "tool_args": args or None}
