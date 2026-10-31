"""一轮交互的材料：从事件集解析出步骤，并渲染为感知层唯一的文本表示。

记忆生成视图、LLM 接力摘要与逻辑块的 token 统计都使用这里的渲染，输入只有
满足一轮不变量的事件集与截断配置；同样的输入总得到同样的材料，感知层不另
写渲染规则。

**步骤**：用户消息之后的事件按序号就是步骤，每一步是一段助手正文、一段思考，
或一次工具调用（结局记在同一个事件上，可能没有结局）。思考只按先后排列，不归属
动作。

**渲染**：

- 用户消息：``[User]: …``；
- 助手正文：``[Assistant]: …``，按出现位置穿插在动作之间；
- 思考：``- THOUGHT: …``，原样渲染；
- 能力步骤（调用带 ``verb``）：一行摘要，不渲染结果——READ、SEARCH 读回的
  是已有记忆，渲染进生成会把它们重复提取。``SEARCH: "查询"``、
  ``READ: a, b``、``RUN: 目标 (状态)``，没有结局时状态为 ``no result``；
  WRITE、UPDATE、CALL 不渲染。外部 harness 经 HiveMemory 的 MCP 工具检索与
  读取时调用同样带 ``verb``，渲染相同；
- 外部工具步骤（调用带 ``tool_name``）：``- TOOL: 工具名 参数``，参数是完整的
  JSON；有结局时其后是 ``RESULT: …`` 或 ``RESULT (error): …``。没有结局不等于
  没有完成（来源可能漏记结果），因此不作断言；
- 无法解析的 MTP 指令（二者都为空）不渲染；空白的思考不渲染。

**截断**只作用于单条外部工具结果：工具输出可能是整份文件或日志，一轮的其余
内容（思考、参数与全部条目）原样保留，不受任何总量上限约束。截断保留首尾、
省略中段：工具输出的报错常在末尾。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import TurnEvent, turn_user_message


def render_turn_material(
    events: Sequence[TurnEvent], config: GenerationTranscriptConfig
) -> list[str]:
    """把一轮用户消息之后的步骤按顺序渲染为材料条目，每条可以多行。"""
    entries: list[str] = []
    for event in events:
        if event.kind == "tool_call":
            if event.verb is not None:
                summary = _capability_summary(event)
                if summary is not None:
                    entries.append(_bullet(summary))
            elif event.tool_name is not None:
                entries.append(_tool_text(event, config))
        elif event.kind == "thought":
            if event.content.strip():
                entries.append(_bullet(f"THOUGHT: {event.content.strip()}"))
        elif event.kind == "assistant_message":
            entries.append(f"[Assistant]: {event.content}")
    return entries


def format_turn_material(user_message: str, entries: Iterable[str]) -> str:
    """一轮材料的文本：用户消息一行，其后按顺序是各条材料。"""
    return "\n".join([f"[User]: {user_message}", *entries])


def render_turn_text(events: Sequence[TurnEvent], config: GenerationTranscriptConfig) -> str:
    """把一轮事件渲染为完整的材料文本。"""
    return format_turn_material(turn_user_message(events), render_turn_material(events, config))


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------


def _capability_summary(call: TurnEvent) -> str | None:
    """能力步骤的一行摘要；WRITE、UPDATE、CALL 不进入材料。"""
    if call.verb == "SEARCH":
        query = call.tool_args.get("query", "") if call.tool_args is not None else ""
        return f'SEARCH: "{query}"'
    if call.verb == "READ":
        return f"READ: {', '.join(call.targets) or '(unknown)'}"
    if call.verb == "RUN":
        status = call.status if call.status is not None else "no result"
        return f"RUN: {', '.join(call.targets) or '(unknown)'} ({status})"
    return None


def _tool_text(call: TurnEvent, config: GenerationTranscriptConfig) -> str:
    """外部工具步骤：调用一行，有结局时其后是按上限截断的结果。"""
    head = f"TOOL: {call.tool_name or '(unknown)'}"
    args = _args_text(call.tool_args)
    if args:
        head = f"{head} {args}"
    lines = [head]
    if call.result is not None:
        label = "RESULT (error)" if call.status == "error" else "RESULT"
        content = _clip(call.result.strip(), config.max_result_chars) or "(empty)"
        lines.append(f"{label}: {content}")
    return _bullet("\n".join(lines))


def _args_text(tool_args: Mapping[str, Any] | None) -> str:
    if not tool_args:
        return ""
    return json.dumps(dict(tool_args), ensure_ascii=False, default=str)


def _bullet(text: str) -> str:
    """列表项：首行加 ``- ``，后续行缩进到条目之下。"""
    first, *rest = text.split("\n")
    return "\n".join([f"- {first}", *(f"  {line}" for line in rest)])


def _clip(text: str, limit: int) -> str:
    """超过上限时保留首尾、省略中段，并标出省略的字符数。"""
    if len(text) <= limit:
        return text
    tail = limit // 2
    head = limit - tail
    omitted = len(text) - limit
    return f"{text[:head]} …[{omitted} chars omitted]… {text[len(text) - tail :]}"


__all__ = [
    "format_turn_material",
    "render_turn_material",
    "render_turn_text",
]
