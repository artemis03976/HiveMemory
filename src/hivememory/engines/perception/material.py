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
- 思考：``- THOUGHT: …``，按单条上限截断；
- 能力步骤（调用带 ``verb``）：一行摘要，不渲染结果——READ、SEARCH 读回的
  是已有记忆，渲染进生成会把它们重复提取。``SEARCH: "查询"``、
  ``READ: a, b``、``RUN: 目标 (状态)``，没有结局时状态为 ``no result``；
  WRITE、UPDATE、CALL 不渲染。外部 harness 经 HiveMemory 的 MCP 工具检索与
  读取时调用同样带 ``verb``，渲染相同；
- 外部工具步骤（调用带 ``tool_name``）：``- TOOL: 工具名 参数摘要``，有结局时
  其后是 ``RESULT: …`` 或 ``RESULT (error): …``，参数与结果按各自上限截断。
  没有结局不等于没有完成（来源可能漏记结果），因此不作断言；
- 无法解析的 MTP 指令（二者都为空）不渲染；空白的思考不渲染。

**单轮上限**只约束思考、能力步骤与外部工具步骤，用户消息与助手正文不计入。
超出时依次保留思考、出错的工具步骤与最近的工具步骤，其余外部工具步骤只保留
首行。上限是硬上限：全部降为一行后仍超出时，按优先级从低到高省略整条（较早
的非出错工具步骤、能力摘要、出错的工具步骤，最后才是较早的思考），并在用户
消息之后以一行标明省略的条目数。

截断保留首尾、省略中段：工具输出的报错与推理的结论常在末尾。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import TurnEvent, turn_user_message

MaterialEntryKind = Literal["assistant", "thought", "capability", "tool", "omitted"]

# 受单轮上限约束的条目种类；用户消息与助手正文不计入。
_LIMITED_KINDS = frozenset({"thought", "capability", "tool"})


@dataclass(frozen=True)
class MaterialEntry:
    """用户消息之后的一条材料；``text`` 是渲染后的文本，可以多行。"""

    kind: MaterialEntryKind
    text: str


def render_turn_material(
    events: Sequence[TurnEvent], config: GenerationTranscriptConfig
) -> list[MaterialEntry]:
    """把一轮用户消息之后的步骤渲染为材料条目，已按配置截断并满足单轮上限。"""
    entries: list[MaterialEntry] = []
    # 外部工具步骤的降级形态（首行），按条目下标登记。
    degraded: dict[int, str] = {}
    failed: set[int] = set()
    for event in events:
        if event.kind == "tool_call":
            if event.verb is not None:
                summary = _capability_summary(event)
                if summary is not None:
                    entries.append(MaterialEntry("capability", _bullet(summary)))
            elif event.tool_name is not None:
                full, head = _tool_texts(event, config)
                degraded[len(entries)] = head
                if event.status == "error":
                    failed.add(len(entries))
                entries.append(MaterialEntry("tool", full))
        elif event.kind == "thought":
            if event.content.strip():
                thought = _clip(event.content.strip(), config.max_thought_chars)
                entries.append(MaterialEntry("thought", _bullet(f"THOUGHT: {thought}")))
        elif event.kind == "assistant_message":
            entries.append(MaterialEntry("assistant", f"[Assistant]: {event.content}"))
    return _fit_turn_limit(entries, degraded, failed, config.max_turn_action_chars)


def format_turn_material(user_message: str, entries: Iterable[str]) -> str:
    """一轮材料的文本：用户消息一行，其后按顺序是各条材料。"""
    return "\n".join([f"[User]: {user_message}", *entries])


def render_turn_text(events: Sequence[TurnEvent], config: GenerationTranscriptConfig) -> str:
    """把一轮事件渲染为完整的材料文本。"""
    return format_turn_material(
        turn_user_message(events),
        (entry.text for entry in render_turn_material(events, config)),
    )


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


def _tool_texts(call: TurnEvent, config: GenerationTranscriptConfig) -> tuple[str, str]:
    """返回外部工具步骤的完整形态与降级形态（首行即降级形态）。"""
    head = f"TOOL: {call.tool_name or '(unknown)'}"
    args = _args_summary(call.tool_args, config.max_args_chars)
    if args:
        head = f"{head} {args}"
    lines = [head]
    if call.result is not None:
        label = "RESULT (error)" if call.status == "error" else "RESULT"
        content = _clip(call.result.strip(), config.max_result_chars) or "(empty)"
        lines.append(f"{label}: {content}")
    return _bullet("\n".join(lines)), _bullet(head)


def _args_summary(tool_args: Mapping[str, Any] | None, limit: int) -> str:
    if not tool_args:
        return ""
    rendered = json.dumps(dict(tool_args), ensure_ascii=False, default=str)
    return _clip(rendered, limit)


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


def _fit_turn_limit(
    entries: list[MaterialEntry],
    degraded: Mapping[int, str],
    failed: set[int],
    limit: int,
) -> list[MaterialEntry]:
    """让受约束的条目满足单轮上限。

    先把外部工具步骤全部降为一行，再按优先级逐个恢复完整形态：出错的步骤、
    其余步骤，各自由近及远；放不下的步骤跳过，继续尝试更短的步骤。全部降为
    一行后仍超出时，按优先级从低到高省略整条，直到连同省略说明放得下。
    """
    if _limited_length(entries) <= limit:
        return entries
    texts = [degraded.get(index, entry.text) for index, entry in enumerate(entries)]
    fitted = [MaterialEntry(entry.kind, text) for entry, text in zip(entries, texts, strict=True)]
    remaining = limit - _limited_length(fitted)
    recent_first = sorted(degraded, reverse=True)
    priority = [index for index in recent_first if index in failed] + [
        index for index in recent_first if index not in failed
    ]
    for index in priority:
        extra = len(entries[index].text) - len(texts[index])
        if extra <= remaining:
            fitted[index] = entries[index]
            remaining -= extra
    if remaining >= 0:
        return fitted
    return _omit_until_fit(fitted, failed, limit)


def _omit_until_fit(
    entries: list[MaterialEntry], failed: set[int], limit: int
) -> list[MaterialEntry]:
    """按优先级从低到高、同级由远及近省略整条，并在最前面说明省略的条目数。"""

    def oldest_first(kind: MaterialEntryKind, *, failed_tools: bool = False) -> list[int]:
        return [
            index
            for index, entry in enumerate(entries)
            if entry.kind == kind and (kind != "tool" or (index in failed) == failed_tools)
        ]

    drop_order = [
        *oldest_first("tool"),
        *oldest_first("capability"),
        *oldest_first("tool", failed_tools=True),
        *oldest_first("thought"),
    ]
    total = _limited_length(entries)
    dropped: set[int] = set()
    for index in drop_order:
        if dropped and total + len(_omitted_text(len(dropped))) <= limit:
            break
        dropped.add(index)
        total -= len(entries[index].text)
    kept = [entry for index, entry in enumerate(entries) if index not in dropped]
    return [MaterialEntry("omitted", _omitted_text(len(dropped))), *kept]


def _omitted_text(count: int) -> str:
    return _bullet(f"OMITTED: {count} entries over the per-turn limit")


def _limited_length(entries: Iterable[MaterialEntry]) -> int:
    return sum(len(entry.text) for entry in entries if entry.kind in _LIMITED_KINDS)


__all__ = [
    "MaterialEntry",
    "MaterialEntryKind",
    "format_turn_material",
    "render_turn_material",
    "render_turn_text",
]
