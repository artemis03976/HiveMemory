"""记忆生成视图的动作段：思考、外部工具动作与 MTP 动作摘要。

生成转写与感知的 token 统计共用本模块，二者因此遵循同一套截断规则，
折叠阈值反映生成实际看到的材料。渲染规则：

- **思考**：按出现顺序渲染。显式附属于某个动作（``action_id`` 指向该动作）
  的思考紧排在该动作之前；未附属于动作的思考在其事件位置单独成条。
  归约器会把未附属的思考吸附到最近的动作，生成视图不采用这种吸附：
  吸附会把引出下一步的推理排到上一个动作的结果之后。
- **外部工具动作**（``tool_kind`` 为空）：渲染工具名、参数摘要与各条结果，
  结果按单条上限截断，出错的结果标明状态。
- **MTP 动作**（带 ``tool_kind`` 的动作）：维持一行摘要（SEARCH、READ、RUN；
  CALL、WRITE、UPDATE 与无法解析的动作不渲染），不渲染结果——READ、SEARCH
  读回的是已有记忆，渲染进生成会把它们重复提取。外部 harness 经 HiveMemory
  的 MCP 工具检索与读取时，导入的事件同样带 ``tool_kind``，按同一规则处理。
- **单轮上限**：动作段超过上限时依次保留思考、出错的动作与最近的动作，其余
  外部工具动作只保留工具名与参数摘要的一行。上限是硬上限：全部降为一行后
  仍超出时，按优先级从低到高省略整条（较早的非出错动作、MTP 摘要、出错的
  动作，最后才是较早的思考），并在段首以一行标明省略的条目数。
- 空白的思考不渲染。

截断保留首尾、省略中段：工具输出的报错与推理的结论常在末尾。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import ActionReducer, AgentAction, TraceItem, TurnRecord

ActionEntryKind = Literal["trace", "thought", "tool", "omitted"]

_ERROR_STATUSES = frozenset({"error", "failed", "failure"})


@dataclass(frozen=True)
class ActionEntry:
    """动作段的一个条目。

    ``kind`` 区分 MTP 动作的一行摘要（``trace``）、思考（``thought``）、
    外部工具动作（``tool``）与超出单轮上限的省略说明（``omitted``）；
    ``text`` 可以多行，首行是条目的标签行。
    """

    kind: ActionEntryKind
    text: str


def render_action_entries(
    turn: TurnRecord, config: GenerationTranscriptConfig
) -> list[ActionEntry]:
    """把一轮的过程渲染为动作段条目，已按配置截断并满足单轮上限。"""
    actions = {action.action_id: action for action in turn.actions}
    ordered, attached_thoughts = _order_segments(turn, actions)
    traces_by_action = {trace.action_id: trace for trace in turn.semantic_traces if trace.action_id}

    entries: list[ActionEntry] = []
    # 外部工具动作的降级形态（工具名与参数摘要一行），按条目下标登记。
    degraded: dict[int, str] = {}
    failed: set[int] = set()
    rendered_traces: set[str] = set()
    for kind, value in ordered:
        if kind == "thought":
            entries.append(_thought_entry(value, config))
            continue
        action = actions[value]
        entries.extend(_thought_entry(text, config) for text in attached_thoughts.get(value, ()))
        if action.tool_kind:
            trace = traces_by_action.get(value)
            if trace is not None:
                rendered_traces.add(value)
                summary = trace_summary(trace)
                if summary is not None:
                    entries.append(ActionEntry(kind="trace", text=summary))
            continue
        if not (action.tool_name or action.results):
            # 只有显式附属思考的动作：思考已经渲染，没有工具可以呈现。
            continue
        full, head = _tool_texts(action, config)
        degraded[len(entries)] = head
        if _action_failed(action):
            failed.add(len(entries))
        entries.append(ActionEntry(kind="tool", text=full))

    # 没有对应动作的轨迹（例如块不带结构化事件）按原顺序补在末尾。
    for trace in turn.semantic_traces:
        if trace.action_id is not None and trace.action_id in rendered_traces:
            continue
        summary = trace_summary(trace)
        if summary is not None:
            entries.append(ActionEntry(kind="trace", text=summary))

    return _fit_turn_limit(entries, degraded, failed, config.max_turn_action_chars)


def trace_summary(trace: TraceItem) -> str | None:
    """MTP 动作的一行摘要；CALL、WRITE、UPDATE 保留在轨迹中，但不渲染进生成。"""
    action = trace.action.upper()
    if action == "SEARCH":
        return f'SEARCH: "{trace.query or ""}"'
    if action == "READ":
        return f"READ: {trace.target or '(unknown)'}"
    if action == "RUN":
        return f"RUN: {trace.tool or '(unknown)'} ({trace.status or 'unknown'})"
    return None


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------


def _order_segments(
    turn: TurnRecord, actions: Mapping[str, AgentAction]
) -> tuple[list[tuple[Literal["thought", "action"], str]], dict[str, list[str]]]:
    """按事件序号排出思考与动作的先后，并收集显式附属于各动作的思考。

    动作排在属于它的第一个事件处；事件到动作的归属沿用归约器的同一条规则，
    只是未附属的思考不吸附到最近动作。
    """
    ordered: list[tuple[Literal["thought", "action"], str]] = []
    attached: dict[str, list[str]] = {}
    placed: set[str] = set()
    last_action_id: str | None = None
    for event in sorted(turn.turn_events, key=lambda item: item.sequence):
        if event.kind == "thought" and not event.content.strip():
            # 空白思考（例如只有签名的思考块）没有可供提取的内容。
            continue
        if event.kind == "thought" and event.action_id not in actions:
            ordered.append(("thought", event.content))
            continue
        if event.kind not in {"thought", "tool_call", "tool_result"}:
            continue
        action_id = ActionReducer.resolve_action_id(event, last_action_id)
        if action_id is None or action_id not in actions:
            continue
        if event.kind == "thought":
            attached.setdefault(action_id, []).append(event.content)
        if action_id not in placed:
            ordered.append(("action", action_id))
            placed.add(action_id)
        last_action_id = action_id
    # 构造时只给出动作、没有对应事件的轮次，动作按归约顺序补在末尾。
    ordered.extend(("action", action_id) for action_id in actions if action_id not in placed)
    return ordered, attached


def _thought_entry(text: str, config: GenerationTranscriptConfig) -> ActionEntry:
    return ActionEntry(
        kind="thought", text=f"THOUGHT: {_clip(text.strip(), config.max_thought_chars)}"
    )


def _tool_texts(action: AgentAction, config: GenerationTranscriptConfig) -> tuple[str, str]:
    """返回外部工具动作的完整形态与降级形态（首行即降级形态）。"""
    head = f"TOOL: {action.tool_name or '(unknown)'}"
    args = _args_summary(action.tool_args, config.max_args_chars)
    if args:
        head = f"{head} {args}"
    lines = [head]
    for result in action.results:
        label = "RESULT (error)" if _is_error(result.status) else "RESULT"
        content = _clip(result.content.strip(), config.max_result_chars) or "(empty)"
        lines.append(f"{label}: {content}")
    return "\n".join(lines), head


def _args_summary(tool_args: Mapping[str, Any] | None, limit: int) -> str:
    if not tool_args:
        return ""
    rendered = json.dumps(dict(tool_args), ensure_ascii=False, default=str)
    return _clip(rendered, limit)


def _action_failed(action: AgentAction) -> bool:
    return _is_error(action.status) or any(_is_error(result.status) for result in action.results)


def _is_error(status: str | None) -> bool:
    return status is not None and status.strip().lower() in _ERROR_STATUSES


def _clip(text: str, limit: int) -> str:
    """超过上限时保留首尾、省略中段，并标出省略的字符数。"""
    if len(text) <= limit:
        return text
    tail = limit // 2
    head = limit - tail
    omitted = len(text) - limit
    return f"{text[:head]} …[{omitted} chars omitted]… {text[len(text) - tail :]}"


def _fit_turn_limit(
    entries: list[ActionEntry],
    degraded: Mapping[int, str],
    failed: set[int],
    limit: int,
) -> list[ActionEntry]:
    """让动作段满足单轮上限。

    先把外部工具动作全部降为一行，再按优先级逐个恢复完整形态：出错的动作、
    其余动作，各自由近及远；放不下的动作跳过，继续尝试更短的动作。全部降为
    一行后仍超出时，按优先级从低到高省略整条，直到连同省略说明放得下。
    """
    if _length(entry.text for entry in entries) <= limit:
        return entries
    texts = [degraded.get(index, entry.text) for index, entry in enumerate(entries)]
    remaining = limit - _length(texts)
    recent_first = sorted(degraded, reverse=True)
    priority = [index for index in recent_first if index in failed] + [
        index for index in recent_first if index not in failed
    ]
    for index in priority:
        extra = len(entries[index].text) - len(texts[index])
        if extra <= remaining:
            texts[index] = entries[index].text
            remaining -= extra
    fitted = [
        ActionEntry(kind=entry.kind, text=text) for entry, text in zip(entries, texts, strict=True)
    ]
    if remaining >= 0:
        return fitted
    return _omit_until_fit(fitted, failed, limit)


def _omit_until_fit(entries: list[ActionEntry], failed: set[int], limit: int) -> list[ActionEntry]:
    """按优先级从低到高、同级由远及近省略整条，并在段首说明省略的条目数。"""

    def oldest_first(kind: ActionEntryKind, *, failed_tools: bool = False) -> list[int]:
        return [
            index
            for index, entry in enumerate(entries)
            if entry.kind == kind and (kind != "tool" or (index in failed) == failed_tools)
        ]

    drop_order = [
        *oldest_first("tool"),
        *oldest_first("trace"),
        *oldest_first("tool", failed_tools=True),
        *oldest_first("thought"),
    ]
    total = _length(entry.text for entry in entries)
    dropped: set[int] = set()
    for index in drop_order:
        if dropped and total + len(_omitted_text(len(dropped))) <= limit:
            break
        dropped.add(index)
        total -= len(entries[index].text)
    kept = [entry for index, entry in enumerate(entries) if index not in dropped]
    return [ActionEntry(kind="omitted", text=_omitted_text(len(dropped))), *kept]


def _omitted_text(count: int) -> str:
    return f"OMITTED: {count} entries over the per-turn limit"


def _length(texts: Iterable[str]) -> int:
    return sum(len(text) for text in texts)


__all__ = [
    "ActionEntry",
    "ActionEntryKind",
    "render_action_entries",
    "trace_summary",
]
