"""Codex 会话记录（rollout JSONL）的读取器：切分轮次并映射为导入事件。

会话记录在 ``CODEX_HOME/sessions/<年>/<月>/<日>/rollout-<时间>-<线程ID>.jsonl``，
续会话在同一文件末尾追加。每行是 ``{"type", "payload"}``，读取依赖的结构：

- 第一行 ``session_meta`` 的 ``payload.id`` 是线程 ID，作为外部会话标识；
  ``payload.timestamp`` 是会话开始时间。
- 一轮以 ``event_msg`` 的 ``task_started`` 开始（``payload.turn_id`` 为轮次标识，
  子 Agent 会话记录中另有指向派生它的那一轮的 ``root_turn_id``），以同一
  ``turn_id`` 的 ``task_complete`` 或 ``turn_aborted`` 结束：``task_complete``
  带 ``error`` 时这一轮失败，否则完成；``turn_aborted`` 是被中断或取代。结束
  记录写在这一轮的最后，读到它时这一轮的内容已经写完。之后出现下一轮而没有
  结束记录的一轮按被打断处理。
- 这一轮的内容只读 ``event_msg`` 的 ``item_completed``：每个已完成条目一行，
  带结构化的类型与字段。同一内容另有 ``response_item`` 形式的模型输入输出，
  其中工具参数是 hook 改写前的，不读取。``turn_context`` 的 ``payload.model``
  是这一轮使用的模型。
- 条目映射：``UserMessage`` 的文本是用户消息（一轮中有多条时按顺序以空行
  拼接，图片写占位）；``AgentMessage`` → 助手消息；``Reasoning`` → 思考
  （有原文时取原文，否则取摘要，都没有时跳过）；``McpToolCall``、
  ``CommandExecution``、``DynamicToolCall``、``FileChange``、``WebSearch`` →
  工具调用（标识、工具名与参数，结局取条目状态）；``CollabAgentToolCall`` 中
  等待子 Agent 的调用里已完成的子 Agent 消息 → 子 Agent 消息，其余子 Agent
  调用 → 工具调用。其他类型的条目（例如 hook 提示、上下文压缩、图片生成）
  不读取。
- 本插件 MCP server 的四个记忆工具：``McpToolCall`` 的 ``server`` 为
  ``hivememory`` 时，按 ``connector.core.memory_tools`` 的规则写为能力调用
  （回声标记）。识别只看 server 名与工具名，与 ``hooks/codex.json`` 的 matcher
  相同：用户在自己的配置中覆盖这个 server（例如改地址）时，Codex 记下的调用
  不带插件 ID，按插件 ID 识别会漏掉回声标记。
- 回复正文不随轮次上传；``assistant_final_text`` 按顺序以空行拼接该轮全部
  助手正文，只作为后续轮次的最近上文。

会话记录格式依据 Codex 0.167.0；不认识的条目与字段一律跳过，不报错。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from connector.core.config import Harness
from connector.core.memory_tools import MEMORY_TOOLS, tool_call_fields
from connector.core.state import Outcome
from connector.core.transcript import TranscriptTurn, complete_size, read_entries

HARNESS = Harness(
    source="codex",
    principal_id="harness:codex",
    default_agent_id="codex",
    token_env="HIVEMEMORY_CODEX_TOKEN",
    state_dir_name="codex-connector",
)

# 与 codex.mcp.json 的 server 名保持一致。
MCP_SERVER_NAME = "hivememory"


def hook_tool_name(tool: str) -> str:
    """Codex 在 hook 输入中给本插件 MCP 工具的名字，``hooks/codex.json`` 的 matcher 依据它。"""
    return f"mcp__{MCP_SERVER_NAME}__{tool}"


@dataclass
class _TurnBuilder:
    turn_id: str
    conversation_id: str
    start_offset: int
    end_offset: int = 0
    user_messages: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_used: str | None = None
    outcome: Outcome | None = None

    def add_event(self, kind: str, content: str, **fields: Any) -> None:
        event = {"kind": kind, "sequence": len(self.events) + 1, "content": content}
        event.update({key: value for key, value in fields.items() if value is not None})
        self.events.append(event)

    def build(self, *, closed_by_next_turn: bool) -> TranscriptTurn:
        texts = [event["content"] for event in self.events if event["kind"] == "assistant_message"]
        return TranscriptTurn(
            turn_id=self.turn_id,
            conversation_id=self.conversation_id,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            user_message="\n\n".join(self.user_messages),
            events=self.events,
            assistant_final_text="\n\n".join(texts) if texts else None,
            model_used=self.model_used,
            closed_by_next_turn=closed_by_next_turn,
            interrupted=self.outcome == "cancelled",
            looks_complete=self.outcome is not None,
            outcome=self.outcome,
        )


class CodexReader:
    """读取 Codex 的会话记录。"""

    source = HARNESS.source

    def read(self, path: str, offset: int) -> list[TranscriptTurn]:
        conversation_id = self.thread_id(path)
        if conversation_id is None:
            return []
        turns: list[TranscriptTurn] = []
        current: _TurnBuilder | None = None
        for line_start, line_end, entry in read_entries(path, offset):
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            kind = payload.get("type") if entry.get("type") == "event_msg" else None
            if kind == "task_started" and isinstance(payload.get("turn_id"), str):
                if current is not None:
                    turns.append(current.build(closed_by_next_turn=True))
                current = _TurnBuilder(
                    turn_id=payload["turn_id"],
                    conversation_id=conversation_id,
                    start_offset=line_start,
                )
            elif current is not None:
                _apply_entry(current, entry.get("type"), kind, payload)
            # 第一轮之前的条目属于游标之前、已经提交的轮次。
            if current is not None:
                current.end_offset = line_end
        if current is not None:
            turns.append(current.build(closed_by_next_turn=False))
        return turns

    def started_at(self, path: str) -> float | None:
        meta = _session_meta(path)
        timestamp = meta.get("timestamp") if meta is not None else None
        if not isinstance(timestamp, str):
            return None
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def turn_id_at(self, path: str, offset: int) -> str | None:
        try:
            with open(path, "rb") as stream:
                stream.seek(offset)
                line = stream.readline()
        except FileNotFoundError:
            return None
        try:
            entry = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        payload = entry.get("payload") if isinstance(entry, dict) else None
        if (
            entry.get("type") == "event_msg"
            and isinstance(payload, dict)
            and payload.get("type") == "task_started"
            and isinstance(payload.get("turn_id"), str)
        ):
            return payload["turn_id"]
        return None

    def existing_offset(self, path: str) -> int:
        """接入之前已经开始的会话首次见到时游标的位置：最后一轮仍在进行时放在它的开头。

        Codex 先写下一轮的开始记录，再触发 ``SessionStart``（``codex exec`` 的第一轮
        如此），首次见到会话时这一轮往往已经开始。游标放在它的开头，这一轮结束后
        照常导入，``PreToolUse`` 也读得到它；之前已经结束的轮次不回填。
        """
        turns = self.read(path, 0)
        if turns and turns[-1].outcome is None:
            return turns[-1].start_offset
        return complete_size(path)

    def thread_id(self, path: str) -> str | None:
        """会话记录所属线程的 ID；文件不存在或第一行不是会话元数据时为 ``None``。"""
        meta = _session_meta(path)
        thread = meta.get("id") if meta is not None else None
        return thread if isinstance(thread, str) and thread else None

    def root_turn_id(self, path: str, turn_id: str) -> str | None:
        """子 Agent 会话记录中一轮的根轮：派生这个子 Agent 的那一轮。

        找不到这一轮时为 ``None``；顶层会话中的一轮没有单独的根轮，返回它自己。
        """
        for _, _, entry in read_entries(path, 0, prefilter=b"task_started"):
            payload = entry.get("payload")
            if (
                entry.get("type") == "event_msg"
                and isinstance(payload, dict)
                and payload.get("type") == "task_started"
                and payload.get("turn_id") == turn_id
            ):
                root = payload.get("root_turn_id")
                return root if isinstance(root, str) and root else turn_id
        return None


# ---------------------------------------------------------------------------
# 条目解析
# ---------------------------------------------------------------------------


def _session_meta(path: str) -> dict[str, Any] | None:
    for _, _, entry in read_entries(path, 0):
        payload = entry.get("payload")
        if entry.get("type") == "session_meta" and isinstance(payload, dict):
            return payload
        return None
    return None


def _apply_entry(
    turn: _TurnBuilder, entry_type: Any, kind: str | None, payload: dict[str, Any]
) -> None:
    if entry_type == "turn_context":
        model = payload.get("model")
        if isinstance(model, str) and model:
            turn.model_used = model
        return
    if payload.get("turn_id") not in (None, turn.turn_id):
        return
    if kind == "item_completed":
        item = payload.get("item")
        if isinstance(item, dict):
            _apply_item(turn, item)
    elif kind == "task_complete":
        turn.outcome = "failed" if payload.get("error") else "completed"
    elif kind == "turn_aborted":
        turn.outcome = "cancelled"


def _apply_item(turn: _TurnBuilder, item: dict[str, Any]) -> None:
    item_type = item.get("type")
    if item_type == "UserMessage":
        text = _user_text(item.get("content"))
        if text:
            turn.user_messages.append(text)
    elif item_type == "AgentMessage":
        text = _agent_text(item.get("content"))
        if text.strip():
            turn.add_event("assistant_message", text)
    elif item_type == "Reasoning":
        text = "\n".join(_strings(item.get("raw_content")) or _strings(item.get("summary_text")))
        if text.strip():
            turn.add_event("thought", text)
    elif item_type == "CollabAgentToolCall":
        _add_collab_call(turn, item)
    elif item_type in _TOOL_ITEMS:
        _add_tool_call(turn, item)


def _user_text(content: Any) -> str:
    parts = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif part.get("type") in ("image", "local_image"):
            parts.append("[image]")
    return "\n".join(part for part in parts if part)


def _agent_text(content: Any) -> str:
    parts = [
        str(part.get("text", ""))
        for part in (content if isinstance(content, list) else [])
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "\n".join(parts)


def _strings(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str) and v.strip()] if isinstance(value, list) else []


def _status(value: Any) -> str | None:
    """条目状态到导入事件的结局；仍在进行或没有状态时不设置。"""
    if value == "completed":
        return "success"
    if value in ("failed", "declined", "error"):
        return "error"
    return None


def _mcp_call(item: dict[str, Any]) -> tuple[str, str | None, dict[str, Any], str | None, str]:
    server = str(item.get("server") or "")
    tool = str(item.get("tool") or "")
    memory_tool = tool if server == MCP_SERVER_NAME and tool in MEMORY_TOOLS else None
    args = item.get("arguments")
    result = item.get("result")
    error = item.get("error")
    if isinstance(error, dict):
        text = str(error.get("message", ""))
        status = "error"
    else:
        text = _content_text(result.get("content") if isinstance(result, dict) else None)
        status = "error" if isinstance(result, dict) and result.get("isError") else None
    return (
        hook_tool_name(tool) if server == MCP_SERVER_NAME else f"mcp__{server}__{tool}",
        memory_tool,
        args if isinstance(args, dict) else {"input": args},
        status or _status(item.get("status")),
        text,
    )


def _command_call(item: dict[str, Any]) -> tuple[str, None, dict[str, Any], str | None, str]:
    command = item.get("command")
    args = {"command": " ".join(command) if isinstance(command, list) else command}
    if item.get("cwd"):
        args["cwd"] = item["cwd"]
    output = item.get("aggregated_output")
    return "exec_command", None, args, _status(item.get("status")), str(output or "")


def _dynamic_call(item: dict[str, Any]) -> tuple[str, None, dict[str, Any], str | None, str]:
    namespace = item.get("namespace")
    tool = str(item.get("tool") or "")
    args = item.get("arguments")
    status = "error" if item.get("success") is False else _status(item.get("status"))
    text = str(item.get("error") or "") or _content_text(item.get("content_items"))
    name = f"{namespace}.{tool}" if namespace else tool
    return name, None, args if isinstance(args, dict) else {"input": args}, status, text


def _file_change(item: dict[str, Any]) -> tuple[str, None, dict[str, Any], str | None, str]:
    changes = item.get("changes")
    paths = sorted(changes) if isinstance(changes, dict) else []
    output = "\n".join(str(item.get(key) or "") for key in ("stdout", "stderr")).strip()
    return "apply_patch", None, {"paths": paths}, _status(item.get("status")), output


def _web_search(item: dict[str, Any]) -> tuple[str, None, dict[str, Any], str | None, str]:
    return "web_search", None, {"query": item.get("query")}, None, ""


_TOOL_ITEMS = {
    "McpToolCall": _mcp_call,
    "CommandExecution": _command_call,
    "DynamicToolCall": _dynamic_call,
    "FileChange": _file_change,
    "WebSearch": _web_search,
}


def _add_tool_call(turn: _TurnBuilder, item: dict[str, Any]) -> None:
    action_id = str(item.get("id") or "")
    if not action_id:
        return
    tool_name, memory_tool, args, status, result = _TOOL_ITEMS[item["type"]](item)
    turn.add_event(
        "tool_call",
        "",
        action_id=action_id,
        **tool_call_fields(tool_name, memory_tool, args),
        status=status,
        result=result if status is not None else None,
    )


def _add_collab_call(turn: _TurnBuilder, item: dict[str, Any]) -> None:
    states = item.get("agents_states")
    if item.get("tool") == "wait" and isinstance(states, dict):
        for state in states.values():
            message = state.get("completed") if isinstance(state, dict) else None
            if isinstance(message, str) and message.strip():
                turn.add_event("subagent_message", message)
        return
    action_id = str(item.get("id") or "")
    if not action_id:
        return
    args = {"prompt": item["prompt"]} if isinstance(item.get("prompt"), str) else {}
    turn.add_event(
        "tool_call",
        "",
        action_id=action_id,
        tool_name=str(item.get("tool") or "collab_agent"),
        tool_args=args,
        status=_status(item.get("status")),
        result="" if _status(item.get("status")) is not None else None,
    )


def _content_text(content: Any) -> str:
    parts = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, dict) and part.get("type") == "image":
            parts.append("[image]")
    return "\n".join(parts)
