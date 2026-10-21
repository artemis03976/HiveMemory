"""Claude Code 会话记录（JSONL）的读取器：切分轮次并映射为导入事件。

会话记录的格式不是 Claude Code 公开承诺的接口，解析只集中在本模块：
不认识的条目、块与字段一律跳过，不报错。依赖的结构：

- 条目的 ``type`` 为 ``user``、``assistant`` 或 ``system``，另有大量内部条目
  （请求记录、标题、文件快照等），只读取前两类；``isSidechain``（子代理）、
  ``isMeta``（命令说明等元条目）与 ``isCompactSummary``（压缩摘要）的条目跳过，
  ``system`` 条目（压缩边界、API 错误等）跳过。
- 一轮从一条由人发出的用户消息开始：``user`` 条目，``turnOrigin`` 为
  ``human``，内容不是工具结果。没有该字段的文本用户条目是 Claude Code 注入的
  内容（技能说明、中断标记等），其他取值是后台任务通知等，都不开启新轮。
  轮次标识取这条消息的 ``uuid``，外部会话标识取它的 ``sessionId``（缺失时取
  会话记录文件名，Claude Code 以会话标识命名会话记录）。
- Claude Code 在用户中断时写入 ``[Request interrupted by user...]`` 文本的用户
  条目，读到即视为该轮已被中断。
- 助手条目的内容块：``text`` → 助手消息，``thinking`` → 思考（没有正文的跳过），
  ``tool_use`` → 工具调用（标识与输入）；用户条目中的 ``tool_result`` → 工具
  结果（``is_error`` 为真时状态为 ``error``）。同一条 API 消息按内容块分成多个
  条目增量写入，每个条目带块在消息中的位置 ``apiBlockIndex``；重复写入的块按
  消息标识与块位置去重（没有块位置时按块内容），工具调用与结果按调用标识去重。
- 思考只有 Claude Code 写入本地记录的部分：较新的模型默认不返回思考正文，
  只有签名，没有正文的思考块跳过。
- ``assistant_final_text`` 取最后一次工具结果之后的助手正文，``model_used``
  取最后一条助手条目的模型（Claude Code 为合成消息标注的 ``<synthetic>`` 除外）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from connector.core.transcript import TranscriptTurn

SOURCE = "claude-code"

_INTERRUPT_MARKER = re.compile(r"^\[Request interrupted by user[^\]]*\]$")
_SYNTHETIC_MODEL = "<synthetic>"


@dataclass
class _TurnBuilder:
    turn_id: str
    conversation_id: str
    start_offset: int
    user_message: str
    end_offset: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    model_used: str | None = None
    interrupted: bool = False
    seen_blocks: set[str] = field(default_factory=set)

    def add_event(self, kind: str, content: str, **fields: Any) -> None:
        event = {"kind": kind, "sequence": len(self.events) + 1, "content": content}
        event.update({key: value for key, value in fields.items() if value is not None})
        self.events.append(event)

    def build(self, *, closed_by_next_turn: bool) -> TranscriptTurn:
        return TranscriptTurn(
            turn_id=self.turn_id,
            conversation_id=self.conversation_id,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            user_message=self.user_message,
            events=self.events,
            assistant_final_text=self._final_text(),
            model_used=self.model_used,
            closed_by_next_turn=closed_by_next_turn,
            interrupted=self.interrupted,
        )

    def _final_text(self) -> str | None:
        """最后一次工具结果之后的助手正文；没有时为 ``None``。"""
        texts: list[str] = []
        for event in self.events:
            if event["kind"] == "tool_result":
                texts = []
            elif event["kind"] == "assistant_message":
                texts.append(event["content"])
        return "\n".join(texts) if texts else None


class ClaudeCodeReader:
    """读取 Claude Code 的会话记录。"""

    source = SOURCE

    def read(self, path: str, offset: int) -> list[TranscriptTurn]:
        turns: list[TranscriptTurn] = []
        current: _TurnBuilder | None = None
        for line_start, line_end, entry in _entries(path, offset):
            if _is_human_prompt(entry):
                if current is not None:
                    turns.append(current.build(closed_by_next_turn=True))
                current = _TurnBuilder(
                    turn_id=str(entry["uuid"]),
                    conversation_id=str(entry.get("sessionId") or Path(path).stem),
                    start_offset=line_start,
                    user_message=_prompt_text(entry),
                )
            elif current is not None:
                _apply_entry(current, entry)
            # 第一轮之前的条目属于游标之前、已经提交的轮次。
            if current is not None:
                current.end_offset = line_end
        if current is not None:
            turns.append(current.build(closed_by_next_turn=False))
        return turns

    def started_at(self, path: str) -> float | None:
        # 逐行读取，取到第一个带时间的条目即停止，不读取整份会话记录。
        for _, _, entry in _entries(path, 0):
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, str):
                parsed = _parse_timestamp(timestamp)
                if parsed is not None:
                    return parsed
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
        if isinstance(entry, dict) and _is_human_prompt(entry):
            return str(entry["uuid"])
        return None

    def last_turn_id(self, path: str, offset: int) -> str | None:
        """``offset`` 之后最后一条由人发出的消息所开启的轮次；hook 据此写结束记录。

        hook 同步运行，HiveMemory 长时间不可用时游标之后可能积压大量条目，
        因此只解析可能是用户条目的行。
        """
        turn_id: str | None = None
        for _, _, entry in _entries(path, offset, prefilter=b'"user"'):
            if _is_human_prompt(entry):
                turn_id = str(entry["uuid"])
        return turn_id


# ---------------------------------------------------------------------------
# 条目解析
# ---------------------------------------------------------------------------


def _entries(
    path: str, offset: int, *, prefilter: bytes | None = None
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """逐行产出 (行首位置, 行尾之后位置, 条目)；只读完整行，跳过无法解析的行。

    ``prefilter`` 给出时，不含该字节串的行不解析，只推进位置。
    """
    try:
        stream = open(path, "rb")
    except FileNotFoundError:
        return
    with stream:
        stream.seek(offset)
        position = offset
        for raw_line in stream:
            line_start = position
            position += len(raw_line)
            if not raw_line.endswith(b"\n"):
                break
            if prefilter is not None and prefilter not in raw_line:
                continue
            try:
                entry = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(entry, dict):
                yield line_start, position, entry


def _content_blocks(entry: dict[str, Any]) -> list[Any]:
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def _skipped(entry: dict[str, Any]) -> bool:
    return bool(entry.get("isSidechain") or entry.get("isMeta") or entry.get("isCompactSummary"))


def _is_human_prompt(entry: dict[str, Any]) -> bool:
    if entry.get("type") != "user" or _skipped(entry) or not entry.get("uuid"):
        return False
    if entry.get("turnOrigin") != "human":
        return False
    blocks = [block for block in _content_blocks(entry) if isinstance(block, dict)]
    if any(block.get("type") == "tool_result" for block in blocks):
        return False
    if _interrupt_marker(blocks):
        return False
    return any(
        (block.get("type") == "text" and str(block.get("text", "")).strip())
        or block.get("type") == "image"
        for block in blocks
    )


def _interrupt_marker(blocks: list[dict[str, Any]]) -> bool:
    return any(
        block.get("type") == "text" and _INTERRUPT_MARKER.match(str(block.get("text", "")).strip())
        for block in blocks
    )


def _prompt_text(entry: dict[str, Any]) -> str:
    parts = []
    for block in _content_blocks(entry):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif block.get("type") == "image":
            parts.append("[image]")
    return "\n".join(part for part in parts if part)


def _apply_entry(turn: _TurnBuilder, entry: dict[str, Any]) -> None:
    if _skipped(entry):
        return
    blocks = [block for block in _content_blocks(entry) if isinstance(block, dict)]
    if entry.get("type") == "user":
        if _interrupt_marker(blocks):
            turn.interrupted = True
        for block in blocks:
            if block.get("type") == "tool_result":
                _add_tool_result(turn, block)
    elif entry.get("type") == "assistant":
        message = entry.get("message")
        model = message.get("model") if isinstance(message, dict) else None
        if isinstance(model, str) and model and model != _SYNTHETIC_MODEL:
            turn.model_used = model
        # 同一条 API 消息分成多个条目时共享消息标识；缺失时退回条目自身的标识。
        message_id = str(
            (message.get("id") if isinstance(message, dict) else None) or entry.get("uuid") or ""
        )
        block_index = entry.get("apiBlockIndex")
        for block in blocks:
            position = (
                f"index:{block_index}"
                if isinstance(block_index, int) and len(blocks) == 1
                else None
            )
            _add_assistant_block(turn, message_id, position, block)


def _add_assistant_block(
    turn: _TurnBuilder, message_id: str, position: str | None, block: dict[str, Any]
) -> None:
    block_type = block.get("type")
    if block_type == "tool_use":
        action_id = str(block.get("id") or "")
        key = f"tool_use:{action_id}"
        if not action_id or key in turn.seen_blocks:
            return
        turn.seen_blocks.add(key)
        tool_input = block.get("input")
        turn.add_event(
            "tool_call",
            "",
            action_id=action_id,
            tool_name=str(block.get("name") or ""),
            tool_args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
        )
        return
    if block_type == "text":
        text = str(block.get("text", ""))
        kind = "assistant_message"
    elif block_type == "thinking":
        text = str(block.get("thinking", ""))
        kind = "thought"
    else:
        return
    # 块位置唯一确定消息中的一个块；同一消息中正文相同的两个块因此都会保留。
    key = f"{message_id}:{position or f'{block_type}:{text}'}"
    if not text.strip() or key in turn.seen_blocks:
        return
    turn.seen_blocks.add(key)
    turn.add_event(kind, text)


def _add_tool_result(turn: _TurnBuilder, block: dict[str, Any]) -> None:
    action_id = str(block.get("tool_use_id") or "")
    key = f"tool_result:{action_id}"
    if not action_id or key in turn.seen_blocks:
        return
    turn.seen_blocks.add(key)
    turn.add_event(
        "tool_result",
        _tool_result_text(block.get("content")),
        action_id=action_id,
        status="error" if block.get("is_error") else None,
    )


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, dict) and item.get("type") == "image":
            parts.append("[image]")
    return "\n".join(parts)


def _parse_timestamp(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
