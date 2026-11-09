"""合成的 Codex 会话记录：按真实条目的字段形状写 rollout JSONL，内容全部是测试样例。

字段形状按 Windows 上 Codex 0.167.0 写出的会话记录核对：第一行 ``session_meta``，
每轮以 ``event_msg`` 的 ``task_started`` 开始、以 ``task_complete`` 或
``turn_aborted`` 结束，内容是 ``item_completed`` 中的结构化条目；同一内容另有
``response_item`` 形式的模型输入输出。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

THREAD_ID = "019a0000-0000-7000-8000-000000000001"
STARTED = "2026-11-08T10:00:00.000Z"
PLUGIN_ID = "hivememory-connector@hivememory-local"


class RolloutWriter:
    """逐条追加会话记录条目，并记录每轮开始的字节位置。"""

    def __init__(
        self,
        directory: Path,
        *,
        thread_id: str = THREAD_ID,
        started: str = STARTED,
        parent_thread_id: str | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.path = directory / f"rollout-2026-11-08T10-00-00-{thread_id}.jsonl"
        self.turn_offsets: dict[str, int] = {}
        self._current = ""
        self._counter = 0
        meta: dict[str, Any] = {
            "id": thread_id,
            "session_id": parent_thread_id or thread_id,
            "timestamp": started,
            "cwd": "C:\\work",
            "originator": "codex_exec",
            "cli_version": "0.167.0",
            "thread_source": "subagent" if parent_thread_id else "user",
        }
        if parent_thread_id:
            meta["parent_thread_id"] = parent_thread_id
            meta["source"] = {"subagent": {"thread_spawn": {"parent_thread_id": parent_thread_id}}}
        self.raw("session_meta", meta)

    def raw(self, entry_type: str, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            entry = {"timestamp": STARTED, "type": entry_type, "payload": payload}
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def start(
        self, turn_id: str, *, root_turn_id: str | None = None, model: str = "gpt-test"
    ) -> None:
        self.turn_offsets[turn_id] = self.path.stat().st_size
        self._current = turn_id
        root = root_turn_id or turn_id
        self.raw(
            "event_msg",
            {
                "type": "task_started",
                "turn_id": turn_id,
                "root_turn_id": root,
                "turn_attribution": {
                    "turn_id": turn_id,
                    "parent_turn_id": root_turn_id,
                    "root_turn_id": root,
                },
            },
        )
        self.raw("turn_context", {"turn_id": turn_id, "root_turn_id": root, "model": model})

    def item(self, item: dict[str, Any]) -> None:
        self.raw(
            "event_msg",
            {
                "type": "item_completed",
                "thread_id": self.thread_id,
                "turn_id": self._current,
                "item": item,
            },
        )

    def user(self, text: str) -> None:
        # 模型输入中另有注入的环境上下文，读取器不应把它当作用户消息。
        self.raw(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context/>"}],
            },
        )
        self.item(
            {
                "type": "UserMessage",
                "id": self._id("user"),
                "content": [{"type": "text", "text": text}],
            }
        )

    def agent(self, text: str) -> None:
        self.item(
            {
                "type": "AgentMessage",
                "id": self._id("msg"),
                "content": [{"type": "Text", "text": text}],
            }
        )

    def reasoning(self, summary: str) -> None:
        self.item(
            {
                "type": "Reasoning",
                "id": self._id("rs"),
                "summary_text": [summary],
                "raw_content": [],
            }
        )

    def command(
        self, call_id: str, command: list[str], output: str, *, status: str = "completed"
    ) -> None:
        self.item(
            {
                "type": "CommandExecution",
                "id": call_id,
                "command": command,
                "cwd": "C:\\work",
                "status": status,
                "aggregated_output": output,
                "exit_code": 0 if status == "completed" else 1,
            }
        )

    def mcp(
        self,
        call_id: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        server: str = "hivememory",
        plugin_id: str | None = PLUGIN_ID,
        result: str = "ok",
        is_error: bool = False,
    ) -> None:
        item: dict[str, Any] = {
            "type": "McpToolCall",
            "id": call_id,
            "server": server,
            "tool": tool,
            "arguments": arguments,
            "status": "completed",
            "result": {"content": [{"type": "text", "text": result}], "isError": is_error},
        }
        if plugin_id is not None:
            item["pluginId"] = plugin_id
        self.item(item)

    def wait(self, call_id: str, child_thread_id: str, message: str) -> None:
        self.item(
            {
                "type": "CollabAgentToolCall",
                "id": call_id,
                "tool": "wait",
                "status": "completed",
                "sender_thread_id": self.thread_id,
                "receiver_thread_ids": [child_thread_id],
                "agents_states": {child_thread_id: {"completed": message}},
            }
        )

    def complete(self, *, error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "type": "task_complete",
            "turn_id": self._current,
            "root_turn_id": self._current,
            "last_agent_message": None,
        }
        if error is not None:
            payload["error"] = {"message": error, "codex_error_info": "other"}
        self.raw("event_msg", payload)

    def abort(self) -> None:
        self.raw(
            "event_msg", {"type": "turn_aborted", "turn_id": self._current, "reason": "interrupted"}
        )

    def turn(self, turn_id: str, question: str = "问题", answer: str = "回答") -> None:
        """一轮完整结束的普通对话。"""
        self.start(turn_id)
        self.user(question)
        self.agent(answer)
        self.complete()
