"""合成的 Claude Code 会话记录：按真实条目的字段形状写 JSONL，内容全部是测试样例。

字段形状按本机 Claude Code 2.1.280–2.1.304 的会话记录核对：由人发出的消息带
``turnOrigin: human`` 与 ``promptId``，助手消息的每个内容块各占一个条目并带
``apiBlockIndex``，工具结果写在用户条目中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SESSION_ID = "session-test"
STARTED = "2026-10-21T08:00:00.000Z"


class TranscriptWriter:
    """逐条追加会话记录条目，并记录每条由人发出的消息所在的字节位置。"""

    def __init__(self, path: Path, *, session_id: str = SESSION_ID) -> None:
        self.path = path
        self.session_id = session_id
        self.prompt_offsets: dict[str, int] = {}
        self._counter = 0
        self._block_index: dict[str, int] = {}
        path.touch()

    def _entry(self, entry_type: str, **fields: Any) -> dict[str, Any]:
        self._counter += 1
        return {
            "type": entry_type,
            "uuid": fields.pop("uuid", f"entry-{self._counter}"),
            "parentUuid": None,
            "sessionId": self.session_id,
            "timestamp": fields.pop("timestamp", STARTED),
            "isSidechain": fields.pop("isSidechain", False),
            **fields,
        }

    def append(self, entry: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def prompt(self, uuid: str, text: str, **fields: Any) -> None:
        self.prompt_offsets[uuid] = self.path.stat().st_size
        fields.setdefault("turnOrigin", "human")
        self.append(
            self._entry(
                "user",
                uuid=uuid,
                promptId=f"prompt-{uuid}",
                message={"role": "user", "content": text},
                **fields,
            )
        )

    def assistant(self, *blocks: dict[str, Any], message_id: str = "msg", **fields: Any) -> None:
        """与 Claude Code 一样，一条 API 消息的每个内容块各写一个条目，带块位置。"""
        model = fields.pop("model", "claude-test")
        for block in blocks:
            index = self._block_index.get(message_id, 0)
            self._block_index[message_id] = index + 1
            self.append(
                self._entry(
                    "assistant",
                    apiBlockIndex=index,
                    message={
                        "id": message_id,
                        "role": "assistant",
                        "model": model,
                        "content": [block],
                    },
                    **fields,
                )
            )

    def tool_result(self, tool_use_id: str, content: Any, *, is_error: bool = False) -> None:
        self.append(
            self._entry(
                "user",
                message={
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content,
                            "is_error": is_error,
                        }
                    ],
                },
            )
        )

    def user_text(self, text: str, **fields: Any) -> None:
        """不是由人发出的用户条目（中断标记、元条目等），字段由调用方给出。"""
        self.append(
            self._entry(
                "user",
                message={"role": "user", "content": [{"type": "text", "text": text}]},
                **fields,
            )
        )

    def raw(self, **entry: Any) -> None:
        self.append(entry)

    def turn(self, uuid: str, question: str = "问题", answer: str = "回答") -> None:
        """一轮只有提问与回复的完整交互。"""
        self.prompt(uuid, question)
        self.assistant(text(answer), message_id=f"msg-{uuid}")


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(value: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": value, "signature": "sig"}


def tool_use(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
