"""读取器接口：会话记录切分出的轮次，以及骨架对会话记录文件本身的假设。

骨架只假设会话记录是按行追加的文本文件：游标是字节位置，必须落在行首；
不完整的末行（仍在写入）不读取。每份会话记录如何切分轮次、如何映射为
导入事件，由各 harness 的读取器实现 ``TranscriptReader``。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from connector.core.state import Outcome

# 从文件末尾向前查找换行时每次读取的字节数。
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class TranscriptTurn:
    """会话记录中的一轮，由各 harness 的读取器按自己的格式切分。

    ``start_offset`` 是开启该轮那一行的位置，``end_offset`` 是该轮最后一行之后
    的位置（下一轮的起点，或已读取的完整行的末尾）。``closed_by_next_turn``
    表示其后已出现下一轮；``interrupted`` 表示 harness 写入了中断标记；
    ``looks_complete`` 表示读到的内容中该轮已经正常收尾（例如最后一条助手
    消息不再等待工具结果），末尾一轮据此判断内容是否已经写完。``outcome``
    是会话记录本身写明的结局（harness 为每轮写入结束记录时），没有已保存的
    结局时采用它，不再从下一轮或静止推断。
    ``events`` 是用户消息之后的导入事件，序号从 1 起递增；
    ``assistant_final_text`` 是该轮的回复正文，只作为后续轮次的最近上文上传。
    """

    turn_id: str
    conversation_id: str
    start_offset: int
    end_offset: int
    user_message: str
    events: list[dict[str, Any]] = field(default_factory=list)
    assistant_final_text: str | None = None
    model_used: str | None = None
    closed_by_next_turn: bool = False
    interrupted: bool = False
    looks_complete: bool = False
    outcome: Outcome | None = None


class TranscriptReader(Protocol):
    """harness 会话记录的读取器。"""

    source: str

    def read(self, path: str, offset: int) -> list[TranscriptTurn]:
        """从 ``offset`` 起读取完整行并切分轮次；第一轮之前的行属于更早的轮次，忽略。"""
        ...

    def started_at(self, path: str) -> float | None:
        """会话记录第一个条目的时间（Unix 秒）；文件不存在或没有条目时为 ``None``。"""
        ...

    def turn_id_at(self, path: str, offset: int) -> str | None:
        """``offset`` 处那一行开启的轮次标识；该行不开启轮次时为 ``None``。"""
        ...


def read_entries(
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


def complete_size(path: str) -> int:
    """文件中完整行的总字节数：末尾仍在写入的不完整行不计入。

    从文件末尾向前查找最后一个换行，不读取整份会话记录。
    """
    try:
        stream = open(path, "rb")
    except FileNotFoundError:
        return 0
    with stream:
        end = stream.seek(0, os.SEEK_END)
        while end > 0:
            start = max(0, end - _CHUNK)
            stream.seek(start)
            newline = stream.read(end - start).rfind(b"\n")
            if newline >= 0:
                return start + newline + 1
            end = start
    return 0


def is_line_start(path: str, offset: int) -> bool:
    """游标是否仍落在这份会话记录的某个行首；文件被重写或缩短时不成立。"""
    if offset == 0:
        return True
    try:
        if offset > os.path.getsize(path):
            return False
        with open(path, "rb") as stream:
            stream.seek(offset - 1)
            return stream.read(1) == b"\n"
    except OSError:
        return False
