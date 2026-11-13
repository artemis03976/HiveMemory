"""关闭时未执行的记忆生成工作的保存文件。

退出时不再等待提取与合并的模型调用：队列中还没执行的工作写进这个文件，下次启动时读回、
重新接纳。文件只保存每项工作的 ``kind``、``schema_version`` 与规范载荷，载荷沿用生成
队列的工作信封编码，不另设序列化。读不回的内容不丢弃，改存到同目录的 rejected 文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hivememory.config.files import rejected_path, write_atomically

SAVED_WORK_FORMAT_VERSION = 1


class SavedWorkFileError(Exception):
    """保存文件无法解析，或文件格式版本不认识；原文件已改名为 ``rejected_path``。"""

    def __init__(self, rejected_path: Path) -> None:
        super().__init__(f"保存文件无法读回，已改存为 {rejected_path.name}")
        self.rejected_path = rejected_path


@dataclass(frozen=True)
class SavedWorkReport:
    """关闭时保存工作的结果。"""

    saved_by_source: dict[str, int] = field(default_factory=dict)
    # 关闭时已经开始写入记忆、等它写完的任务数。
    waited_for_commit: int = 0
    file_written: bool = False

    @property
    def saved(self) -> int:
        return sum(self.saved_by_source.values())


@dataclass(frozen=True)
class RestoredWorkReport:
    """启动时读回保存工作的结果。"""

    read: int = 0
    resubmitted: int = 0
    rejected: int = 0
    rejected_file: str | None = None


def write_saved_work(path: Path, items: list[dict[str, object]]) -> None:
    """把待保存的工作原子写入 ``path``；调用方保证 ``items`` 非空。"""
    document = {"format_version": SAVED_WORK_FORMAT_VERSION, "work": items}
    write_atomically(path, json.dumps(document, ensure_ascii=False).encode("utf-8"))


def take_saved_work(path: Path) -> list[object] | None:
    """读出并删除保存文件，返回其中的工作项；文件不存在时返回 ``None``。

    先删除再交给调用方接纳：接纳中途崩溃时这些工作丢失，与其他崩溃相同，但不会在下一次
    启动重复生成。文件无法解析或格式版本不认识时，整个文件改名保留在原目录，抛出
    :class:`SavedWorkFileError`。
    """
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("saved work document must be an object")
        if document.get("format_version") != SAVED_WORK_FORMAT_VERSION:
            raise ValueError(f"unknown saved work format: {document.get('format_version')!r}")
        items = document.get("work")
        if not isinstance(items, list):
            raise ValueError("saved work document has no work list")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        rejected = rejected_path(path)
        path.replace(rejected)
        raise SavedWorkFileError(rejected) from error
    path.unlink()
    return items


def write_rejected_work(path: Path, items: list[object]) -> Path:
    """把读不回或接纳被拒绝的工作项写到保存文件旁边，返回写入的路径。"""
    rejected = rejected_path(path)
    document = {"format_version": SAVED_WORK_FORMAT_VERSION, "work": items}
    write_atomically(rejected, json.dumps(document, ensure_ascii=False).encode("utf-8"))
    return rejected


__all__ = [
    "SAVED_WORK_FORMAT_VERSION",
    "RestoredWorkReport",
    "SavedWorkFileError",
    "SavedWorkReport",
    "take_saved_work",
    "write_rejected_work",
    "write_saved_work",
]
