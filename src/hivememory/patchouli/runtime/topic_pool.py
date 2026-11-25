"""关闭时保存的话题池快照。

正常关闭时，关闭时的沉淀与保存生成工作之后，整个话题池写进这个文件；下次启动时读回、读回即
删除，话题带着交互记录、折叠记录、沉淀进度与写入记录跨过重启，崩溃之后也不会读回一份过时的
话题池。只在正常关闭时写出：话题的沉淀进度在生成队列接纳时就推进，而生成工作只在正常关闭时
保存，单独让话题随时落盘，崩溃之后读回的进度会越过随进程丢失的生成工作，那些交互再也不会
沉淀。每个话题以 ``TopicData`` 的 JSON 形式保存，读回时以完整构造校验，不另设序列化；读不回
的内容不丢弃，改存到同目录的 rejected 文件。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from hivememory.config.files import rejected_path, write_atomically
from hivememory.core.models import TopicData

logger = logging.getLogger(__name__)

TOPIC_POOL_FORMAT_VERSION = 1


class TopicPoolFileError(Exception):
    """话题池快照无法解析，或文件格式版本不认识；原文件已改名为 ``rejected_path``。"""

    def __init__(self, rejected_path: Path) -> None:
        super().__init__(f"话题池快照无法读回，已改存为 {rejected_path.name}")
        self.rejected_path = rejected_path


@dataclass(frozen=True)
class SavedTopicPoolReport:
    """关闭时保存话题池的结果；``saved`` 是要保存的话题数，文件没写出时它们随进程丢失。"""

    saved: int = 0
    file_written: bool = False


@dataclass(frozen=True)
class RestoredTopicPoolReport:
    """启动时读回话题池的结果。"""

    read: int = 0
    restored: int = 0
    rejected: int = 0
    rejected_file: str | None = None


def save_topic_pool(path: Path, topics: Sequence[TopicData]) -> SavedTopicPoolReport:
    """把话题池写成快照，供下次启动读回；话题池为空时不写文件，写不出时只记日志。"""
    if not topics:
        return SavedTopicPoolReport()
    try:
        _write(path, [topic.model_dump(mode="json") for topic in topics])
    except (OSError, ValueError):
        # 磁盘错误，或话题中有不能编码为 UTF-8 的文本（例如模型给出的标题里半个代理对）：
        # 关闭照常完成。关闭时的沉淀已经把未沉淀的交互交给生成，丢失的只是话题池本身。
        logger.error(
            "保存话题池失败: reason=topic_pool_write_failed, path=%s, count=%d",
            path,
            len(topics),
            exc_info=True,
        )
        return SavedTopicPoolReport(saved=len(topics))
    return SavedTopicPoolReport(saved=len(topics), file_written=True)


def restore_topic_pool(
    path: Path, restore: Callable[[list[TopicData]], list[TopicData]]
) -> RestoredTopicPoolReport:
    """读回并删除话题池快照，把校验通过的话题交给 ``restore``。

    ``restore`` 读回话题并返回超出容量、没有读回的话题。整份文件读不回时改名为 rejected
    文件、话题池从空开始；单个话题校验失败或超出容量时，这些话题写进 rejected 文件，其余
    照常读回。
    """
    try:
        items = _take(path)
    except TopicPoolFileError as error:
        logger.error(
            "读回话题池失败: reason=topic_pool_unreadable, rejected_file=%s",
            error.rejected_path,
            exc_info=True,
        )
        return RestoredTopicPoolReport(rejected_file=str(error.rejected_path))
    if items is None:
        return RestoredTopicPoolReport()

    topics: list[TopicData] = []
    rejected: list[object] = []
    for item in items:
        try:
            topics.append(TopicData.model_validate(item))
        except ValidationError:
            logger.warning("话题池中的话题无法读回: reason=topic_pool_item_invalid", exc_info=True)
            rejected.append(item)
    over_capacity = restore(topics)
    rejected.extend(topic.model_dump(mode="json") for topic in over_capacity)
    rejected_file = None
    if rejected:
        rejected_file = rejected_path(path)
        _write(rejected_file, rejected)
        logger.error(
            "部分话题没有读回: reason=topic_pool_items_rejected, invalid=%d, "
            "over_capacity=%d, rejected_file=%s",
            len(rejected) - len(over_capacity),
            len(over_capacity),
            rejected_file,
        )
    return RestoredTopicPoolReport(
        read=len(items),
        restored=len(topics) - len(over_capacity),
        rejected=len(rejected),
        rejected_file=str(rejected_file) if rejected_file is not None else None,
    )


def _take(path: Path) -> list[object] | None:
    """读出并删除话题池快照，返回其中的话题项；文件不存在时返回 ``None``。

    先删除再交给调用方读回：读回中途崩溃时这些话题丢失，与其他崩溃相同，但下一次启动不会
    读回同一份快照、把已经沉淀的交互再沉淀一次。文件无法解析或格式版本不认识时，整个文件
    改名保留在原目录，抛出 :class:`TopicPoolFileError`。
    """
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("topic pool document must be an object")
        if document.get("format_version") != TOPIC_POOL_FORMAT_VERSION:
            raise ValueError(f"unknown topic pool format: {document.get('format_version')!r}")
        items = document.get("topics")
        if not isinstance(items, list):
            raise ValueError("topic pool document has no topics list")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        rejected = rejected_path(path)
        path.replace(rejected)
        raise TopicPoolFileError(rejected) from error
    path.unlink()
    return items


def _write(path: Path, items: list[object]) -> None:
    document = {"format_version": TOPIC_POOL_FORMAT_VERSION, "topics": items}
    write_atomically(path, json.dumps(document, ensure_ascii=False).encode("utf-8"))


__all__ = [
    "TOPIC_POOL_FORMAT_VERSION",
    "RestoredTopicPoolReport",
    "SavedTopicPoolReport",
    "restore_topic_pool",
    "save_topic_pool",
]
