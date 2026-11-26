"""短期存储端口的文件实现：每个话题一个 JSON Lines 文件。

文件是 ``<topic_dir>/<topic_id>.jsonl``，一行一条记录，只在末尾追加。目录不按 Workspace
分层：Workspace 归属写在第一条记录里，数据根的路径长度有限制，文件路径越短越好。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from hivememory.config.files import rejected_path
from hivememory.memory_library.models import StorageHealthComponent
from hivememory.memory_library.ports import ShortTermStoragePort

logger = logging.getLogger(__name__)

_LOG_SUFFIX = ".jsonl"
# rejected 文件与日志同目录、同后缀，读回时按文件名中的这一段跳过。
_REJECTED_MARKER = ".rejected-"


class JsonlShortTermStorage(ShortTermStoragePort):
    """:class:`ShortTermStoragePort` 的文件实现。

    追加不经 Python 的缓冲，写入返回时数据已经交给操作系统：进程崩溃不丢，断电可能丢失
    最后几行（不 fsync）。追加失败时把文件截回追加之前的长度，日志里不留半行。
    """

    def __init__(self, topic_dir: str | Path) -> None:
        self._dir = Path(topic_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, topic_id: str) -> Path:
        return self._dir / f"{topic_id}{_LOG_SUFFIX}"

    def append(self, topic_id: str, line: bytes) -> None:
        data = line + b"\n"
        with open(self._path(topic_id), "ab", buffering=0) as file:
            start = file.seek(0, os.SEEK_END)
            try:
                written = 0
                while written < len(data):
                    written += file.write(data[written:])
            except BaseException:
                file.truncate(start)
                raise

    def read_all(self) -> dict[str, list[bytes]]:
        logs: dict[str, list[bytes]] = {}
        for path in sorted(self._dir.glob(f"*{_LOG_SUFFIX}")):
            if _REJECTED_MARKER in path.name:
                continue
            data = path.read_bytes()
            complete = data.rfind(b"\n") + 1
            if complete < len(data):
                # 最后一行没有换行：写入中途进程退出，这条记录没有写完，之后的追加不能接在它后面。
                logger.warning(
                    "丢弃话题日志中不完整的最后一行: reason=topic_log_partial_line, "
                    "topic_id=%s, dropped_bytes=%d",
                    path.stem,
                    len(data) - complete,
                )
                with path.open("r+b") as file:
                    file.truncate(complete)
            logs[path.stem] = data[:complete].split(b"\n")[:-1]
        return logs

    def reject(self, topic_id: str) -> str:
        path = self._path(topic_id)
        rejected = rejected_path(path)
        path.replace(rejected)
        return str(rejected)

    def delete(self, topic_id: str) -> bool:
        try:
            self._path(topic_id).unlink()
        except FileNotFoundError:
            return False
        return True

    async def check_health(self) -> StorageHealthComponent:
        """报告话题目录是否可写：在目录中写入并删除一个探测文件。"""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            probe = self._dir / ".healthcheck"
            probe.write_bytes(b"")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            return StorageHealthComponent(name="short_term", healthy=False, detail=str(exc))
        return StorageHealthComponent(name="short_term", healthy=True)


__all__ = ["JsonlShortTermStorage"]
