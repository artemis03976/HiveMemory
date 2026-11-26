"""JsonlShortTermStorage 文件端口的单元测试。

被测对象：每个话题一个 JSON Lines 文件的短期存储端口，文件写在临时目录。
保护的契约：
- 追加的行按话题读回，rejected 文件与其他文件不读回；
- 追加失败时文件截回追加之前的长度，不留半行；
- 读回时最后一行不完整（写入中途进程退出）的丢弃并从文件中截掉，之后的追加接在完整的行后；
- 移到一旁的日志不再读回；删除报告日志是否存在；健康检查报告目录是否可写。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hivememory.memory_library.adapters import short_term
from hivememory.memory_library.adapters.short_term import JsonlShortTermStorage


def test_appended_lines_are_read_back_per_topic(tmp_path):
    storage = JsonlShortTermStorage(tmp_path)
    storage.append("t1", b'{"n":1}')
    storage.append("t2", b'{"n":2}')
    storage.append("t1", b'{"n":3}')
    (tmp_path / "notes.txt").write_text("not a log", encoding="utf-8")

    assert storage.read_all() == {"t1": [b'{"n":1}', b'{"n":3}'], "t2": [b'{"n":2}']}


def test_failed_append_truncates_the_partial_line(tmp_path, monkeypatch):
    """写到一半磁盘出错时错误照常传播，文件回到追加之前的内容。"""
    storage = JsonlShortTermStorage(tmp_path)
    storage.append("t1", b'{"n":1}')
    before = (tmp_path / "t1.jsonl").read_bytes()
    real_open = open

    class _HalfWritingFile:
        def __init__(self, file):
            self._file = file

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._file.close()

        def seek(self, *args):
            return self._file.seek(*args)

        def truncate(self, size):
            return self._file.truncate(size)

        def write(self, data):
            self._file.write(data[: len(data) // 2])
            raise OSError("disk full")

    monkeypatch.setattr(
        short_term,
        "open",
        lambda *args, **kwargs: _HalfWritingFile(real_open(*args, **kwargs)),
        raising=False,
    )

    with pytest.raises(OSError, match="disk full"):
        storage.append("t1", b'{"n":2,"padding":"................"}')

    assert (tmp_path / "t1.jsonl").read_bytes() == before


def test_incomplete_last_line_is_dropped_and_truncated(tmp_path):
    """最后一行没有换行时不读回，文件截到最后一个完整的行，之后的追加从那里接着写。"""
    storage = JsonlShortTermStorage(tmp_path)
    storage.append("t1", b'{"n":1}')
    with (tmp_path / "t1.jsonl").open("ab") as file:
        file.write(b'{"n":2,"half')

    assert storage.read_all() == {"t1": [b'{"n":1}']}
    storage.append("t1", b'{"n":3}')
    assert (tmp_path / "t1.jsonl").read_bytes() == b'{"n":1}\n{"n":3}\n'


def test_rejected_log_is_kept_aside_and_not_read_back(tmp_path):
    storage = JsonlShortTermStorage(tmp_path)
    storage.append("t1", b"{broken")
    storage.append("t2", b'{"n":2}')

    rejected = storage.reject("t1")

    assert storage.read_all() == {"t2": [b'{"n":2}']}
    assert Path(rejected).read_bytes() == b"{broken\n"


def test_delete_reports_whether_the_log_existed(tmp_path):
    storage = JsonlShortTermStorage(tmp_path)
    storage.append("t1", b'{"n":1}')

    assert storage.delete("t1") is True
    assert storage.delete("t1") is False
    assert storage.read_all() == {}


@pytest.mark.asyncio
async def test_health_reports_an_unwritable_topic_directory(tmp_path):
    """话题目录被一个同名文件占住时写不进日志，健康检查报告不健康。"""
    topic_dir = tmp_path / "topics"
    storage = JsonlShortTermStorage(topic_dir)
    assert (await storage.check_health()).healthy is True

    topic_dir.rmdir()
    topic_dir.write_text("", encoding="utf-8")

    health = await storage.check_health()
    assert (health.name, health.healthy) == ("short_term", False)
