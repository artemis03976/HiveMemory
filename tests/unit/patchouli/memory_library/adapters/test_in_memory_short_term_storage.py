"""
InMemoryShortTermStorage 单元测试

测试覆盖:
- get/put/pop: 基本 CRUD 操作
- list_by_user/list_all: 查询操作
- 线程安全性测试
"""

import threading

from hivememory.core.models import LogicalBlock, TurnRecord
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.buffer import SemanticBuffer


def _make_buffer(topic_id="t1", user_id="u1") -> SemanticBuffer:
    return SemanticBuffer(
        topic_id=topic_id,
        user_id=user_id,
        topic_title=f"话题{topic_id}",
        blocks=[LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))],
    )


class TestInMemoryShortTermStorageBasic:
    """InMemoryShortTermStorage 基本操作测试"""

    def setup_method(self):
        self.storage = InMemoryShortTermStorage()

    def test_get_returns_none_for_missing(self):
        result = self.storage.get("missing")
        assert result is None

    def test_put_and_get(self):
        buf = _make_buffer("t1", "u1")
        self.storage.put("t1", buf)

        result = self.storage.get("t1")
        assert result is buf

    def test_put_overwrites_existing(self):
        buf1 = _make_buffer("t1", "u1")
        buf2 = _make_buffer("t1", "u2")

        self.storage.put("t1", buf1)
        self.storage.put("t1", buf2)

        result = self.storage.get("t1")
        assert result is buf2
        assert result.user_id == "u2"

    def test_pop_returns_and_removes(self):
        buf = _make_buffer("t1", "u1")
        self.storage.put("t1", buf)

        result = self.storage.pop("t1")

        assert result is buf
        assert self.storage.get("t1") is None

    def test_pop_returns_none_for_missing(self):
        result = self.storage.pop("missing")
        assert result is None


class TestInMemoryShortTermStorageList:
    """InMemoryShortTermStorage 列表查询测试"""

    def setup_method(self):
        self.storage = InMemoryShortTermStorage()

    def test_list_by_user_returns_user_buffers(self):
        self.storage.put("t1", _make_buffer("t1", "u1"))
        self.storage.put("t2", _make_buffer("t2", "u1"))
        self.storage.put("t3", _make_buffer("t3", "u2"))

        result = self.storage.list_by_user("u1")

        assert len(result) == 2
        topic_ids = {b.topic_id for b in result}
        assert topic_ids == {"t1", "t2"}

    def test_list_by_user_returns_empty_for_no_such_user(self):
        result = self.storage.list_by_user("missing")
        assert result == []

    def test_list_all_returns_all_buffers(self):
        self.storage.put("t1", _make_buffer("t1", "u1"))
        self.storage.put("t2", _make_buffer("t2", "u2"))

        result = self.storage.list_all()

        assert len(result) == 2

    def test_list_all_returns_empty_when_empty(self):
        result = self.storage.list_all()
        assert result == []

    def test_count_returns_buffer_count(self):
        assert self.storage.count() == 0
        self.storage.put("t1", _make_buffer("t1", "u1"))
        assert self.storage.count() == 1


class TestInMemoryShortTermStorageSync:
    """InMemoryShortTermStorage 同步快捷方法测试"""

    def setup_method(self):
        self.storage = InMemoryShortTermStorage()

    def test_get_sync(self):
        buf = _make_buffer("t1", "u1")
        self.storage.put("t1", buf)

        result = self.storage._get_sync("t1")
        assert result is buf

    def test_get_sync_returns_none_for_missing(self):
        result = self.storage._get_sync("missing")
        assert result is None

    def test_put_sync(self):
        buf = _make_buffer("t1", "u1")
        self.storage._put_sync("t1", buf)

        assert self.storage.get("t1") is buf

    def test_pop_sync(self):
        buf = _make_buffer("t1", "u1")
        self.storage.put("t1", buf)

        result = self.storage._pop_sync("t1")

        assert result is buf
        assert self.storage.get("t1") is None

    def test_list_by_user_sync(self):
        self.storage.put("t1", _make_buffer("t1", "u1"))
        self.storage.put("t2", _make_buffer("t2", "u1"))
        self.storage.put("t3", _make_buffer("t3", "u2"))

        result = self.storage._list_by_user_sync("u1")

        assert len(result) == 2

    def test_list_all_sync(self):
        self.storage.put("t1", _make_buffer("t1", "u1"))
        self.storage.put("t2", _make_buffer("t2", "u2"))

        result = self.storage._list_all_sync()

        assert len(result) == 2

    def test_count(self):
        assert self.storage._count() == 0

        self.storage.put("t1", _make_buffer("t1", "u1"))
        assert self.storage._count() == 1

        self.storage.put("t2", _make_buffer("t2", "u1"))
        assert self.storage._count() == 2

        self.storage.pop("t1")
        assert self.storage._count() == 1


class TestInMemoryShortTermStorageThreadSafety:
    """InMemoryShortTermStorage 线程安全测试"""

    def test_concurrent_put_get(self):
        """验证并发 put/get 操作不会导致数据竞争"""
        storage = InMemoryShortTermStorage()
        errors = []

        def put_task(topic_id):
            try:
                for i in range(100):
                    buf = _make_buffer(f"{topic_id}_{i}", f"user_{topic_id}")
                    storage._put_sync(f"{topic_id}_{i}", buf)
            except Exception as e:
                errors.append(e)

        def get_task(topic_id):
            try:
                for i in range(100):
                    storage.get(f"{topic_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t1 = threading.Thread(target=put_task, args=(i,))
            t2 = threading.Thread(target=get_task, args=(i,))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        assert storage._count() == 500
