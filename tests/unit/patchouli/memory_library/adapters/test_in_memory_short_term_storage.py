"""
InMemoryShortTermStorage 单元测试

测试覆盖:
- get/put/pop: 基本 CRUD 操作
- list_by_user/list_all: 查询操作
- 线程安全性测试
"""

import threading

from hivememory.core.models import LogicalBlock, TurnRecord, WorkspaceTopicKey
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from tests.helpers.workspace import make_identity_scope


def _make_buffer(topic_id="t1", user_id="u1") -> SemanticBuffer:
    return SemanticBuffer(
        topic_id=topic_id,
        workspace_identity=_context(user_id).workspace_identity,
        topic_title=f"话题{topic_id}",
        blocks=[LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))],
    )


def _context(user_id: str):
    return make_identity_scope(user_id=user_id)


def _key(topic_id: str, user_id: str) -> WorkspaceTopicKey:
    return WorkspaceTopicKey.from_identity_scope(_context(user_id), topic_id)


class TestInMemoryShortTermStorageBasic:
    """InMemoryShortTermStorage 基本操作测试"""

    def setup_method(self):
        self.storage = InMemoryShortTermStorage()

    def test_get_returns_none_for_missing(self):
        result = self.storage.get(_key("missing", "u1"))
        assert result is None

    def test_put_and_get(self):
        buf = _make_buffer("t1", "u1")
        self.storage.put(_key("t1", "u1"), buf)

        result = self.storage.get(_key("t1", "u1"))
        assert result is buf

    def test_put_overwrites_existing(self):
        buf1 = _make_buffer("t1", "u1")
        buf2 = _make_buffer("t1", "u2")

        self.storage.put(_key("t1", "u1"), buf1)
        self.storage.put(_key("t1", "u2"), buf2)

        assert self.storage.get(_key("t1", "u1")) is buf1
        assert self.storage.get(_key("t1", "u2")) is buf2

    def test_pop_returns_and_removes(self):
        buf = _make_buffer("t1", "u1")
        self.storage.put(_key("t1", "u1"), buf)

        result = self.storage.pop(_key("t1", "u1"))

        assert result is buf
        assert self.storage.get(_key("t1", "u1")) is None

    def test_pop_returns_none_for_missing(self):
        result = self.storage.pop(_key("missing", "u1"))
        assert result is None


class TestInMemoryShortTermStorageList:
    """InMemoryShortTermStorage 列表查询测试"""

    def setup_method(self):
        self.storage = InMemoryShortTermStorage()

    def test_list_by_user_returns_user_buffers(self):
        self.storage.put(_key("t1", "u1"), _make_buffer("t1", "u1"))
        self.storage.put(_key("t2", "u1"), _make_buffer("t2", "u1"))
        self.storage.put(_key("t3", "u2"), _make_buffer("t3", "u2"))

        result = self.storage.list_by_workspace(_context("u1").workspace_identity)

        assert len(result) == 2
        topic_ids = {b.topic_id for b in result}
        assert topic_ids == {"t1", "t2"}

    def test_list_by_user_returns_empty_for_no_such_user(self):
        result = self.storage.list_by_workspace(_context("missing").workspace_identity)
        assert result == []

    def test_list_all_returns_all_buffers(self):
        self.storage.put(_key("t1", "u1"), _make_buffer("t1", "u1"))
        self.storage.put(_key("t2", "u2"), _make_buffer("t2", "u2"))

        result = self.storage.list_all()

        assert len(result) == 2

    def test_list_all_returns_empty_when_empty(self):
        result = self.storage.list_all()
        assert result == []

    def test_count_returns_buffer_count(self):
        workspace = _context("u1").workspace_identity
        assert self.storage.count(workspace) == 0
        self.storage.put(_key("t1", "u1"), _make_buffer("t1", "u1"))
        assert self.storage.count(workspace) == 1


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
                    storage.put(_key(f"{topic_id}_{i}", f"user_{topic_id}"), buf)
            except Exception as e:
                errors.append(e)

        def get_task(topic_id):
            try:
                for i in range(100):
                    storage.get(_key(f"{topic_id}_{i}", f"user_{topic_id}"))
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
        assert sum(
            storage.count(_context(f"user_{topic_id}").workspace_identity)
            for topic_id in range(5)
        ) == 500
