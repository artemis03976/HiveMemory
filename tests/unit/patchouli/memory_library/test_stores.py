"""MidTermMemoryStore 门面参数透传的单元测试。

回归保护：``RetrievalFamiliar.get_memory`` 向 ``mid_term.get`` 传递
``enforce_actor_visibility`` 关键字参数；store 门面曾不透传该参数导致
GET 读取路径必然 TypeError（WRX-1 修复）。本文件同时锁定 get_by_alias
与 search/scroll 的同构透传行为（list_memories 曾因门面 scroll/search
缺参导致 /api/v1/memories 与 /api/v1/agents 全量 500）。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from hivememory.patchouli.memory_library.ports import MidTermStoragePort
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.workspace import make_identity_scope


class _RecordingPort(MidTermStoragePort):
    """记录 get/get_by_alias/search/scroll 收到的关键字参数；其余方法不应被触达。"""

    def __init__(self):
        self.enforce_seen: list[bool] = []

    async def get(self, identity_scope, memory_id, *, enforce_actor_visibility=True):
        self.enforce_seen.append(enforce_actor_visibility)
        return None

    async def get_by_alias(self, identity_scope, alias, *, enforce_actor_visibility=True):
        self.enforce_seen.append(enforce_actor_visibility)
        return None

    async def get_for_mutation(self, identity_scope, memory_id):  # pragma: no cover
        raise AssertionError("not expected")

    async def get_by_key(self, key):  # pragma: no cover
        raise AssertionError("not expected")

    async def update_access_info(self, identity_scope, memory_id):  # pragma: no cover
        raise AssertionError("not expected")

    async def upsert(self, memory):  # pragma: no cover
        raise AssertionError("not expected")

    async def delete(self, identity_scope, memory_id):  # pragma: no cover
        raise AssertionError("not expected")

    async def delete_by_key(self, key):  # pragma: no cover
        raise AssertionError("not expected")

    async def search(
        self,
        identity_scope,
        query,
        top_k,
        filters=None,
        mode="dense",
        score_threshold=0.0,
        *,
        enforce_actor_visibility=True,
    ):
        self.enforce_seen.append(enforce_actor_visibility)
        return []

    async def scroll(
        self,
        identity_scope,
        filters=None,
        limit=100,
        *,
        enforce_actor_visibility=True,
    ):
        self.enforce_seen.append(enforce_actor_visibility)
        return []

    async def batch_delete(self, identity_scope, memory_ids):  # pragma: no cover
        raise AssertionError("not expected")

    async def count(self, identity_scope):  # pragma: no cover
        raise AssertionError("not expected")

    async def list_all_for_maintenance(self):  # pragma: no cover
        raise AssertionError("not expected")


def test_store_forwards_enforce_flag_for_point_read():
    """门面必须把 enforce_actor_visibility 透传给底层 port。"""
    port = _RecordingPort()
    store = MidTermMemoryStore(port)
    scope = make_identity_scope()

    asyncio.run(store.get(scope, uuid4(), enforce_actor_visibility=False))
    asyncio.run(store.get_by_alias(scope, "alias", enforce_actor_visibility=False))

    assert port.enforce_seen == [False, False]


def test_store_forwards_enforce_flag_for_search_and_scroll():
    """门面 search/scroll 同样必须透传 enforce_actor_visibility。"""
    port = _RecordingPort()
    store = MidTermMemoryStore(port)
    scope = make_identity_scope()

    asyncio.run(store.search(scope, "query", top_k=5, enforce_actor_visibility=False))
    asyncio.run(store.scroll(scope, limit=10, enforce_actor_visibility=False))

    assert port.enforce_seen == [False, False]


def test_retrieval_familiar_get_memory_does_not_raise_type_error():
    """回归：familiar.get_memory 携带 enforce 参数经门面调用不再 TypeError。"""
    port = _RecordingPort()
    familiar = RetrievalFamiliar(
        engine=object(), memory_library=type("_Lib", (), {"mid_term": MidTermMemoryStore(port)})()
    )

    result = asyncio.run(familiar.get_memory(uuid4(), identity_scope=make_identity_scope()))

    assert result is None
    assert port.enforce_seen == [True]


def test_retrieval_familiar_list_memories_scroll_does_not_raise_type_error():
    """回归：familiar.list_memories 无 query 走 scroll，经门面不再 TypeError。"""
    port = _RecordingPort()
    familiar = RetrievalFamiliar(
        engine=object(), memory_library=type("_Lib", (), {"mid_term": MidTermMemoryStore(port)})()
    )

    result = asyncio.run(
        familiar.list_memories(identity_scope=make_identity_scope(), enforce_actor_visibility=False)
    )

    assert result == []
    assert port.enforce_seen == [False]
