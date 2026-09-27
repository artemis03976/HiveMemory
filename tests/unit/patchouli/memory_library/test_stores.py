"""MidTermMemoryStore 门面参数透传与写前 alias 唯一性校验的单元测试。

回归保护：``RetrievalFamiliar.get_memory`` 向 ``mid_term.get`` 传递
``enforce_actor_visibility`` 关键字参数；store 门面曾不透传该参数导致
GET 读取路径必然 TypeError（WRX-1 修复）。本文件同时锁定 get_by_alias
与 search/scroll 的同构透传行为（list_memories 曾因门面 scroll/search
缺参导致 /api/v1/memories 与 /api/v1/agents 全量 500），以及 ``upsert``
在主后端写入前的 alias 唯一性兜底（A2 §8 D-4 第二层）。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.patchouli.memory_library.ports import MidTermStoragePort
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.utils.time import utc_now
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


class _RecordingPort(MidTermStoragePort):
    """记录 get/get_by_alias/search/scroll/patch_payload 收到的参数；其余方法不应被触达。"""

    def __init__(self, *, name: str = "port", call_order: list[str] | None = None):
        self.enforce_seen: list[bool] = []
        self.name = name
        self.call_order = call_order if call_order is not None else []
        self.patch_calls: list[tuple[WorkspaceMemoryKey, dict]] = []
        self.patch_result: object | None = None
        # alias -> 当前占用者 memory_id 列表；缺省视为空闲。
        self.alias_holders: dict[str, list] = {}

    async def list_alias_holders(self, workspace_identity, alias, *, limit):
        return list(self.alias_holders.get(alias, []))[:limit]

    async def get(self, identity_scope, memory_id, *, enforce_actor_visibility=True):
        self.enforce_seen.append(enforce_actor_visibility)
        return None

    async def get_by_alias(self, identity_scope, alias, *, enforce_actor_visibility=True):
        self.enforce_seen.append(enforce_actor_visibility)
        return None

    async def get_by_key(self, key):  # pragma: no cover
        raise AssertionError("not expected")

    async def patch_payload(self, key, patch):
        self.patch_calls.append((key, dict(patch)))
        self.call_order.append(f"{self.name}:patch_payload")
        return self.patch_result

    async def upsert(self, memory, *, recompute_vectors=True):  # pragma: no cover
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

    async def list_all_for_maintenance(self):  # pragma: no cover
        raise AssertionError("not expected")


class _MutatingPort(_RecordingPort):
    """在记录端口之上支持成功的 upsert，或按 fail_with 注入可变操作失败。"""

    def __init__(self, *, fail_with: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self.fail_with = fail_with
        self.upsert_calls: list[MemoryAtom] = []

    async def upsert(self, memory, *, recompute_vectors=True):
        if self.fail_with is not None:
            raise self.fail_with
        self.upsert_calls.append(memory)
        self.call_order.append(f"{self.name}:upsert")

    async def patch_payload(self, key, patch):
        if self.fail_with is not None:
            raise self.fail_with
        return await super().patch_payload(key, patch)


def _memory_atom(alias: str | None = None) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="agent1", user_id="user1"),
        index=IndexLayer(
            title="Secondary failure probe",
            summary="Memory used to assert secondary failure propagation",
            tags=["test"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content="content"),
    )


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


def test_store_forwards_patch_payload_to_primary_and_secondary():
    """MVL-2: patch_payload 必须把同一 patch 沿顺序转发 primary 与全部 secondary。"""
    call_order: list[str] = []
    primary = _RecordingPort(name="primary", call_order=call_order)
    secondary = _RecordingPort(name="secondary", call_order=call_order)
    store = MidTermMemoryStore(primary, secondary=[secondary])
    scope = make_identity_scope()
    memory_id = uuid4()
    key = WorkspaceMemoryKey.from_identity_scope(scope, memory_id)
    patch = {
        "meta.lifecycle.access_count": 4,
        "meta.lifecycle.last_accessed_at": utc_now(),
    }

    result = asyncio.run(store.patch_payload(key, patch))

    assert primary.patch_calls == [(key, patch)]
    assert secondary.patch_calls == [(key, patch)]
    assert call_order == ["primary:patch_payload", "secondary:patch_payload"]
    assert result is primary.patch_result, "门面应返回 primary 的 patch_payload 结果"


def test_store_propagates_secondary_failure_for_upsert_and_patch():
    """MVL-4：primary 成功后 secondary 失败时，门面必须让错误按序直接传播（不吞）。"""
    primary = _MutatingPort(name="primary")
    secondary = _MutatingPort(name="secondary", fail_with=RuntimeError("secondary down"))
    store = MidTermMemoryStore(primary, secondary=[secondary])
    memory = _memory_atom()
    scope = make_identity_scope()
    key = WorkspaceMemoryKey.from_identity_scope(scope, memory.id)
    patch = {"meta.lifecycle.access_count": 1}

    with pytest.raises(RuntimeError, match="secondary down"):
        asyncio.run(store.upsert(memory))
    # primary 已成功写入，secondary 失败不被吞掉、也不回滚 primary。
    assert primary.upsert_calls == [memory]
    assert secondary.upsert_calls == []

    with pytest.raises(RuntimeError, match="secondary down"):
        asyncio.run(store.patch_payload(key, patch))
    assert primary.patch_calls == [(key, patch)]
    assert secondary.patch_calls == []
    # 两次调用都是 primary 先执行、错误由 secondary 抛出。
    assert primary.call_order == ["primary:upsert", "primary:patch_payload"]


def test_store_upsert_rejects_alias_held_by_another_memory_without_writing():
    """alias 已被同 Workspace 其他记忆占用时拒绝提交，primary 与 secondary 都不写入。

    捕获写前校验缺失、或校验放在主后端写入之后导致重名已落库的缺陷。
    """
    primary = _MutatingPort(name="primary")
    secondary = _MutatingPort(name="secondary")
    store = MidTermMemoryStore(primary, secondary=[secondary])
    memory = _memory_atom(alias="fact_shared")
    holder_id = uuid4()
    primary.alias_holders["fact_shared"] = [holder_id]

    with pytest.raises(MemoryAliasConflictError) as exc_info:
        asyncio.run(store.upsert(memory))

    assert exc_info.value.details["reason"] == "alias_occupied"
    assert exc_info.value.details["conflicting_memory_id"] == str(holder_id)
    assert primary.upsert_calls == []
    assert secondary.upsert_calls == []


def test_store_upsert_allows_memory_that_already_holds_its_alias():
    """保留原 alias 的内容更新：自身占用不算冲突，提交正常写入。

    捕获把自身也计为占用者、导致任何带 alias 的记忆都无法再更新的缺陷。
    """
    primary = _MutatingPort(name="primary")
    store = MidTermMemoryStore(primary)
    memory = _memory_atom(alias="fact_owned")
    primary.alias_holders["fact_owned"] = [memory.id]

    asyncio.run(store.upsert(memory))

    assert primary.upsert_calls == [memory]


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
