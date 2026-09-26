"""AtomCache 的单元测试（A2 §3 完整原子缓存）。

被测对象：Workspace 分区寻址、alias 正反索引与 LRU 淘汰的一致性，以及
copy-on-read 的引用隔离。缓存不做授权与回源，相关行为由 resolver 测试覆盖。
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.workspace.cache.atom import AtomCache
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="other_workspace")


def _atom(
    alias: str | None,
    *,
    content: str = "content",
    workspace_id: str = "main_workspace",
    memory_id: UUID | None = None,
) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1", workspace_id=workspace_id),
        index=IndexLayer(
            title="Cached memory",
            summary="Memory used to verify cache indexing.",
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


def test_alias_rebound_to_another_atom_resolves_to_new_holder():
    """alias 改指另一原子后按新占用者解析；驱逐旧原子不删除新的 alias 映射。

    捕获反向索引残留导致驱逐旧原子时误删新占用者 alias 的缺陷。
    """
    cache = AtomCache(capacity=8)
    old = _atom("fact_shared", content="old")
    new = _atom("fact_shared", content="new")
    cache.put(old)
    cache.put(new)

    cache.evict(MAIN, old.id)

    assert cache.get_by_alias(MAIN, "fact_shared").id == new.id


def test_renamed_atom_releases_its_old_alias():
    """同一原子以新 alias 重新写入后，旧 alias 不再返回该原子（被释放的旧 alias）。"""
    cache = AtomCache(capacity=8)
    atom = _atom("fact_old")
    cache.put(atom)
    renamed = atom.model_copy(deep=True)
    renamed.index.alias = "fact_new"

    cache.put(renamed)

    assert cache.get_by_alias(MAIN, "fact_old") is None
    assert cache.get_by_alias(MAIN, "fact_new").id == atom.id


def test_evict_removes_atom_and_its_alias_index():
    """驱逐原子时连同其 alias 一并失效，alias 与 UUID 均不再命中。"""
    cache = AtomCache(capacity=8)
    atom = _atom("fact_gone")
    cache.put(atom)

    assert cache.evict(MAIN, atom.id) is True
    assert (cache.get_by_id(MAIN, atom.id), cache.get_by_alias(MAIN, "fact_gone")) == (None, None)


def test_lru_eviction_drops_least_recent_atom_and_alias():
    """超出容量时淘汰最久未访问的原子及其 alias；最近访问的条目保留。"""
    cache = AtomCache(capacity=2)
    first = _atom("fact_first")
    second = _atom("fact_second")
    cache.put(first)
    cache.put(second)
    cache.get_by_id(MAIN, first.id)  # first 变为最近访问

    cache.put(_atom("fact_third"))

    assert cache.get_by_alias(MAIN, "fact_second") is None
    assert cache.get_by_alias(MAIN, "fact_first").id == first.id
    assert cache.evictions == 1


def test_cache_isolates_stored_and_returned_objects():
    """写入后修改源对象、读取后修改返回值，都不改变缓存内容。"""
    cache = AtomCache(capacity=8)
    atom = _atom("fact_isolated", content="original")
    cache.put(atom)
    atom.payload.content = "mutated source"

    first = cache.get_by_id(MAIN, atom.id)
    first.payload.content = "mutated copy"

    assert cache.get_by_id(MAIN, atom.id).payload.content == "original"


def test_same_uuid_and_alias_are_partitioned_by_workspace():
    """两个 Workspace 复用同一 UUID 与 alias 时各自独立命中，互不覆盖。"""
    cache = AtomCache(capacity=8)
    shared_id = uuid4()
    cache.put(_atom("fact_same", content="main", memory_id=shared_id))
    cache.put(_atom("fact_same", content="other", memory_id=shared_id, workspace_id="other_workspace"))

    assert cache.get_by_id(MAIN, shared_id).payload.content == "main"
    assert cache.get_by_alias(OTHER, "fact_same").payload.content == "other"


def test_capacity_must_be_positive():
    """容量配置非正时显式失败，不静默成为无上限缓存。"""
    with pytest.raises(ValueError, match="capacity"):
        AtomCache(capacity=0)
