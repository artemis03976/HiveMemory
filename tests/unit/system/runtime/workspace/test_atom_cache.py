"""
KoakumaAtomCache 单元测试
"""

from uuid import uuid4

import pytest

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.system.runtime.workspace import KoakumaAtomCache
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity()
ISOLATED = make_workspace_identity(workspace_id="isolation_workspace")


@pytest.fixture
def sample_atom():
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test"),
        index=IndexLayer(
            title="Test Memory",
            summary="Test summary",
            memory_type=MemoryType.FACT,
            alias="fact_test_memory",
        ),
        payload=PayloadLayer(content="Test content"),
    )


def test_ingest_and_retrieve_by_alias(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom, workspace_identity=MAIN)

    retrieved = cache.get_atom_by_alias("fact_test_memory", workspace_identity=MAIN)
    assert retrieved.id == sample_atom.id
    assert retrieved.payload.content == "Test content"


def test_ingest_and_retrieve_by_uuid(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom, workspace_identity=MAIN)

    retrieved = cache.get_atom_by_uuid(str(sample_atom.id))
    assert retrieved is not None
    assert retrieved.payload.content == "Test content"


def test_ingest_multiple_atoms():
    cache = KoakumaAtomCache()
    atoms = [
        MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(user_id="test", source_agent_id="test"),
            index=IndexLayer(
                title=f"Memory {i}",
                summary=f"Test summary for memory {i}",
                memory_type=MemoryType.FACT,
                alias=f"fact_memory_{i}",
            ),
            payload=PayloadLayer(content=f"Content {i}"),
        )
        for i in range(3)
    ]

    cache.ingest_atoms(atoms, workspace_identity=MAIN)
    assert cache.size == 3

    for i in range(3):
        atom = cache.get_atom_by_alias(f"fact_memory_{i}", workspace_identity=MAIN)
        assert atom is not None
        assert atom.payload.content == f"Content {i}"


def test_cache_miss():
    cache = KoakumaAtomCache()
    assert cache.get_atom_by_alias("nonexistent", workspace_identity=MAIN) is None
    assert cache.get_atom_by_uuid("00000000-0000-0000-0000-000000000000") is None


def test_invalidate_alias(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom, workspace_identity=MAIN)

    cache.invalidate_alias("fact_test_memory", workspace_identity=MAIN)

    assert not cache.has_alias("fact_test_memory", workspace_identity=MAIN)
    assert cache.get_atom_by_alias("fact_test_memory", workspace_identity=MAIN) is None
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is None


def test_alias_hit_and_miss_counters():
    """alias 读取路径按命中/未命中累计统计，供运行时可观测。"""
    cache = KoakumaAtomCache()
    atom = MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="test", source_agent_id="test"),
        index=IndexLayer(
            title="Counted Memory",
            summary="Counted summary",
            memory_type=MemoryType.FACT,
            alias="fact_counted",
        ),
        payload=PayloadLayer(content="counted"),
    )
    cache.ingest_atom(atom, workspace_identity=MAIN)

    assert cache.get_atom_by_alias("fact_counted", workspace_identity=MAIN) is atom
    assert cache.get_atom_by_alias("fact_missing", workspace_identity=MAIN) is None

    assert cache.alias_hits == 1
    assert cache.alias_misses == 1


def test_cache_read_rejects_missing_workspace_identity(sample_atom):
    """无 Workspace 坐标的 cache 读写不存在，缺失或错误坐标直接拒绝。"""
    cache = KoakumaAtomCache()

    with pytest.raises(TypeError):
        cache.ingest_atom(sample_atom, workspace_identity=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cache.get_atom_by_alias("fact_test_memory", workspace_identity="main")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cache.invalidate_alias("fact_test_memory", workspace_identity=None)  # type: ignore[arg-type]

    assert cache.get_atom_by_alias("fact_test_memory", workspace_identity=MAIN) is None


def test_same_alias_resolves_per_workspace(sample_atom):
    """同 alias 在不同 Workspace 分区各自命中各自的 atom，互不串扰。"""
    replacement = sample_atom.model_copy(deep=True)
    replacement.id = uuid4()
    replacement.meta.workspace_identity = replacement.meta.workspace_identity.model_copy(
        update={
            "workspace_key": "isolation_workspace",
            "workspace_id": "isolation_workspace",
        }
    )
    cache = KoakumaAtomCache()

    cache.ingest_atom(sample_atom, workspace_identity=MAIN)
    cache.ingest_atom(replacement, workspace_identity=ISOLATED)

    assert cache.get_atom_by_alias("fact_test_memory", workspace_identity=MAIN) is sample_atom
    assert (
        cache.get_atom_by_alias("fact_test_memory", workspace_identity=ISOLATED)
        is replacement
    )
    # UUID 索引保持全局：两个 atom 都能按 UUID 命中。
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is sample_atom
    assert cache.get_atom_by_uuid(str(replacement.id)) is replacement
    # 未写入的分区不产生命中。
    other = make_workspace_identity(workspace_id="third_workspace")
    assert cache.has_alias("fact_test_memory", workspace_identity=other) is False


def test_same_workspace_alias_replacement_keeps_old_uuid_entry(sample_atom):
    """同分区内同 alias 重复写入按替换语义覆盖，UUID 全局条目仍可反查。"""
    replacement = sample_atom.model_copy(deep=True)
    replacement.id = uuid4()
    cache = KoakumaAtomCache()

    cache.ingest_atom(sample_atom, workspace_identity=MAIN)
    cache.ingest_atom(replacement, workspace_identity=MAIN)

    assert cache.get_atom_by_alias("fact_test_memory", workspace_identity=MAIN) is replacement
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is sample_atom
    assert cache.get_atom_by_uuid(str(replacement.id)) is replacement


def test_invalidate_keeps_shared_uuid_entry_for_other_workspace(sample_atom):
    """同 UUID 跨 Workspace 共享时，单分区 invalidate 不得留下悬空 alias 索引。"""
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom, workspace_identity=MAIN)
    cache.ingest_atom(sample_atom, workspace_identity=ISOLATED)

    cache.invalidate_alias("fact_test_memory", workspace_identity=MAIN)

    # main 分区条目已失效；UUID 条目仍被 isolation 分区引用，必须保留。
    assert cache.has_alias("fact_test_memory", workspace_identity=MAIN) is False
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is sample_atom
    assert (
        cache.get_atom_by_alias("fact_test_memory", workspace_identity=ISOLATED)
        is sample_atom
    )


def test_clear():
    cache = KoakumaAtomCache()
    atoms = [
        MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(user_id="test", source_agent_id="test"),
            index=IndexLayer(
                title=f"Memory {i}",
                summary=f"Test summary for memory {i}",
                memory_type=MemoryType.FACT,
                alias=f"fact_memory_{i}",
            ),
            payload=PayloadLayer(content=f"Content {i}"),
        )
        for i in range(3)
    ]
    cache.ingest_atoms(atoms, workspace_identity=MAIN)
    cache.ingest_atoms(atoms, workspace_identity=ISOLATED)
    assert cache.size == 3

    cache.clear()
    assert cache.size == 0
    assert cache.get_atom_by_alias("fact_memory_0", workspace_identity=MAIN) is None
    assert cache.get_atom_by_alias("fact_memory_0", workspace_identity=ISOLATED) is None


def test_alias_maps_to_cached_atom(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom, workspace_identity=MAIN)
    atom = cache.get_atom_by_alias("fact_test_memory", workspace_identity=MAIN)
    assert atom is not None
    assert cache.get_atom_by_uuid(str(atom.id)) is atom
