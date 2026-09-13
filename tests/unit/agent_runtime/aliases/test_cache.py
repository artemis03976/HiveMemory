"""
KoakumaAtomCache 单元测试
"""

from uuid import uuid4

import pytest

from hivememory.agent_runtime.aliases.cache import KoakumaAtomCache
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from tests.helpers.memory import make_memory_metadata


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
    cache.ingest_atom(sample_atom)

    retrieved = cache.get_atom_by_alias("fact_test_memory")
    assert retrieved.id == sample_atom.id
    assert retrieved.payload.content == "Test content"


def test_ingest_and_retrieve_by_uuid(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom)

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

    cache.ingest_atoms(atoms)
    assert cache.size == 3

    for i in range(3):
        atom = cache.get_atom_by_alias(f"fact_memory_{i}")
        assert atom is not None
        assert atom.payload.content == f"Content {i}"


def test_cache_miss():
    cache = KoakumaAtomCache()
    assert cache.get_atom_by_alias("nonexistent") is None
    assert cache.get_atom_by_uuid("00000000-0000-0000-0000-000000000000") is None


def test_invalidate_alias(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom)

    cache.invalidate_alias("fact_test_memory")

    assert not cache.has_alias("fact_test_memory")
    assert cache.get_atom_by_alias("fact_test_memory") is None
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is None


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
    cache.ingest_atoms(atoms)
    assert cache.size == 3

    cache.clear()
    assert cache.size == 0
    assert cache.get_atom_by_alias("fact_memory_0") is None


def test_alias_maps_to_cached_atom(sample_atom):
    cache = KoakumaAtomCache()
    cache.ingest_atom(sample_atom)
    atom = cache.get_atom_by_alias("fact_test_memory")
    assert atom is not None
    assert cache.get_atom_by_uuid(str(atom.id)) is atom


def test_same_alias_keeps_existing_global_cache_semantics(sample_atom):
    """固定迁移前基线：alias 索引全局共享，同 alias 后写入者覆盖先写入者。

    WRT-0 标记待改行为：v0.6.2 cache 迁移（WRT-2）后 alias 索引改为
    ``(WorkspaceIdentity, alias) -> UUID``，同 alias 在不同 Workspace 各自
    命中，本测试的全局覆盖断言将被跨 Workspace 不串扰断言替换。迁移完成
    前勿把本测试当作稳定契约。
    参考：docs/plans/v0.6.2-workspace-runtime-cache-migration.md §5.2 / §7
    """
    replacement = sample_atom.model_copy(deep=True)
    replacement.id = uuid4()
    replacement.meta.workspace_identity = replacement.meta.workspace_identity.model_copy(
        update={
            "workspace_key": "isolation_workspace",
            "workspace_id": "isolation_workspace",
        }
    )
    cache = KoakumaAtomCache()

    cache.ingest_atom(sample_atom)
    cache.ingest_atom(replacement)

    assert cache.get_atom_by_alias("fact_test_memory") is replacement
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is sample_atom
