"""
KoakumaAtomCache 单元测试
"""

import pytest
from uuid import uuid4
from hivememory.patchouli.kernel.runtime.cache import KoakumaAtomCache
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType


@pytest.fixture
def sample_atom():
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test"),
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
    assert retrieved is not None
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
            meta=MetaData(user_id="test", source_agent_id="test"),
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

    assert cache.has_alias("fact_test_memory")
    cache.invalidate_alias("fact_test_memory")

    assert not cache.has_alias("fact_test_memory")
    assert cache.get_atom_by_alias("fact_test_memory") is None
    assert cache.get_atom_by_uuid(str(sample_atom.id)) is None


def test_clear():
    cache = KoakumaAtomCache()
    atoms = [
        MemoryAtom(
            id=uuid4(),
            meta=MetaData(user_id="test", source_agent_id="test"),
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
