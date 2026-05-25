import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from hivememory.alice.runtime.cache import KoakumaAtomCache, PendingAtomCache
from hivememory.alice.runtime.models import MTPExecutionContext
from hivememory.alice.runtime.resolver import RuntimeAliasResolver
from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import BusRouteUnavailableError, StorageReadError

from tests.unit.patchouli.mtp.conftest import make_mock_bus


def _make_memory(alias: str, content: str = "content") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="Test Memory",
            summary="A resolver test memory",
            tags=["test"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


@pytest.fixture
def resolver_parts():
    bus = make_mock_bus()
    pending_cache = PendingAtomCache()
    atom_cache = KoakumaAtomCache()
    resolver = RuntimeAliasResolver(
        pending_cache=pending_cache,
        atom_cache=atom_cache,
        bus=bus,
    )
    return resolver, pending_cache, atom_cache, bus


@pytest.mark.asyncio
async def test_resolve_l0_pending_hit(resolver_parts):
    resolver, pending_cache, _atom_cache, bus = resolver_parts
    pending = pending_cache.register_write(
        content="pending content",
        title="Pending Note",
        reason=None,
        identity=Identity(user_id="test_user"),
    )

    result = await resolver.resolve(pending.pending_alias)

    assert result.kind == "pending"
    assert result.pending is pending
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_l1_atom_hit(resolver_parts):
    resolver, _pending_cache, atom_cache, bus = resolver_parts
    atom = _make_memory(alias="fact_l1", content="from l1")
    atom_cache.ingest_atom(atom)

    result = await resolver.resolve("fact_l1")

    assert result.kind == "atom"
    assert result.atom is atom
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_l2_hit_promotes_to_l1(resolver_parts):
    resolver, _pending_cache, atom_cache, bus = resolver_parts
    atom = _make_memory(alias="fact_l2", content="from l2")
    bus._mock_storage.get_memory_by_alias.return_value = atom

    context = MTPExecutionContext(identity=Identity(user_id="test_user"))
    result = await resolver.resolve("fact_l2", context=context)

    assert result.kind == "atom"
    assert result.atom is atom
    assert atom_cache.get_atom_by_alias("fact_l2") is atom

    bus._mock_storage.get_memory_by_alias.reset_mock()
    second = await resolver.resolve("fact_l2", context=context)
    assert second.kind == "atom"
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_l2_miss(resolver_parts):
    resolver, _pending_cache, _atom_cache, bus = resolver_parts
    bus._mock_storage.get_memory_by_alias.return_value = None

    result = await resolver.resolve("missing")

    assert result.kind == "not_found"


@pytest.mark.asyncio
async def test_resolve_route_failure_raises_bus_unavailable(resolver_parts):
    resolver, _pending_cache, _atom_cache, bus = resolver_parts
    bus._mock_storage.get_memory_by_alias.side_effect = KeyError("route missing")

    with pytest.raises(BusRouteUnavailableError):
        await resolver.resolve("fact_route_missing")


@pytest.mark.asyncio
async def test_resolve_storage_failure_raises_storage_read_error(resolver_parts):
    resolver, _pending_cache, _atom_cache, bus = resolver_parts
    bus._mock_storage.get_memory_by_alias.side_effect = RuntimeError("boom")

    with pytest.raises(StorageReadError):
        await resolver.resolve("fact_boom")
