from uuid import uuid4

import pytest

from hivememory.agent_runtime.aliases.cache import KoakumaAtomCache
from hivememory.agent_runtime.aliases.resolver import RuntimeAliasResolver
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from hivememory.core.mtp.exceptions import BusRouteUnavailableError, StorageReadError
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_runtime_scope
from tests.unit.agent_runtime.mtp.conftest import make_mock_bus


def _context(*, workspace_id: str = "main_workspace") -> MTPExecutionContext:
    return MTPExecutionContext(
        runtime_scope=make_runtime_scope(workspace_id=workspace_id)
    )


def _make_memory(
    alias: str,
    content: str = "content",
    *,
    workspace_id: str = "main_workspace",
) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(
            user_id="test_user",
            source_agent_id="test_agent",
            workspace_id=workspace_id,
        ),
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
    pending_runtime = PendingAtomRuntime()
    atom_cache = KoakumaAtomCache()
    resolver = RuntimeAliasResolver(
        pending_runtime=pending_runtime,
        atom_cache=atom_cache,
        bus=bus,
    )
    return resolver, pending_runtime, atom_cache, bus


@pytest.mark.asyncio
async def test_resolve_l0_pending_hit(resolver_parts):
    resolver, pending_runtime, _atom_cache, bus = resolver_parts
    pending = pending_runtime.register_write(
        content="pending content",
        title="Pending Note",
        reason=None,
        identity=Identity(user_id="test_user"),
        runtime_scope=make_runtime_scope(),
    )

    result = await resolver.resolve(pending.pending_alias, context=_context())

    assert result.kind == "pending"
    assert result.pending is pending
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_settled_pending_redirect_l1_hit(resolver_parts):
    resolver, pending_runtime, atom_cache, bus = resolver_parts
    pending = pending_runtime.register_write(
        content="pending content",
        title="Pending Note",
        reason=None,
        identity=Identity(user_id="test_user"),
        runtime_scope=make_runtime_scope(),
    )
    canonical = _make_memory(alias="fact_canonical", content="canonical content")
    atom_cache.ingest_atom(canonical)
    settlement = PendingAtomSettlement(
        pending_alias=pending.pending_alias,
        intent_id=pending.intent_id,
        resolution=PendingAtomResolution.CREATED,
        canonical_alias="fact_canonical",
        canonical_uuid=str(canonical.id),
    )

    pending_runtime.claim_for_materialization([pending.pending_alias])
    pending_runtime.settle(settlement)
    result = await resolver.resolve(pending.pending_alias, context=_context())

    assert result.kind == "redirect"
    assert result.requested_alias == pending.pending_alias
    assert result.pending is pending
    assert result.settlement is settlement
    assert result.canonical_alias == "fact_canonical"
    assert result.canonical_uuid == str(canonical.id)
    assert result.atom is canonical
    assert pending_runtime.get_redirect(pending.pending_alias) is settlement
    assert pending.pending_alias in pending_runtime.get_pending_aliases_for_canonical_uuid(
        str(canonical.id)
    )
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_settled_pending_redirect_l2_hit(resolver_parts):
    resolver, pending_runtime, atom_cache, bus = resolver_parts
    pending = pending_runtime.register_write(
        content="pending content",
        title="Pending Note",
        reason=None,
        identity=Identity(user_id="test_user"),
        runtime_scope=make_runtime_scope(),
    )
    canonical = _make_memory(alias="fact_canonical", content="from storage")
    bus._mock_storage.get_memory_by_alias.return_value = canonical
    settlement = PendingAtomSettlement(
        pending_alias=pending.pending_alias,
        intent_id=pending.intent_id,
        resolution=PendingAtomResolution.MERGED,
        canonical_alias="fact_canonical",
        canonical_uuid=str(canonical.id),
    )

    pending_runtime.claim_for_materialization([pending.pending_alias])
    pending_runtime.settle(settlement)
    result = await resolver.resolve(pending.pending_alias, context=_context())

    assert result.kind == "redirect"
    assert result.atom is canonical
    assert atom_cache.get_atom_by_alias("fact_canonical") is canonical


@pytest.mark.asyncio
async def test_resolve_discarded_pending_without_redirect(resolver_parts):
    resolver, pending_runtime, _atom_cache, bus = resolver_parts
    pending = pending_runtime.register_write(
        content="pending content",
        title="Pending Note",
        reason=None,
        identity=Identity(user_id="test_user"),
        runtime_scope=make_runtime_scope(),
    )
    settlement = PendingAtomSettlement(
        pending_alias=pending.pending_alias,
        intent_id=pending.intent_id,
        resolution=PendingAtomResolution.DISCARDED,
        message="Not materialized.",
    )

    pending_runtime.claim_for_materialization([pending.pending_alias])
    pending_runtime.settle(settlement)
    result = await resolver.resolve(pending.pending_alias, context=_context())

    assert result.kind == "discarded"
    assert result.pending is pending
    assert result.settlement is settlement
    assert result.atom is None
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_l1_atom_hit(resolver_parts):
    resolver, _pending_runtime, atom_cache, bus = resolver_parts
    atom = _make_memory(alias="fact_l1", content="from l1")
    atom_cache.ingest_atom(atom)

    result = await resolver.resolve("fact_l1", context=_context())

    assert result.kind == "atom"
    assert result.atom is atom
    bus._mock_storage.get_memory_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_l1_hit_revalidates_workspace_without_partitioning_cache(resolver_parts):
    """捕获共享 alias cache 命中被误作 Workspace 授权结果的缺陷。"""
    resolver, _pending_runtime, atom_cache, bus = resolver_parts
    cached = _make_memory(
        alias="fact_shared",
        content="isolated",
        workspace_id="isolation_workspace",
    )
    authorized = _make_memory(
        alias="fact_shared",
        content="main",
        workspace_id="main_workspace",
    )
    atom_cache.ingest_atom(cached)
    bus._mock_storage.get_memory_by_alias.return_value = authorized

    result = await resolver.resolve("fact_shared", context=_context())

    assert result.kind == "atom"
    assert result.atom is authorized
    assert atom_cache.get_atom_by_alias("fact_shared") is authorized


@pytest.mark.asyncio
async def test_resolve_l2_hit_promotes_to_l1(resolver_parts):
    resolver, _pending_runtime, atom_cache, bus = resolver_parts
    atom = _make_memory(alias="fact_l2", content="from l2")
    bus._mock_storage.get_memory_by_alias.return_value = atom

    context = _context()
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
    resolver, _pending_runtime, _atom_cache, bus = resolver_parts
    bus._mock_storage.get_memory_by_alias.return_value = None

    result = await resolver.resolve("missing", context=_context())

    assert result.kind == "not_found"


@pytest.mark.asyncio
async def test_resolve_route_failure_raises_bus_unavailable(resolver_parts):
    resolver, _pending_runtime, _atom_cache, bus = resolver_parts
    bus._mock_storage.get_memory_by_alias.side_effect = KeyError("route missing")

    with pytest.raises(BusRouteUnavailableError):
        await resolver.resolve("fact_route_missing", context=_context())


@pytest.mark.asyncio
async def test_resolve_storage_failure_raises_storage_read_error(resolver_parts):
    resolver, _pending_runtime, _atom_cache, bus = resolver_parts
    bus._mock_storage.get_memory_by_alias.side_effect = RuntimeError("boom")

    with pytest.raises(StorageReadError):
        await resolver.resolve("fact_boom", context=_context())


@pytest.mark.asyncio
async def test_resolve_expired_pending_returns_expired(resolver_parts):
    resolver, pending_runtime, _atom_cache, _bus = resolver_parts
    pending = pending_runtime.register_write(
        content="will expire",
        title="Expire Test",
        reason=None,
        identity=Identity(user_id="test_user"),
        runtime_scope=make_runtime_scope(),
    )
    pending_runtime.expire(pending.pending_alias)

    result = await resolver.resolve(pending.pending_alias, context=_context())

    assert result.kind == "expired"
    assert result.requested_alias == pending.pending_alias
