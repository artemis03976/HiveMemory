from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, MetaData, PayloadLayer
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


@pytest.fixture
def bus():
    bus = AsyncMock()
    bus.request = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_create_memory_requests_memory_create(bus):
    service = MemoryManagementService(bus=bus)
    atom = _make_memory_atom(title="Created memory")

    result = await service.create_memory(atom)

    assert result is atom
    bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_CREATE, atom)


@pytest.mark.asyncio
async def test_list_memories_uses_memory_list_and_refreshes_vitality(bus):
    atom = _make_memory_atom()

    async def request(route, *args, **kwargs):
        if route == PatchouliLocalRoutes.MEMORY_LIST:
            return [atom]
        if route == PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY:
            atom.meta.vitality_score = 33.0
            return None
        raise AssertionError(route)

    bus.request.side_effect = request
    service = MemoryManagementService(bus=bus)

    atoms = await service.list_memories(
        filters={"meta.user_id": "u1", "index.memory_type": "FACT"},
        limit=10,
    )

    assert atoms == [atom]
    assert atoms[0].meta.vitality_score == 33.0
    assert bus.request.await_args_list[0].args == (PatchouliLocalRoutes.MEMORY_LIST,)
    assert bus.request.await_args_list[0].kwargs == {
        "query": None,
        "filters": {"meta.user_id": "u1", "index.memory_type": "FACT"},
        "limit": 10,
    }
    assert bus.request.await_args_list[1].args == (
        PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
        [atom],
    )
    assert bus.request.await_args_list[1].kwargs == {"persist": False}


@pytest.mark.asyncio
async def test_list_memories_excludes_agent_profiles_after_route_response(bus):
    fact = _make_memory_atom(title="Fact")
    profile = _make_memory_atom(title="Agent")
    profile.index.memory_type = MemoryType.AGENT_PROFILE
    bus.request.return_value = [fact, profile]
    service = MemoryManagementService(bus=bus)

    atoms = await service.list_memories(
        query="test",
        limit=5,
        exclude_types=[MemoryType.AGENT_PROFILE.value],
        refresh_vitality=False,
    )

    assert atoms == [fact]
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_LIST,
        query="test",
        filters=None,
        limit=5,
    )


@pytest.mark.asyncio
async def test_get_memory_requests_memory_get_and_skips_refresh_when_missing(bus):
    memory_id = uuid4()
    bus.request.return_value = None
    service = MemoryManagementService(bus=bus)

    assert await service.get_memory(memory_id) is None
    bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_GET, memory_id)


@pytest.mark.asyncio
async def test_update_memory_updates_editable_fields_and_requests_memory_update(bus):
    atom = _make_memory_atom()

    async def request(route, *args, **kwargs):
        if route == PatchouliLocalRoutes.MEMORY_GET:
            return atom
        if route == PatchouliLocalRoutes.MEMORY_UPDATE:
            return None
        raise AssertionError(route)

    bus.request.side_effect = request
    service = MemoryManagementService(bus=bus)

    updated = await service.update_memory(
        atom.id,
        title="Updated",
        summary="Updated summary",
        content="Updated content",
        alias="updated-alias",
        tags=["updated"],
        agent_config={"mode": "test"},
    )

    assert updated is atom
    assert atom.index.title == "Updated"
    assert atom.index.summary == "Updated summary"
    assert atom.payload.content == "Updated content"
    assert atom.index.alias == "updated-alias"
    assert atom.index.tags == ["updated"]
    assert atom.payload.artifacts.agent_config == {"mode": "test"}
    assert bus.request.await_args_list[0].args == (PatchouliLocalRoutes.MEMORY_GET, atom.id)
    assert bus.request.await_args_list[1].args == (PatchouliLocalRoutes.MEMORY_UPDATE, atom)


@pytest.mark.asyncio
async def test_record_feedback_requests_lifecycle_route(bus):
    mid = uuid4()
    expected = object()
    bus.request.return_value = expected
    service = MemoryManagementService(bus=bus)

    result = await service.record_feedback(mid, positive=True, source="ui.memory_ref")

    assert result is expected
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
        mid,
        positive=True,
        source="ui.memory_ref",
    )


@pytest.mark.asyncio
async def test_retrieve_routes_to_memory_retrieve(bus):
    service = MemoryManagementService(bus=bus)
    request = RetrievalRequest(semantic_query="query")
    response = RetrievalResponse()
    bus.request.return_value = response

    result = await service.retrieve(request, mode="passive")

    assert result is response
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_RETRIEVE,
        request,
    )


@pytest.mark.asyncio
async def test_retrieve_by_aliases_routes_to_memory_retrieve_by_aliases(bus):
    service = MemoryManagementService(bus=bus)
    response = RetrievalResponse()
    bus.request.return_value = response

    result = await service.retrieve_by_aliases(["a", "b"], identity="identity", mode="active")

    assert result is response
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
        ["a", "b"],
        "identity",
    )


@pytest.mark.asyncio
async def test_retrieve_does_not_forward_deprecated_mode_to_local_handler():
    local_bus = PatchouliBus()
    service = MemoryManagementService(bus=local_bus)
    request = RetrievalRequest(semantic_query="query")
    response = RetrievalResponse()
    seen = {}

    async def retrieve_handler(actual_request):
        seen["request"] = actual_request
        return response

    local_bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieve_handler)

    result = await service.retrieve(request, mode="passive")

    assert result is response
    assert seen["request"] is request


@pytest.mark.asyncio
async def test_retrieve_by_aliases_does_not_forward_deprecated_mode_to_local_handler():
    local_bus = PatchouliBus()
    service = MemoryManagementService(bus=local_bus)
    response = RetrievalResponse()
    seen = {}

    async def retrieve_by_aliases_handler(aliases, identity=None):
        seen["aliases"] = aliases
        seen["identity"] = identity
        return response

    local_bus.register(
        PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
        retrieve_by_aliases_handler,
    )

    result = await service.retrieve_by_aliases(
        ["a", "b"],
        identity="identity",
        mode="active",
    )

    assert result is response
    assert seen == {"aliases": ["a", "b"], "identity": "identity"}
