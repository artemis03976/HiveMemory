from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=user_id),
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
async def test_list_memories_uses_memory_list_and_refreshes_vitality(bus):
    atom = _make_memory_atom()

    async def request(route, *args, **kwargs):
        if route == PatchouliLocalRoutes.MEMORY_LIST:
            return [atom]
        if route == PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY:
            return None
        raise AssertionError(route)

    bus.request.side_effect = request
    service = MemoryManagementService(bus=bus)
    access_context = make_access_context(user_id="u1")

    atoms = await service.list_memories(
        access_context=access_context,
        filters={"index.memory_type": "FACT"},
        limit=10,
    )

    # 编排顺序契约：MEMORY_LIST → REFRESH_MEMORY_VITALITY(persist=False)
    assert bus.request.await_args_list[0].args == (PatchouliLocalRoutes.MEMORY_LIST,)
    assert bus.request.await_args_list[0].kwargs == {
        "query": None,
        "filters": {"index.memory_type": "FACT"},
        "limit": 10,
        "access_context": access_context,
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
    access_context = make_access_context(user_id="u1")

    atoms = await service.list_memories(
        access_context=access_context,
        query="test",
        limit=5,
        exclude_types=[MemoryType.AGENT_PROFILE.value],
        refresh_vitality=False,
    )

    assert atoms == [fact]
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_LIST,
        access_context=access_context,
        query="test",
        filters=None,
        limit=5,
    )


@pytest.mark.asyncio
async def test_get_memory_requests_memory_get_and_skips_refresh_when_missing(bus):
    memory_id = uuid4()
    bus.request.return_value = None
    service = MemoryManagementService(bus=bus)
    access_context = make_access_context(user_id="u1")

    assert await service.get_memory(
        memory_id,
        access_context=access_context,
    ) is None
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_GET,
        memory_id,
        access_context=access_context,
    )
