from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    OMNI_DOLL_PROFILE,
    PayloadLayer,
)
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


def _make_memory_atom(title: str = "Worker") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["agent"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="agent profile content"),
    )


@pytest.fixture
def bus():
    bus = AsyncMock()
    bus.request = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_create_agent_profile_forces_profile_type_and_requests_memory_create(bus):
    service = AgentProfileManagementService(bus=bus)
    atom = _make_memory_atom()

    result = await service.create_agent_profile(atom)

    assert result is atom
    assert atom.index.memory_type == MemoryType.AGENT_PROFILE
    bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_CREATE, atom)


@pytest.mark.asyncio
async def test_list_agent_profiles_uses_agent_profile_filter(bus):
    bus.request.return_value = []
    service = AgentProfileManagementService(bus=bus)

    assert await service.list_agent_profiles() == []
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_LIST,
        filters={"index.memory_type": "AGENT_PROFILE"},
        limit=100,
    )


@pytest.mark.asyncio
async def test_get_agent_profile_requests_profile_route(bus):
    bus.request.return_value = OMNI_DOLL_PROFILE
    service = AgentProfileManagementService(bus=bus)

    result = await service.get_agent_profile("omni_doll")

    assert result is OMNI_DOLL_PROFILE
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.GET_AGENT_PROFILE,
        "omni_doll",
    )
