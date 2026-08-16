from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
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

    assert atom.index.memory_type == MemoryType.AGENT_PROFILE
    bus.request.assert_awaited_once_with(PatchouliLocalRoutes.MEMORY_CREATE, atom)


@pytest.mark.asyncio
async def test_list_agent_profiles_uses_agent_profile_filter(bus):
    service = AgentProfileManagementService(bus=bus)

    await service.list_agent_profiles()

    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_LIST,
        filters={"index.memory_type": "AGENT_PROFILE"},
        limit=100,
    )
