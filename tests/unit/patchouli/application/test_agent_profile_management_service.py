from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


def _make_memory_atom(title: str = "Worker", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=user_id),
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
    access_context = make_access_context(user_id="u1")

    result = await service.create_agent_profile(access_context, atom)

    assert atom.index.memory_type == MemoryType.AGENT_PROFILE
    assert result is atom
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_CREATE,
        access_context,
        atom,
    )


@pytest.mark.asyncio
async def test_create_agent_profile_rejects_foreign_workspace_atom(bus):
    """捕获 profile 创建在下游前篡改异域 Memory 类型的缺陷。"""
    service = AgentProfileManagementService(bus=bus)
    foreign_atom = _make_memory_atom(user_id="u2")

    with pytest.raises(WorkspaceMismatchError):
        await service.create_agent_profile(
            make_access_context(user_id="u1"),
            foreign_atom,
        )

    assert foreign_atom.index.memory_type == MemoryType.FACT
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_agent_profiles_uses_agent_profile_filter(bus):
    service = AgentProfileManagementService(bus=bus)
    access_context = make_access_context(user_id="u1")

    await service.list_agent_profiles(access_context=access_context)

    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_LIST,
        access_context=access_context,
        filters={"index.memory_type": "AGENT_PROFILE"},
        limit=100,
    )
