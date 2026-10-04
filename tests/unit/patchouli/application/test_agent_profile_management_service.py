from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.errors import ScopeRequiredError, WorkspaceMismatchError
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


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


def test_constructor_rejects_access_guard():
    """构造函数不再接收 access_guard：guard 注入已随边界返工删除。"""
    with pytest.raises(TypeError, match="access_guard"):
        AgentProfileManagementService(bus=AsyncMock(), access_guard=object())


@pytest.mark.asyncio
async def test_create_agent_profile_forces_profile_type_and_requests_memory_create(bus):
    service = AgentProfileManagementService(bus=bus)
    atom = _make_memory_atom()
    identity_scope = make_identity_scope(user_id="u1")

    result = await service.create_agent_profile(identity_scope, atom)

    assert atom.index.memory_type == MemoryType.AGENT_PROFILE
    assert result is atom
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_CREATE,
        identity_scope,
        atom,
    )


@pytest.mark.asyncio
async def test_create_agent_profile_rejects_foreign_workspace_atom(bus):
    """捕获 profile 创建在下游前篡改异域 Memory 类型的缺陷。"""
    service = AgentProfileManagementService(bus=bus)
    foreign_atom = _make_memory_atom(user_id="u2")

    with pytest.raises(WorkspaceMismatchError, match="workspace.mismatch"):
        await service.create_agent_profile(
            make_identity_scope(user_id="u1"),
            foreign_atom,
        )

    assert foreign_atom.index.memory_type == MemoryType.FACT
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_agent_profile_requires_identity_scope(bus):
    """identity_scope 缺失按 ScopeRequiredError 拒绝，且不触达后端。"""
    service = AgentProfileManagementService(bus=bus)

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        await service.create_agent_profile(None, _make_memory_atom())

    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_agent_profiles_uses_agent_profile_filter(bus):
    service = AgentProfileManagementService(bus=bus)
    identity_scope = make_identity_scope(user_id="u1")

    await service.list_agent_profiles(identity_scope=identity_scope)

    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_LIST,
        identity_scope=identity_scope,
        filters={"index.memory_type": "AGENT_PROFILE"},
        limit=100,
    )


@pytest.mark.asyncio
async def test_get_agent_profile_forwards_alias_and_scope_to_local_route(bus):
    service = AgentProfileManagementService(bus=bus)
    identity_scope = make_identity_scope(user_id="u1")

    await service.get_agent_profile("worker", identity_scope=identity_scope)

    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.GET_AGENT_PROFILE,
        "worker",
        identity_scope=identity_scope,
    )
