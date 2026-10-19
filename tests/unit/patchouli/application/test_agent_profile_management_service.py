"""AgentProfileManagementService 单元测试。

被测对象：Patchouli Profile 管理与读取 backing 的应用服务；local bus 为 mock（边界外协作者）。
Profile 读取经全局总线到真实 handler 的链路见
``tests/integration/alice/orchestration/test_call_profile_requests.py``。本文件只守护本层的
判断：scope 与资源归属校验、创建时强制 Profile 类型、列表按 Profile 类型过滤。
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_management import AgentProfileCreation
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


@pytest.mark.asyncio
async def test_create_agent_profile_fixes_profile_type_and_passes_materials_with_actor(bus):
    """Profile 入口固定类型，材料与本次发起者原样交给创建原语，返回其持久化结果。"""
    service = AgentProfileManagementService(bus=bus)
    identity_scope = make_identity_scope(user_id="u1")
    created = _make_memory_atom()
    bus.request = AsyncMock(return_value=created)

    result = await service.create_agent_profile(
        identity_scope,
        AgentProfileCreation(
            title="Worker",
            alias="worker",
            tags=("ops",),
            agent_config={"model": "m"},
        ),
    )

    assert result is created
    bus.request.assert_awaited_once_with(
        PatchouliLocalRoutes.MEMORY_CREATE,
        identity_scope.workspace_identity,
        from_actor=identity_scope.actor_identity,
        memory_type=MemoryType.AGENT_PROFILE,
        title="Worker",
        summary="",
        content="",
        tags=["ops"],
        alias="worker",
        agent_config={"model": "m"},
    )


@pytest.mark.asyncio
async def test_create_agent_profile_requires_identity_scope(bus):
    """identity_scope 缺失按 ScopeRequiredError 拒绝，且不触达后端。"""
    service = AgentProfileManagementService(bus=bus)

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        await service.create_agent_profile(None, AgentProfileCreation(title="W", alias="w"))

    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_agent_profiles_filters_by_profile_type_with_owner_semantics(bus):
    """管理列表只请求 Profile 类型的记忆（默认上限 100），按 owner-management 语义不做可见性过滤。"""
    service = AgentProfileManagementService(bus=bus)
    identity_scope = make_identity_scope(user_id="u1")

    await service.list_agent_profiles(identity_scope=identity_scope)

    route = bus.request.await_args.args[0]
    kwargs = bus.request.await_args.kwargs
    assert route == PatchouliLocalRoutes.MEMORY_LIST
    assert kwargs["belong_to"] == identity_scope.workspace_identity
    assert (kwargs["filters"], kwargs["limit"]) == ({"index.memory_type": "AGENT_PROFILE"}, 100)
    assert kwargs["enforce_actor_visibility"] is False


@pytest.mark.asyncio
async def test_profile_read_without_scope_does_not_enter_resolver(bus):
    """公开读取入口拒绝缺失 scope，且不继续执行内部 Profile 解析。"""
    service = AgentProfileManagementService(bus=bus)

    with pytest.raises(ScopeRequiredError):
        await service.get_agent_profile("custom_agent", identity_scope=None)

    bus.request.assert_not_awaited()
