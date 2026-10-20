"""actor CALL 目标解析：operation 授权边界与只交回图纸的结果。"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import AgentProfile, ResolvedAgentProfile
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.contracts import DeriveCallThreadRequest
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_runtime,
)


@pytest.mark.asyncio
async def test_call_target_without_operation_denied_before_backing():
    """无 CALL operation 时，即使 backing 未装配也先表达操作拒绝。"""
    bus = GlobalSystemBus()
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", allowed_operations=frozenset())]
    )
    service = ActorAgentService(
        operation_authorizer=composition.authorizer,
        profile_reader=make_workspace_runtime(global_bus=bus).profiles,
    )
    access = await composition.authenticate_thread()

    with pytest.raises(OperationDeniedError) as exc_info:
        await service.resolve_call_target(DeriveCallThreadRequest("worker"), access=access)

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "agent.call"


@pytest.mark.asyncio
async def test_call_target_resolution_returns_only_the_visible_profile():
    """持有 ``agent.call`` 时交回按调用方身份读取的图纸；派生不属于这项 actor 操作。"""
    bus = GlobalSystemBus()
    scopes = []

    async def profile_route(agent_alias, *, identity_scope):
        scopes.append(identity_scope)
        return ResolvedAgentProfile(profile=AgentProfile(agent_id=agent_alias, persona="被调用方"))

    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile_route)
    composition = make_access_composition(
        [make_actor_access_record(allowed_operations=[WorkspaceOperation.AGENT_CALL])]
    )
    service = ActorAgentService(
        operation_authorizer=composition.authorizer,
        profile_reader=make_workspace_runtime(global_bus=bus).profiles,
    )
    access = await composition.authenticate_thread()

    profile = await service.resolve_call_target(DeriveCallThreadRequest("worker"), access=access)

    assert (profile.agent_id, profile.persona) == ("worker", "被调用方")
    # 图纸按调用方的身份读取。
    assert [scope.actor_identity.agent_id for scope in scopes] == ["test_agent"]
