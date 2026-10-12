"""actor Agent Profile 读取的 operation 授权边界。"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.errors import OperationDeniedError
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
        thread_spawner=composition.thread_spawner,
    )
    access = await composition.authenticate_thread()

    with pytest.raises(OperationDeniedError) as exc_info:
        await service.derive_call_thread(DeriveCallThreadRequest("worker"), access=access)

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "agent.call"
