"""actor Agent Profile 读取的 operation 授权边界。"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.errors import OperationDeniedError
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_runtime,
)


@pytest.mark.asyncio
async def test_profile_read_without_operation_denied_before_backing():
    """无 Profile 读取 operation 时，即使 backing 未装配也先表达操作拒绝。"""
    bus = GlobalSystemBus()
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", allowed_operations=frozenset())]
    )
    service = AgentApplicationService(
        bus,
        operation_authorizer=composition.authorizer,
        profile_reader=make_workspace_runtime(global_bus=bus).profiles,
    )
    access = await composition.authenticate()

    with pytest.raises(OperationDeniedError) as exc_info:
        await service.get_agent_profile(
            "worker", target_workspace=composition.default_workspace, access=access
        )

    assert exc_info.value.details["reason"] == "operation_not_allowed"
    assert exc_info.value.details["operation"] == "profile.read"
