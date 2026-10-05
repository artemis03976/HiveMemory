"""Actor 身份与 system 发起者契约：会话兼容读取和结算查重可见性。"""

import pytest

from hivememory.core.memory_access import access_policy_permits
from hivememory.core.models import (
    ActorIdentity,
    MemoryAccessPolicy,
    MemoryVisibility,
    system_actor_for_workspace,
)
from tests.helpers.workspace import make_workspace_identity


def test_legacy_actor_session_does_not_affect_identity_or_serialization():
    """旧 JSON 的 session 键可读取，但不参与身份相等性、hash 或新记录。"""
    restored = ActorIdentity.model_validate(
        {"user_id": "u1", "agent_id": "agent-a", "team_id": "team-a", "session_id": "old"}
    )
    current = ActorIdentity(user_id="u1", agent_id="agent-a", team_id="team-a")

    assert restored == current
    assert hash(restored) == hash(current)
    assert restored.model_dump() == {
        "user_id": "u1",
        "agent_id": "agent-a",
        "team_id": "team-a",
    }
    assert set(ActorIdentity.model_fields) == {"user_id", "agent_id", "team_id"}


@pytest.mark.parametrize(
    ("policy", "readable"),
    [
        (MemoryAccessPolicy.public(), True),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="agent-a",
            ),
            False,
        ),
        (
            MemoryAccessPolicy(visibility=MemoryVisibility.TEAM, target_team_id="team-a"),
            False,
        ),
    ],
)
def test_system_actor_for_workspace_only_reads_public_policy(policy, readable):
    """结算发起者使用 owner 用户、无 Team 的 system 身份，只能读取 PUBLIC。"""
    actor = system_actor_for_workspace(make_workspace_identity(owner_user_id="owner-a"))

    assert actor.model_dump() == {"user_id": "owner-a", "agent_id": "system", "team_id": None}
    assert access_policy_permits(policy, actor) is readable
