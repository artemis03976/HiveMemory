"""Memory ownership hard boundary 与 actor read policy 顺序。"""

from uuid import uuid4

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
    WorkspaceIdentity,
)
from hivememory.engines.retrieval.policy import memory_is_readable


def _workspace(workspace_id: str) -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id="u1",
        workspace_key=workspace_id,
        workspace_id=workspace_id,
    )


def _memory(policy: MemoryAccessPolicy) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            workspace_identity=_workspace("main_workspace"),
            source_agent_id="source-agent",
            source_team_id="source-team",
            access_policy=policy,
        ),
        index=IndexLayer(
            title="Scoped policy",
            summary="Policy evaluation must follow the ownership hard boundary.",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="scope"),
    )


@pytest.mark.parametrize(
    ("policy", "actor", "expected"),
    [
        (MemoryAccessPolicy.public(), Identity(user_id="u1", agent_id="other"), True),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="target-agent",
            ),
            Identity(user_id="u1", agent_id="target-agent"),
            True,
        ),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="target-agent",
            ),
            Identity(user_id="u1", agent_id="other"),
            False,
        ),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id="team-a",
            ),
            Identity(user_id="u1", agent_id="agent", team_id="team-a"),
            True,
        ),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id="team-a",
            ),
            Identity(user_id="u1", agent_id="agent", team_id="team-b"),
            False,
        ),
    ],
)
def test_read_policy_is_applied_within_owning_workspace(
    policy: MemoryAccessPolicy,
    actor: Identity,
    expected: bool,
) -> None:
    """捕获 PRIVATE/TEAM target 匹配方向错误或 PUBLIC 被错误拒绝的缺陷。"""
    assert memory_is_readable(
        _memory(policy),
        workspace_identity=_workspace("main_workspace"),
        actor_identity=actor,
    ) is expected


def test_public_policy_cannot_cross_workspace() -> None:
    """捕获先判断 PUBLIC 后判断 owner、导致全局公开的越权缺陷。"""
    assert not memory_is_readable(
        _memory(MemoryAccessPolicy.public()),
        workspace_identity=_workspace("isolation_workspace"),
        actor_identity=Identity(user_id="u1", agent_id="source-agent"),
    )
