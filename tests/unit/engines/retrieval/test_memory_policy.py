"""Memory ownership hard boundary 与 actor read policy 顺序。"""

from uuid import uuid4

import pytest

from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryProvenance,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
    WorkspaceIdentity,
)
from hivememory.engines.retrieval.policy import memory_is_readable
from hivememory.utils.time import utc_now


def _workspace(workspace_id: str) -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id="u1",
        workspace_key=workspace_id,
        workspace_id=workspace_id,
    )


def _memory(policy: MemoryAccessPolicy) -> MemoryAtom:
    created_at = utc_now()
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            workspace_identity=_workspace("main_workspace"),
            provenance=MemoryProvenance(
                source_agent_id="source-agent",
                source_team_id="source-team",
            ),
            access_policy=policy,
            created_at=created_at,
            updated_at=created_at,
            lifecycle=MemoryLifecycleState(decay_anchor_at=created_at),
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
        (MemoryAccessPolicy.public(), ActorIdentity(user_id="u1", agent_id="other"), True),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="target-agent",
            ),
            ActorIdentity(user_id="u1", agent_id="target-agent"),
            True,
        ),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="target-agent",
            ),
            ActorIdentity(user_id="u1", agent_id="other"),
            False,
        ),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id="team-a",
            ),
            ActorIdentity(user_id="u1", agent_id="agent", team_id="team-a"),
            True,
        ),
        (
            MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id="team-a",
            ),
            ActorIdentity(user_id="u1", agent_id="agent", team_id="team-b"),
            False,
        ),
    ],
)
def test_read_policy_is_applied_within_owning_workspace(
    policy: MemoryAccessPolicy,
    actor: ActorIdentity,
    expected: bool,
) -> None:
    """捕获 PRIVATE/TEAM target 匹配方向错误或 PUBLIC 被错误拒绝的缺陷。"""
    assert (
        memory_is_readable(
            _memory(policy),
            workspace_identity=_workspace("main_workspace"),
            actor_identity=actor,
        )
        is expected
    )


def test_owner_management_read_skips_actor_visibility_within_workspace() -> None:
    """D4：管理读取在 ownership 通过后可读 Workspace 内 PRIVATE/TEAM Memory。"""
    private = MemoryAccessPolicy(
        visibility=MemoryVisibility.PRIVATE,
        target_agent_id="target-agent",
    )
    team = MemoryAccessPolicy(
        visibility=MemoryVisibility.TEAM,
        target_team_id="team-a",
    )

    for policy in (private, team):
        assert (
            memory_is_readable(
                _memory(policy),
                workspace_identity=_workspace("main_workspace"),
                actor_identity=ActorIdentity(user_id="u1", agent_id="any-agent"),
                enforce_actor_visibility=False,
            )
            is True
        )


def test_owner_management_read_still_enforces_workspace_boundary() -> None:
    """关闭 actor 可见性过滤不能绕过 ownership hard boundary。"""
    assert not memory_is_readable(
        _memory(MemoryAccessPolicy.public()),
        workspace_identity=_workspace("isolation_workspace"),
        actor_identity=ActorIdentity(user_id="u1", agent_id="any-agent"),
        enforce_actor_visibility=False,
    )


def test_public_policy_cannot_cross_workspace() -> None:
    """捕获先判断 PUBLIC 后判断 owner、导致全局公开的越权缺陷。"""
    assert not memory_is_readable(
        _memory(MemoryAccessPolicy.public()),
        workspace_identity=_workspace("isolation_workspace"),
        actor_identity=ActorIdentity(user_id="u1", agent_id="source-agent"),
    )


def test_changing_provenance_does_not_change_v2_visibility() -> None:
    """provenance 与授权分离：改变来源记录不得改变 v2 可见性。

    legacy 分支曾以 source_agent_id 推断 PRIVATE 目标；该测试防止
    V2 判定重新耦合来源字段。
    """
    atom = _memory(
        MemoryAccessPolicy(
            visibility=MemoryVisibility.PRIVATE,
            target_agent_id="target-agent",
        )
    )
    readable_before = memory_is_readable(
        atom,
        workspace_identity=_workspace("main_workspace"),
        actor_identity=ActorIdentity(user_id="u1", agent_id="other"),
    )

    atom.meta.provenance.source_agent_id = "system"
    atom.meta.provenance.source_team_id = None
    atom.meta.provenance.contributing_agent_ids = ("target-agent",)

    readable_after = memory_is_readable(
        atom,
        workspace_identity=_workspace("main_workspace"),
        actor_identity=ActorIdentity(user_id="u1", agent_id="other"),
    )

    assert readable_before is False
    assert readable_after is False
