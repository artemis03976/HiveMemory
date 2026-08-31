"""Memory schema v2 ownership、provenance 与 read policy 领域约束。"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.errors import OwnerMismatchError
from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    IdentityScope,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
    WorkspaceIdentity,
)


def _workspace(user_id: str = "u1", workspace_id: str = "main_workspace") -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id=user_id,
        workspace_key=workspace_id,
        workspace_id=workspace_id,
    )


def _atom() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            workspace_identity=_workspace(),
            source_agent_id="source-agent",
            source_team_id="source-team",
            access_policy=MemoryAccessPolicy.public(),
        ),
        index=IndexLayer(
            title="Workspace policy",
            summary="Memory v2 keeps one canonical ownership authority.",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="canonical ownership"),
    )


@pytest.mark.parametrize(
    ("visibility", "target_agent_id", "target_team_id"),
    [
        (MemoryVisibility.PUBLIC, "agent-a", None),
        (MemoryVisibility.PRIVATE, None, None),
        (MemoryVisibility.PRIVATE, "agent-a", "team-a"),
        (MemoryVisibility.TEAM, None, None),
        (MemoryVisibility.TEAM, "agent-a", "team-a"),
    ],
)
def test_access_policy_rejects_invalid_target_combinations(
    visibility: MemoryVisibility,
    target_agent_id: str | None,
    target_team_id: str | None,
) -> None:
    """捕获 source/target 混用或 PUBLIC 携带隐式授权目标的缺陷。"""
    with pytest.raises(ValidationError):
        MemoryAccessPolicy(
            visibility=visibility,
            target_agent_id=target_agent_id,
            target_team_id=target_team_id,
        )


def test_qdrant_payload_projects_v2_owner_without_legacy_user_authority() -> None:
    """捕获新写入继续双写 legacy user_id、形成第二 owner 权威的缺陷。"""
    payload = _atom().to_qdrant_payload()

    assert payload["schema_version"] == 2
    assert payload["meta"]["workspace_identity"] == {
        "owner_user_id": "u1",
        "workspace_key": "main_workspace",
        "workspace_id": "main_workspace",
    }
    assert payload["meta"]["owner_user_id"] == "u1"
    assert "user_id" not in payload["meta"]


def test_memory_atom_rejects_non_v2_schema() -> None:
    """捕获未知或旧 schema 被直接送入 v2 领域模型的缺陷。"""
    with pytest.raises(ValidationError):
        _atom().model_copy(update={"schema_version": 3}).model_validate(
            {**_atom().model_dump(mode="json"), "schema_version": 3}
        )


def test_identity_scope_rejects_actor_owner_drift() -> None:
    """捕获生成入口用 actor user 覆盖另一资源 owner 的缺陷。"""
    with pytest.raises(OwnerMismatchError):
        IdentityScope(
            actor_identity=Identity(user_id="actor", agent_id="agent-a"),
            workspace_identity=_workspace(user_id="owner"),
        )
