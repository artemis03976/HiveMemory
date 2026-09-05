"""验证 Workspace 核心值对象、顶层解析和稳定失败语义。"""

import pytest
from pydantic import ValidationError

from hivememory.core.errors import (
    AssetExpiredError,
    AssetFailedError,
    AssetNotFoundError,
    AssetNotReadyError,
    AssetOperationConflictError,
    AssetRemovedError,
    OwnerMismatchError,
    ScopeRequiredError,
    StaleAssetResultError,
    WorkspaceMismatchError,
)
from hivememory.core.models import (
    ISOLATION_WORKSPACE_ID,
    MAIN_WORKSPACE_ID,
    ActorIdentity,
    IdentityScope,
    WorkspaceIdentity,
    WorkspaceTopicKey,
    build_internal_identity_scope,
    require_identity_scope,
    resolve_default_identity_scope,
)


@pytest.mark.parametrize("field_name", ["owner_user_id", "workspace_key", "workspace_id"])
def test_workspace_identity_rejects_blank_coordinate(field_name: str):
    """防止空坐标在领域深层被解释为默认 Workspace。"""
    values = {
        "owner_user_id": "user-a",
        "workspace_key": MAIN_WORKSPACE_ID,
        "workspace_id": MAIN_WORKSPACE_ID,
    }
    values[field_name] = "  "

    with pytest.raises(ValidationError, match="不能为空"):
        WorkspaceIdentity(**values)


def test_workspace_identity_rejects_different_key_and_id():
    """防止 MVP 期间出现无法唯一寻址的第二套 Workspace ID。"""
    with pytest.raises(ValidationError, match="workspace_key 与 workspace_id 相同"):
        WorkspaceIdentity(
            owner_user_id="user-a",
            workspace_key="workspace-key",
            workspace_id="workspace-id",
        )


def test_identity_scope_round_trip_preserves_scope_and_fingerprint():
    """防止序列化重建时丢失 actor 或 Workspace 坐标。"""
    original = resolve_default_identity_scope(
        ActorIdentity(user_id="user-a", agent_id="agent-a", team_id="team-a"),
    )

    restored = IdentityScope.model_validate_json(original.model_dump_json())

    assert restored == original


def test_identity_scope_is_frozen_and_recursively_immutable():
    """防止 Run 传播过程中原地改写 IdentityScope 的 hard boundary。"""
    scope = resolve_default_identity_scope(ActorIdentity(user_id="user-a"))

    with pytest.raises(ValidationError, match="frozen"):
        scope.actor_identity = ActorIdentity(user_id="user-b")
    with pytest.raises(ValidationError, match="frozen"):
        scope.workspace_identity.workspace_id = "other"


def test_default_resolver_builds_current_users_main_workspace():
    """防止公共入口把默认 Workspace 解析到其他 owner 或非规范名称。"""
    scope = resolve_default_identity_scope(ActorIdentity(user_id="user-a"))

    assert scope.workspace_identity == WorkspaceIdentity(
        owner_user_id="user-a",
        workspace_key=MAIN_WORKSPACE_ID,
        workspace_id=MAIN_WORKSPACE_ID,
    )


def test_internal_builder_can_address_isolation_workspace_explicitly():
    """防止双 Workspace 验收 seam 偷偷回退到 main_workspace。"""
    scope = build_internal_identity_scope(
        ActorIdentity(user_id="user-a"),
        ISOLATION_WORKSPACE_ID,
    )

    assert scope.workspace_identity.workspace_id == ISOLATION_WORKSPACE_ID
    assert scope.workspace_identity.owner_user_id == "user-a"


def test_identity_scope_rejects_cross_owner_actor():
    """防止 actor user 借用另一用户的 Workspace 资源域。"""
    workspace = WorkspaceIdentity(
        owner_user_id="owner-a",
        workspace_key=MAIN_WORKSPACE_ID,
        workspace_id=MAIN_WORKSPACE_ID,
    )

    with pytest.raises(OwnerMismatchError) as caught:
        IdentityScope(
            actor_identity=ActorIdentity(user_id="attacker"),
            workspace_identity=workspace,
        )

    assert caught.value.code == "workspace.owner_mismatch"


def test_identity_scope_rejects_interaction_id():
    """防止公共身份载体重新承载 interaction_id 等关联 ID。"""
    with pytest.raises(ValidationError):
        IdentityScope(
            actor_identity=ActorIdentity(user_id="user-a"),
            workspace_identity=WorkspaceIdentity(
                owner_user_id="user-a",
                workspace_key=MAIN_WORKSPACE_ID,
                workspace_id=MAIN_WORKSPACE_ID,
            ),
            interaction_id="interaction-a",
        )


def test_workspace_topic_key_round_trip_keeps_owner_and_workspace():
    """防止 Topic key 序列化后退化为裸 topic_id。"""
    scope = build_internal_identity_scope(
        ActorIdentity(user_id="user-a"),
        ISOLATION_WORKSPACE_ID,
    )
    original = WorkspaceTopicKey.from_identity_scope(scope, "topic-a")

    restored = WorkspaceTopicKey.model_validate_json(original.model_dump_json())

    assert restored == WorkspaceTopicKey(
        owner_user_id="user-a",
        workspace_id=ISOLATION_WORKSPACE_ID,
        topic_id="topic-a",
    )


def test_internal_boundary_rejects_missing_scope_with_stable_code():
    """防止领域内部在缺少 scope 时静默回退 main_workspace。"""
    with pytest.raises(ScopeRequiredError) as caught:
        require_identity_scope(None)

    assert caught.value.code == "workspace.scope_required"


def test_topic_key_construction_rejects_missing_scope_with_stable_code():
    """防止 from_identity_scope 对缺失作用域退回 AttributeError 或静默构造。"""
    with pytest.raises(ScopeRequiredError) as caught:
        WorkspaceTopicKey.from_identity_scope(None, "topic-a")

    assert caught.value.code == "workspace.scope_required"


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [
        (WorkspaceMismatchError, "workspace.mismatch"),
        (AssetNotFoundError, "workspace.asset.not_found"),
        (AssetExpiredError, "workspace.asset.expired"),
        (AssetNotReadyError, "workspace.asset.not_ready"),
        (AssetFailedError, "workspace.asset.failed"),
        (AssetRemovedError, "workspace.asset.removed"),
        (StaleAssetResultError, "workspace.asset.stale_result"),
        (AssetOperationConflictError, "workspace.asset.operation_conflict"),
    ],
)
def test_workspace_errors_expose_stable_machine_codes(error_type, expected_code: str):
    """防止调用方被迫依赖可变的人类异常文本做控制流。"""
    error = error_type("诊断消息")

    assert error.code == expected_code
    assert str(error) == "诊断消息"
