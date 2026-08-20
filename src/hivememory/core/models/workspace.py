"""Workspace 资源键、默认作用域解析与兼容别名。"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models.identity import (
    ActorIdentity,
    IdentityScope,
    WorkspaceIdentity,
    _validate_non_empty,
)

MAIN_WORKSPACE_ID = "main_workspace"
"""公共产品入口解析到的默认 Workspace 标识。"""

ISOLATION_WORKSPACE_ID = "isolation_workspace"
"""仅供内部隔离验证使用的第二 Workspace 标识。"""


# P2.5 兼容别名：WorkspaceAccessContext 只能作为 IdentityScope 的直接别名，
# 不再拥有自己的字段、validator 或 wire schema，也不再包含 interaction_id。
WorkspaceAccessContext = IdentityScope


class WorkspaceTopicKey(BaseModel):
    """Topic 在短期存储中的权威复合键。"""

    owner_user_id: str
    workspace_id: str
    topic_id: str

    @field_validator("owner_user_id", "workspace_id", "topic_id")
    @classmethod
    def _require_non_empty(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, info.field_name)

    @classmethod
    def from_access_context(
        cls,
        access_context: IdentityScope,
        topic_id: str,
    ) -> Self:
        """从已验证的访问作用域构造 Topic 复合键。"""
        workspace = access_context.workspace_identity
        return cls(
            owner_user_id=workspace.owner_user_id,
            workspace_id=workspace.workspace_id,
            topic_id=topic_id,
        )

    model_config = ConfigDict(frozen=True)


def resolve_default_workspace_identity(owner_user_id: str) -> WorkspaceIdentity:
    """在顶层入口为当前用户解析唯一默认 Workspace。"""
    return WorkspaceIdentity(
        owner_user_id=owner_user_id,
        workspace_key=MAIN_WORKSPACE_ID,
        workspace_id=MAIN_WORKSPACE_ID,
    )


def resolve_default_workspace_access(
    actor_identity: ActorIdentity,
) -> IdentityScope:
    """在顶层入口一次性冻结默认 Workspace 的 IdentityScope。

    P2.5 起不再生成或接受 interaction_id；关联 ID 由具体领域载体独立持有。
    """
    return IdentityScope(
        actor_identity=actor_identity,
        workspace_identity=resolve_default_workspace_identity(actor_identity.user_id),
    )


def build_internal_workspace_access(
    actor_identity: ActorIdentity,
    workspace_id: str,
) -> IdentityScope:
    """为内部服务和隔离测试显式构造非默认 Workspace 的 IdentityScope。

    该 seam 不属于 HTTP 产品入口，也不会创建或注册 Workspace。
    """
    workspace = _validate_non_empty(workspace_id, "workspace_id")
    return IdentityScope(
        actor_identity=actor_identity,
        workspace_identity=WorkspaceIdentity(
            owner_user_id=actor_identity.user_id,
            workspace_key=workspace,
            workspace_id=workspace,
        ),
    )


def require_workspace_access_context(
    access_context: IdentityScope | None,
) -> IdentityScope:
    """在内部边界拒绝缺失或错误类型的 Workspace 作用域。"""
    if not isinstance(access_context, IdentityScope):
        raise ScopeRequiredError()
    return access_context


__all__ = [
    "MAIN_WORKSPACE_ID",
    "ISOLATION_WORKSPACE_ID",
    "WorkspaceIdentity",
    "IdentityScope",
    "WorkspaceAccessContext",
    "WorkspaceTopicKey",
    "resolve_default_workspace_identity",
    "resolve_default_workspace_access",
    "build_internal_workspace_access",
    "require_workspace_access_context",
]
