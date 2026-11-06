"""Workspace 资源键、默认 Workspace 解析与严格作用域校验。"""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models.identity import (
    IdentityScope,
    WorkspaceIdentity,
    _validate_non_empty,
)

MAIN_WORKSPACE_ID = "main_workspace"
"""公共产品入口解析到的默认 Workspace 标识。"""

ISOLATION_WORKSPACE_ID = "isolation_workspace"
"""仅供内部隔离验证使用的第二 Workspace 标识。"""


class WorkspaceTopicKey(BaseModel):
    """Topic 在短期存储中的权威复合键。"""

    owner_user_id: str
    workspace_id: str
    topic_id: str

    @field_validator("owner_user_id", "workspace_id", "topic_id")
    @classmethod
    def _require_non_empty(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, info.field_name)

    model_config = ConfigDict(frozen=True)


def resolve_default_workspace_identity(owner_user_id: str) -> WorkspaceIdentity:
    """在顶层入口为当前用户解析唯一默认 Workspace。"""
    return WorkspaceIdentity(
        owner_user_id=owner_user_id,
        workspace_key=MAIN_WORKSPACE_ID,
        workspace_id=MAIN_WORKSPACE_ID,
    )


def require_identity_scope(
    identity_scope: IdentityScope | None,
) -> IdentityScope:
    """在内部边界拒绝缺失或错误类型的 Workspace 作用域。"""
    if not isinstance(identity_scope, IdentityScope):
        raise ScopeRequiredError()
    return identity_scope


__all__ = [
    "MAIN_WORKSPACE_ID",
    "ISOLATION_WORKSPACE_ID",
    "WorkspaceIdentity",
    "IdentityScope",
    "WorkspaceTopicKey",
    "resolve_default_workspace_identity",
    "require_identity_scope",
]
