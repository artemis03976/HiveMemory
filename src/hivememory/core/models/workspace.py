"""Workspace 身份、访问上下文与复合资源键。"""

import hashlib
import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.errors import OwnerMismatchError, ScopeRequiredError
from hivememory.core.models.interaction import Identity

MAIN_WORKSPACE_ID = "main_workspace"
"""公共产品入口解析到的默认 Workspace 标识。"""

ISOLATION_WORKSPACE_ID = "isolation_workspace"
"""仅供内部隔离验证使用的第二 Workspace 标识。"""


def _validate_non_empty(value: str, field_name: str) -> str:
    """拒绝空白标识，避免领域层出现隐式默认值。"""
    if not value or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value.strip()


class WorkspaceIdentity(BaseModel):
    """不可变的 Workspace 资源归属坐标。"""

    owner_user_id: str = Field(description="资源域所有者用户 ID")
    workspace_key: str = Field(description="Workspace 规范键")
    workspace_id: str = Field(description="Workspace 资源标识")

    @field_validator("owner_user_id", "workspace_key", "workspace_id")
    @classmethod
    def _require_non_empty(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def _require_mvp_key_identity(self) -> Self:
        if self.workspace_key != self.workspace_id:
            raise ValueError("Workspace MVP 要求 workspace_key 与 workspace_id 相同")
        return self

    model_config = ConfigDict(frozen=True)


class WorkspaceAccessContext(BaseModel):
    """一次顶层交互冻结的执行者与 Workspace 访问坐标。"""

    actor_identity: Identity
    workspace_identity: WorkspaceIdentity
    interaction_id: str = Field(description="一次顶层交互的唯一标识")

    @field_validator("interaction_id")
    @classmethod
    def _require_interaction_id(cls, value: str) -> str:
        return _validate_non_empty(value, "interaction_id")

    @model_validator(mode="after")
    def _require_same_owner(self) -> Self:
        if self.actor_identity.user_id != self.workspace_identity.owner_user_id:
            raise OwnerMismatchError(
                details={
                    "actor_user_id": self.actor_identity.user_id,
                    "owner_user_id": self.workspace_identity.owner_user_id,
                }
            )
        return self

    @property
    def scope_fingerprint(self) -> str:
        """返回覆盖完整访问上下文的稳定指纹。"""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    model_config = ConfigDict(frozen=True)


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
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> Self:
        """从已验证的访问上下文构造 Topic 复合键。"""
        workspace = access_context.workspace_identity
        return cls(
            owner_user_id=workspace.owner_user_id,
            workspace_id=workspace.workspace_id,
            topic_id=topic_id,
        )

    model_config = ConfigDict(frozen=True)


class WorkScopeSnapshot(BaseModel):
    """可持久化 work payload 使用的不可变作用域快照。"""

    actor_identity: Identity
    workspace_identity: WorkspaceIdentity
    interaction_id: str | None = None
    operation_id: str | None = None

    @field_validator("interaction_id", "operation_id")
    @classmethod
    def _normalize_optional_id(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _validate_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        if self.actor_identity.user_id != self.workspace_identity.owner_user_id:
            raise OwnerMismatchError(
                details={
                    "actor_user_id": self.actor_identity.user_id,
                    "owner_user_id": self.workspace_identity.owner_user_id,
                }
            )
        if (self.interaction_id is None) == (self.operation_id is None):
            raise ValueError("WorkScopeSnapshot 必须且只能包含 interaction_id 或 operation_id")
        return self

    @classmethod
    def from_access_context(cls, access_context: WorkspaceAccessContext) -> Self:
        """冻结 Chat 交互的 work scope，不读取任何进程当前状态。"""
        return cls(
            actor_identity=access_context.actor_identity,
            workspace_identity=access_context.workspace_identity,
            interaction_id=access_context.interaction_id,
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
    actor_identity: Identity,
    interaction_id: str,
) -> WorkspaceAccessContext:
    """在顶层入口一次性冻结默认 Workspace 访问上下文。"""
    return WorkspaceAccessContext(
        actor_identity=actor_identity,
        workspace_identity=resolve_default_workspace_identity(actor_identity.user_id),
        interaction_id=interaction_id,
    )


def build_internal_workspace_access(
    actor_identity: Identity,
    workspace_id: str,
    interaction_id: str,
) -> WorkspaceAccessContext:
    """为内部服务和隔离测试显式构造非默认 Workspace 上下文。

    该 seam 不属于 HTTP 产品入口，也不会创建或注册 Workspace。
    """
    workspace = _validate_non_empty(workspace_id, "workspace_id")
    return WorkspaceAccessContext(
        actor_identity=actor_identity,
        workspace_identity=WorkspaceIdentity(
            owner_user_id=actor_identity.user_id,
            workspace_key=workspace,
            workspace_id=workspace,
        ),
        interaction_id=interaction_id,
    )


def require_workspace_access_context(
    access_context: WorkspaceAccessContext | None,
) -> WorkspaceAccessContext:
    """在内部边界拒绝缺失或错误类型的 Workspace 上下文。"""
    if not isinstance(access_context, WorkspaceAccessContext):
        raise ScopeRequiredError()
    return access_context


__all__ = [
    "MAIN_WORKSPACE_ID",
    "ISOLATION_WORKSPACE_ID",
    "WorkspaceIdentity",
    "WorkspaceAccessContext",
    "WorkspaceTopicKey",
    "WorkScopeSnapshot",
    "resolve_default_workspace_identity",
    "resolve_default_workspace_access",
    "build_internal_workspace_access",
    "require_workspace_access_context",
]
