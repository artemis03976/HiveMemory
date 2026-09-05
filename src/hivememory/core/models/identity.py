"""HiveMemory 核心身份模型。

集中承载三条正交身份轴，作为跨领域传播的唯一身份事实：

- ``ActorIdentity``：谁在执行；
- ``WorkspaceIdentity``：正在访问哪个资源归属域；
- ``IdentityScope``：前两者的冻结组合，是 W0 唯一的公共身份作用域。

``IdentityScope`` 只冻结身份坐标，不携带 interaction/generation/agent_run/frame/
request/trace 等关联 ID，也不缓存授权结果或 Workspace 当前状态。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.constants import DEFAULT_AGENT_ID, DEFAULT_TEAM_ID, DEFAULT_USER_ID
from hivememory.core.errors import OwnerMismatchError


def _validate_non_empty(value: str, field_name: str) -> str:
    """拒绝空白标识，避免领域层出现隐式默认值。"""
    if not value or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value.strip()


class ActorIdentity(BaseModel):
    """
    执行者身份标识 - 回答"谁在执行本次操作"。

    用于替代散落的 user_id, agent_id 参数，
    提供统一的执行者身份标识和便捷的操作方法。

    ``session_id`` 仅作为旧协议/旧调用的兼容字段保留；当前 Topic
    生命周期不依赖它，也不应将其用作 Workspace、Topic 或资源身份。话题的
    生命周期由 PerceptionLayer 的 ``topic_id`` 管理。

    Attributes:
        user_id: 用户标识符
        agent_id: Agent 标识符
        team_id: 团队标识符（用于执行者可见性策略）
    """
    user_id: str = Field(default=DEFAULT_USER_ID, description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    team_id: str | None = Field(default=DEFAULT_TEAM_ID, description="团队 ID（用于执行者可见性策略）")
    session_id: str | None = Field(default=None, description="会话 ID（兼容字段）")

    @property
    def buffer_key(self) -> str:
        """生成用于缓冲区的唯一键"""
        return f"{self.user_id}:{self.agent_id}"

    @property
    def is_valid(self) -> bool:
        """检查身份标识是否有效"""
        return bool(self.user_id and self.agent_id)

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "agent_id": "chatbot",
                "session_id": "sess_456",
            }
        }
    )


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


class IdentityScope(BaseModel):
    """一次顶层操作冻结的执行者与 Workspace 访问硬边界。

    只回答两个问题：谁在执行（``actor_identity``）、正在访问哪个资源归属域
    （``workspace_identity``）。不携带 interaction/generation/agent_run/frame/
    request/trace 等关联 ID，也不缓存授权结果或 Workspace 当前状态。
    """

    actor_identity: ActorIdentity
    workspace_identity: WorkspaceIdentity

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
        """返回覆盖完整访问作用域的稳定指纹。"""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = [
    "ActorIdentity",
    "WorkspaceIdentity",
    "IdentityScope",
]
