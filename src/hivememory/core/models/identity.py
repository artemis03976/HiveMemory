"""HiveMemory 核心身份模型。

集中承载三条正交身份轴，作为跨领域传播的唯一身份事实：

- ``ActorIdentity``：谁在执行；
- ``WorkspaceIdentity``：正在访问哪个资源归属域；
- ``IdentityScope``：前两者的冻结组合，是 W0 唯一的公共身份作用域。

``IdentityScope`` 只冻结身份坐标，不携带 interaction/generation/agent_run/frame/
request/trace 等关联 ID，也不缓存授权结果或 Workspace 当前状态。
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.constants import (
    DEFAULT_AGENT_ID,
    DEFAULT_TEAM_ID,
    DEFAULT_USER_ID,
    SYSTEM_AGENT_ID,
)


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

    Attributes:
        user_id: 用户标识符
        agent_id: Agent 标识符
        team_id: 团队标识符（用于执行者可见性策略）
    """

    user_id: str = Field(default=DEFAULT_USER_ID, description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    team_id: str | None = Field(
        default=DEFAULT_TEAM_ID, description="团队 ID（用于执行者可见性策略）"
    )

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "agent_id": "chatbot",
            }
        },
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


def system_actor_for_workspace(belong_to: WorkspaceIdentity) -> ActorIdentity:
    """为非主动生成构造 system 发起者，参与 Agent 仅记录在贡献者集合。

    system 不属于任何 Team，也不能成为 PRIVATE policy 的 target，因此
    沿用普通读取规则即可将结算查重限制在所属 Workspace 的 PUBLIC 记忆内。
    """
    return ActorIdentity(
        user_id=belong_to.owner_user_id,
        agent_id=SYSTEM_AGENT_ID,
        team_id=None,
    )


class IdentityScope(BaseModel):
    """一次顶层操作冻结的执行者与目标 Workspace（Idea 前提第 1 条）。

    只回答两个问题：谁在执行（``actor_identity``）、这次操作作用于哪个
    资源归属域（``workspace_identity``）。它由授权点在操作授权通过后组装，
    向下流动到资源 owner 的公共边界，由 owner 拆为归属与发起者；不携带 interaction/generation/
    agent_run/frame/request/trace 等关联 ID，也不缓存授权结果或 Workspace
    当前状态。

    身份类型本身不承担授权规则（不变量 6）："actor 用户等于 workspace
    owner"等 owner 约束属于两阶段认证的第 2 阶段与两阶段授权的第 3 阶段，
    由认证网关（准入）与操作授权者（操作授权）检查，不在本模型构造时校验。
    """

    actor_identity: ActorIdentity
    workspace_identity: WorkspaceIdentity

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = [
    "ActorIdentity",
    "WorkspaceIdentity",
    "IdentityScope",
    "system_actor_for_workspace",
]
