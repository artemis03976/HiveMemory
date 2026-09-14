"""Patchouli 公开服务契约模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hivememory.core.models import AgentProfile, IdentityScope, TopicSnapshot
from hivememory.core.models.workspace_asset import RepresentationLease
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import AgentRunContext


@dataclass(frozen=True)
class StreamPrelude:
    """流式输出 Agent token 前发出的数据。"""

    topic_id: str
    is_new_topic: bool
    pool_topics: list[TopicSnapshot]
    memory_refs: list[Any]


@dataclass(frozen=True)
class PreparedAgentRun:
    """Complete context prepared by Patchouli for one Agent run.

    ``attachment_leases`` 是本轮附件选择的进程内 lease 关联（按用户选择
    顺序冻结）：由 prepare 边界 acquire，随 prepared run 交给 W1-E 编译
    边界，并在 finalize continuation 或 prepared cleanup 中释放。它不是
    新的 WorkspaceAsset 状态，也不进入任何序列化载荷。
    """

    agent_run_context: AgentRunContext
    gateway_decision: GatewayDecision
    stream_prelude: StreamPrelude
    generation_options: dict[str, Any] | None = field(default=None)
    attachment_leases: tuple[RepresentationLease, ...] = field(default=())

    @property
    def identity_scope(self) -> IdentityScope:
        """从权威 AgentRunContext 派生唯一的请求级身份作用域。"""
        return self.agent_run_context.identity_scope

    @property
    def interaction_id(self) -> str:
        return self.agent_run_context.interaction_id

    @property
    def agent_id(self) -> str:
        return self.agent_run_context.identity_scope.actor_identity.agent_id

    @property
    def topic_id(self) -> str:
        return self.agent_run_context.topic_id

    @property
    def user_message(self) -> str:
        return self.agent_run_context.user_message

    @property
    def agent_profile(self) -> AgentProfile:
        return self.agent_run_context.agent_profile
