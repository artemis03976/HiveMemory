"""Public Patchouli service contract models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hivememory.core.models import AgentProfile, Identity, IdentityScope, TopicSnapshot
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import AgentRunContext


@dataclass(frozen=True)
class StreamPrelude:
    """Data emitted before streaming Agent tokens."""

    topic_id: str
    is_new_topic: bool
    pool_topics: list[TopicSnapshot]
    memory_refs: list[Any]


@dataclass(frozen=True)
class PreparedAgentRun:
    """Complete context prepared by Patchouli for one Agent run."""

    agent_run_context: AgentRunContext
    gateway_decision: GatewayDecision
    stream_prelude: StreamPrelude
    generation_options: dict[str, Any] | None = field(default=None)

    @property
    def identity_scope(self) -> IdentityScope:
        """从权威 AgentRunContext 派生唯一的请求级身份作用域。"""
        return self.agent_run_context.identity_scope

    @property
    def identity(self) -> Identity:
        return self.identity_scope.actor_identity

    @property
    def interaction_id(self) -> str:
        return self.agent_run_context.interaction_id

    @property
    def agent_id(self) -> str:
        return self.agent_run_context.identity.agent_id

    @property
    def topic_id(self) -> str:
        return self.agent_run_context.topic_id

    @property
    def user_message(self) -> str:
        return self.agent_run_context.user_message

    @property
    def agent_profile(self) -> AgentProfile:
        return self.agent_run_context.agent_profile
