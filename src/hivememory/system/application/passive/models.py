"""
被动接入事件模型

定义顶层 passive ingress 使用的统一事件输入模型与路由结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import InteractionPayload, RetrievalResponse


@dataclass(frozen=True)
class PassiveSessionKey:
    """被动 ingress 专用的会话分桶键。"""

    user_id: str
    agent_id: str
    session_id: str | None = None

    @classmethod
    def from_identity(cls, identity: Identity) -> PassiveSessionKey:
        return cls(
            user_id=identity.user_id,
            agent_id=identity.agent_id,
            session_id=identity.session_id,
        )

    @property
    def label(self) -> str:
        session = self.session_id or "<default>"
        return f"{self.user_id}:{self.agent_id}:{session}"


class PassiveIngressEvent(BaseModel):
    """被动模式统一事件输入模型。"""

    role: Literal["user", "assistant", "tool_call", "tool_result"]
    content: str
    action_id: str | None = None
    tool_name: str | None = None
    tool_kind: str | None = None
    tool_args: dict[str, Any] | None = None
    target: str | None = None
    status: str | None = None
    render_as: str = "plain"


@dataclass(frozen=True)
class PassiveIngressOutcome:
    """被动事件路由结果。"""

    kind: Literal["user", "buffered", "ignored"]
    gateway_decision: GatewayDecision | None = None
    retrieval_result: RetrievalResponse | None = None
    flushed: tuple[InteractionPayload, str | None] | None = None


__all__ = [
    "PassiveSessionKey",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
]
