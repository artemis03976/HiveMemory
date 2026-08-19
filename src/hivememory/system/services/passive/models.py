"""
被动接入事件模型

定义顶层 passive ingress 使用的统一事件输入模型、外部会话分桶键与路由结果。

契约要点（v0.6.0 设计 §4）：
    - `source + external_conversation_id` 构成外部会话命名空间，
      不能只按 `user_id + agent_id` 分桶。
    - `source + external_event_id` 是幂等键；同一事件重试不得重复追加、
      重复 retrieval 或重复提交 interaction。
    - `is_final=True` 表示当前事件完成该 turn；它不等价于 assistant role，
      因为外部 assistant 可能分段输出，也可能在 tool result 后才结束。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from hivememory.core.models import WorkspaceAccessContext
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import RetrievalResponse


@dataclass(frozen=True)
class PassiveConversationKey:
    """被动 ingress 的外部会话分桶键。

    外部来源与外部会话 ID 一起构成命名空间，再叠加 HiveMemory 身份维度，
    确保不同 connector 的同名会话 ID 不会互相污染。
    """

    source: str
    external_conversation_id: str
    owner_user_id: str
    workspace_id: str
    agent_id: str
    team_id: str | None = None

    @classmethod
    def build(
        cls,
        *,
        source: str,
        external_conversation_id: str,
        access_context: WorkspaceAccessContext,
    ) -> PassiveConversationKey:
        return cls(
            source=source,
            external_conversation_id=external_conversation_id,
            owner_user_id=access_context.workspace_identity.owner_user_id,
            workspace_id=access_context.workspace_identity.workspace_id,
            agent_id=access_context.actor_identity.agent_id,
            team_id=access_context.actor_identity.team_id,
        )

    @property
    def label(self) -> str:
        team = self.team_id or "<no-team>"
        return (
            f"{self.source}/{self.external_conversation_id}"
            f"@{self.owner_user_id}:{self.workspace_id}:{self.agent_id}:{team}"
        )

    @property
    def ordering_key(self) -> str:
        """为通用 submission queue 提供稳定的会话内顺序键。"""
        return self.label


DEFAULT_PASSIVE_SOURCE = "external"
DEFAULT_EXTERNAL_CONVERSATION_ID = "default"

SealReason = Literal[
    "next_user",
    "explicit_final",
    "idle_timeout",
    "manual_flush",
    "shutdown_drain",
]


def _new_external_event_id() -> str:
    return f"pie_{uuid4().hex}"


class PassiveIngressEvent(BaseModel):
    """被动模式统一事件输入模型。"""

    # ---------- 外部来源与关联标识 ----------
    source: str = Field(
        default=DEFAULT_PASSIVE_SOURCE,
        description="外部来源标识，如 claude_code / codex / telegram_bot",
    )
    external_conversation_id: str = Field(
        default=DEFAULT_EXTERNAL_CONVERSATION_ID,
        description="外部会话 ID，与 source 一起构成外部会话命名空间",
    )
    external_event_id: str = Field(
        default_factory=_new_external_event_id,
        description="外部事件 ID，与 source 一起构成进程内幂等键",
    )
    turn_id: str | None = Field(
        default=None,
        description="connector 能提供时的外部 turn 关联 ID",
    )
    occurred_at: datetime = Field(
        default_factory=datetime.now,
        description="事件在外部系统发生的时间",
    )
    sequence: int | None = Field(
        default=None,
        description="connector 能提供时的外部事件序号",
    )
    is_final: bool = Field(
        default=False,
        description="该事件是否完成当前 turn（与 role 无关）",
    )

    # ---------- 事件内容 ----------
    role: Literal["user", "assistant", "tool_call", "tool_result"]
    content: str
    action_id: str | None = None
    tool_name: str | None = None
    tool_kind: str | None = None
    tool_args: dict[str, Any] | None = None
    target: str | None = None
    status: str | None = None
    render_as: str = "plain"

    @property
    def dedup_key(self) -> tuple[str, str]:
        """进程内幂等键。"""
        return (self.source, self.external_event_id)

    def conversation_key(
        self,
        access_context: WorkspaceAccessContext,
    ) -> PassiveConversationKey:
        return PassiveConversationKey.build(
            source=self.source,
            external_conversation_id=self.external_conversation_id,
            access_context=access_context,
        )


@dataclass(frozen=True)
class PassiveIngressOutcome:
    """被动事件路由结果。

    只承载 service 构造公共响应所需的字段。队列状态属于内部观测量，
    不在 outcome 中重复携带。
    """

    kind: Literal["user", "buffered", "duplicate", "ignored"]
    gateway_decision: GatewayDecision | None = None
    retrieval_result: RetrievalResponse | None = None


__all__ = [
    "DEFAULT_EXTERNAL_CONVERSATION_ID",
    "DEFAULT_PASSIVE_SOURCE",
    "PassiveConversationKey",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
    "SealReason",
]
