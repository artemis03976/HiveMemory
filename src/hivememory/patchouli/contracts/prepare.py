"""Patchouli prepare 的公开契约模型。

``PreparedAgentRun`` 由 prepare 路由返回、交回 finalize/cleanup 路由，并由 workspace
的任务进程读取其中的 Topic 与检索结果，因此位于 Patchouli 公开的 contracts 子包。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hivememory.core.models import IdentityScope, TopicData, TopicSnapshot
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import RetrievalResponse


@dataclass(frozen=True)
class PreparedAgentRun:
    """Patchouli prepare 的结果，也是 finalize 与 cleanup 的输入句柄。

    只承载 Patchouli 自己的内容（Topic 准备与未编译检索结果）：Profile
    解析、附件租借与记忆/附件编译由 chat 任务进程在 CPU 分配时完成。
    ``topic_context`` 与 ``pool_topics`` 在 ConversationSession 一批之前
    暂时保留。它不进入任何序列化载荷。
    """

    identity_scope: IdentityScope
    interaction_id: str
    user_message: str
    gateway_decision: GatewayDecision
    topic_id: str
    is_new_topic: bool
    topic_context: TopicData | None = None
    pool_topics: list[TopicSnapshot] = field(default_factory=list)
    retrieval_result: RetrievalResponse = field(default_factory=RetrievalResponse)
    storage_available: bool = True

    @property
    def agent_id(self) -> str:
        return self.identity_scope.actor_identity.agent_id


__all__ = ["PreparedAgentRun"]
