"""Topic HTTP 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hivememory.core.models import TopicLastTurn, TopicSnapshot
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)


class TopicLastTurnResponse(BaseModel):
    """活跃 Topic 最后一轮对话的 HTTP 表示。"""

    user: str = ""
    assistant: str = ""

    @classmethod
    def from_domain(cls, last_turn: TopicLastTurn | None) -> TopicLastTurnResponse | None:
        """从领域快照投影最后一轮对话。"""

        if last_turn is None:
            return None
        return cls(user=last_turn.user, assistant=last_turn.assistant)


class ActiveTopicResponse(BaseModel):
    """活跃 Topic 的 HTTP 表示。"""

    topic_id: str
    topic_title: str
    topic_summary: str = ""
    state_summary: str = ""
    last_turn: TopicLastTurnResponse | None = None
    block_count: int = 0
    total_tokens: int = 0
    last_accessed_at: float = 0.0
    model_used: str = Field(
        default="",
        description="最近 run 使用的模型展示名，空字符串表示尚未运行",
    )

    @classmethod
    def from_domain(cls, snapshot: TopicSnapshot) -> ActiveTopicResponse:
        """从不包含 Workspace 归属信息的安全字段构造 HTTP 投影。"""

        return cls(
            topic_id=snapshot.topic_id,
            topic_title=snapshot.topic_title,
            topic_summary=snapshot.topic_summary,
            state_summary=snapshot.state_summary,
            last_turn=TopicLastTurnResponse.from_domain(snapshot.last_turn),
            block_count=snapshot.block_count,
            total_tokens=snapshot.total_tokens,
            last_accessed_at=snapshot.last_accessed_at,
            model_used=snapshot.model_used,
        )


class ActiveTopicListResponse(BaseModel):
    """活跃 Topic 列表响应。"""

    topics: list[ActiveTopicResponse]


class TopicSettleResponse(BaseModel):
    """Topic settle HTTP 响应。"""

    topic_id: str
    generation_task_id: str | None = None
    generation_submitted: bool

    @classmethod
    def from_domain(cls, result: TopicSettleResult) -> TopicSettleResponse:
        """从 Topic settle 业务结果构造 HTTP 投影。"""

        return cls(
            topic_id=result.topic_id,
            generation_task_id=result.generation_task_id,
            generation_submitted=result.generation_submitted,
        )


class TopicDeleteResponse(BaseModel):
    """Topic delete/evict HTTP 响应。"""

    topic_id: str
    removed: bool

    @classmethod
    def from_domain(cls, result: TopicEvictionResult) -> TopicDeleteResponse:
        """从 Topic eviction 业务结果构造 HTTP 投影。"""

        return cls(topic_id=result.topic_id, removed=result.removed)


__all__ = [
    "ActiveTopicListResponse",
    "ActiveTopicResponse",
    "TopicDeleteResponse",
    "TopicLastTurnResponse",
    "TopicSettleResponse",
]
