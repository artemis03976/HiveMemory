"""Topic HTTP 请求/响应模型。"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from hivememory.core.models import (
    TopicLastTurn,
    TopicProgressEntry,
    TopicSnapshot,
    TopicTurnExcerpt,
)
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


class TopicProgressEntryResponse(BaseModel):
    """进展记录一条的 HTTP 表示。"""

    text: str = Field(description="这一条的正文")
    from_turn_id: str = Field(
        description="指回原文的区间：覆盖的第一轮的轮次标识，区间从这一轮的用户消息（序号 0）开始"
    )
    to_turn_id: str = Field(description="指回原文的区间：覆盖的最后一轮的轮次标识")
    to_event_seq: int = Field(description="指回原文的区间：最后一轮中最后一个事件的序号")
    covered_until: int = Field(description="这一条覆盖到话题的第几轮（从 1 数起）")
    turn_count: int = Field(description="这一条覆盖的轮数")
    written_at: float = Field(description="写下这一条的 Unix 时间戳（秒）")

    @classmethod
    def from_record(cls, record: Sequence[TopicProgressEntry]) -> list[TopicProgressEntryResponse]:
        """投影整份进展记录；每一条覆盖的轮数是它与上一条的覆盖位置之差。"""

        entries: list[TopicProgressEntryResponse] = []
        start = 0
        for entry in record:
            entries.append(
                cls(
                    text=entry.text,
                    from_turn_id=entry.from_turn_id,
                    to_turn_id=entry.to_turn_id,
                    to_event_seq=entry.to_event_seq,
                    covered_until=entry.covered_until,
                    turn_count=entry.covered_until - start,
                    written_at=entry.written_at,
                )
            )
            start = entry.covered_until
        return entries


class TopicTurnExcerptResponse(BaseModel):
    """一轮交互节选的 HTTP 表示。"""

    turn_number: int = Field(description="这一轮在话题中的轮次号（从 1 数起）")
    user: str = Field(default="", description="用户消息的节选")
    assistant: str = Field(default="", description="回复的节选")

    @classmethod
    def from_domain(cls, excerpt: TopicTurnExcerpt) -> TopicTurnExcerptResponse:
        return cls(turn_number=excerpt.turn_number, user=excerpt.user, assistant=excerpt.assistant)


class ActiveTopicResponse(BaseModel):
    """活跃 Topic 的 HTTP 表示。"""

    topic_id: str
    topic_title: str
    topic_summary: str = ""
    fold_summary: str = ""
    last_turn: TopicLastTurnResponse | None = None
    block_count: int = 0
    total_tokens: int = 0
    model_used: str = Field(
        default="",
        description="最近 run 使用的模型展示名，空字符串表示尚未运行",
    )
    last_update: float = Field(description="话题最近一次写入的 Unix 时间戳（秒）")
    has_unsettled_interactions: bool = Field(
        description="话题是否有还没有沉淀为长期记忆的交互；没有时空闲不会再触发沉淀"
    )
    progress_record: list[TopicProgressEntryResponse] = Field(
        description="话题的进展记录，按写下的顺序；还没有写过一条时为空"
    )
    unrecorded_turn_count: int = Field(description="进展记录还没有覆盖到的交互轮数")
    unrecorded_turns: list[TopicTurnExcerptResponse] = Field(
        description="还没有整理的交互中最近至多 3 轮的节选，按轮次先后；用户消息与回复各截到 200 个字符"
    )

    @classmethod
    def from_domain(cls, snapshot: TopicSnapshot) -> ActiveTopicResponse:
        """从不包含 Workspace 归属信息的安全字段构造 HTTP 投影。"""

        return cls(
            topic_id=snapshot.topic_id,
            topic_title=snapshot.topic_title,
            topic_summary=snapshot.topic_summary,
            fold_summary=snapshot.fold_summary,
            last_turn=TopicLastTurnResponse.from_domain(snapshot.last_turn),
            block_count=snapshot.block_count,
            total_tokens=snapshot.total_tokens,
            model_used=snapshot.model_used,
            last_update=snapshot.last_update,
            has_unsettled_interactions=snapshot.has_unsettled_interactions,
            progress_record=TopicProgressEntryResponse.from_record(snapshot.progress_record),
            unrecorded_turn_count=snapshot.unrecorded_turn_count,
            unrecorded_turns=[
                TopicTurnExcerptResponse.from_domain(turn) for turn in snapshot.unrecorded_turns
            ],
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
    "TopicProgressEntryResponse",
    "TopicSettleResponse",
    "TopicTurnExcerptResponse",
]
