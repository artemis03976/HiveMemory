"""User Query Analysis 的稳定输入输出边界。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import ActorIdentity, TopicData
from hivememory.core.protocol.gateway import (
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.gateway.context import CandidateTopics


class UserQueryAnalysisContext(BaseModel):
    """路由完成后交给查询分析能力的只读上下文。"""

    raw_message: str
    identity: ActorIdentity
    candidate_topics: CandidateTopics
    topic_id: str
    new_topic_title: str | None = None
    new_topic_summary: str | None = None
    routed_topic_data: TopicData | None = None

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class UserQueryAnalysisResult(BaseModel):
    """一次完整且不可拆分提交的查询分析结果。"""

    intent_type: IntentType
    rewritten_query: str
    search_keywords: tuple[str, ...] = Field(default_factory=tuple)
    memory_write_signal: MemoryWriteSignal
    retrieval_plan: RetrievalPlan

    model_config = ConfigDict(frozen=True)


class UserQueryAnalysisResolver(Protocol):
    """查询分析能力的可替换私有边界。"""

    async def resolve(
        self,
        context: UserQueryAnalysisContext,
    ) -> UserQueryAnalysisResult: ...


__all__ = [
    "UserQueryAnalysisContext",
    "UserQueryAnalysisResolver",
    "UserQueryAnalysisResult",
]
