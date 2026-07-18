"""Gateway 两阶段上下文准备契约。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import Identity, TopicData, TopicSnapshot


class CandidateTopics(BaseModel):
    """话题路由前可见的候选话题及其提示词菜单。"""

    topic_snapshots: tuple[TopicSnapshot, ...] = Field(default_factory=tuple)
    active_topics_menu: str = ""

    model_config = ConfigDict(frozen=True)


class GatewayContextProvider(Protocol):
    """只负责读取 Gateway 所需上下文，不参与业务决策。"""

    async def prepare_candidate_topics(
        self,
        *,
        identity: Identity,
    ) -> CandidateTopics: ...

    async def prepare_routed_topic(
        self,
        *,
        identity: Identity,
        topic_id: str,
    ) -> TopicData | None: ...


__all__ = ["CandidateTopics", "GatewayContextProvider"]
