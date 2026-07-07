"""
Phase 3 Gateway Context Hydration 骨架。

本阶段只固定上下文构造边界，不接入 active chat 主路径。
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from hivememory.core.models import Identity, TopicSnapshot


class TopicSnapshotProvider(Protocol):
    """Gateway Context Hydration 读取活跃话题的抽象边界。"""

    async def list_active_topics(
        self,
        *,
        identity: Identity,
        include_empty: bool = False,
    ) -> list[TopicSnapshot]:
        ...


class SessionContext(BaseModel):
    """Gateway Pipeline 的只读会话上下文输入。"""

    identity: Identity = Field(default_factory=Identity)
    topic_snapshots: list[TopicSnapshot] = Field(default_factory=list)


class GatewayContextBuilder:
    """Phase 3 Context Hydration 构造器骨架。"""

    def __init__(self, topic_provider: TopicSnapshotProvider | None = None) -> None:
        self._topic_provider = topic_provider

    async def build(
        self,
        *,
        message: str,
        identity: Identity,
    ) -> SessionContext:
        """
        构造 Gateway Pipeline 输入上下文。

        message 参数保留给后续按消息裁剪上下文的策略；Phase 3A 不使用。
        """

        _ = message
        topic_snapshots: list[TopicSnapshot] = []
        if self._topic_provider is not None:
            topic_snapshots = await self._topic_provider.list_active_topics(
                identity=identity,
            )
        return SessionContext(identity=identity, topic_snapshots=topic_snapshots)


__all__ = ["GatewayContextBuilder", "SessionContext", "TopicSnapshotProvider"]
