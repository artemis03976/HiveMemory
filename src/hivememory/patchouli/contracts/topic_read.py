"""Patchouli 话题只读公共协议。"""

from __future__ import annotations

from typing import Protocol

from hivememory.core.models import TopicData, TopicSnapshot, WorkspaceAccessContext


class TopicReadPublicApi(Protocol):
    """Gateway 可依赖的最小话题读取表面。"""

    async def list_active_topics(
        self,
        *,
        access_context: WorkspaceAccessContext,
        include_empty: bool = False,
    ) -> tuple[TopicSnapshot, ...]: ...

    async def get_topic_data(
        self,
        *,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> TopicData | None: ...


__all__ = ["TopicReadPublicApi"]
