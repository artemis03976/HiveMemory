from __future__ import annotations

from typing import Any, TYPE_CHECKING

from hivememory.core.models import Identity

if TYPE_CHECKING:
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.patchouli.services.retrieval import RetrievalFamiliar


class TopicManagementService:
    """Patchouli application service for public topic management APIs."""

    def __init__(
        self,
        *,
        perception_layer: "BasePerceptionLayer",
        retrieval_familiar: "RetrievalFamiliar | None" = None,
    ) -> None:
        self._perception_layer = perception_layer
        self._retrieval_familiar = retrieval_familiar

    async def list_active_topics(self, *, identity: Identity):
        if self._retrieval_familiar is None:
            return []
        return self._retrieval_familiar.list_active_topics(identity)

    async def archive_topic(self, *, topic_id: str | None = None) -> dict[str, Any]:
        return await self._perception_layer.manual_trigger(topic_id)

    async def evict_topic(self, *, topic_id: str) -> dict[str, Any]:
        removed = self._perception_layer.swap_out_topic(topic_id)
        if not removed:
            return {"success": False, "message": "话题不存在或已被驱逐"}
        return {"success": True, "message": f"话题 {topic_id} 已删除"}

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity: Identity,
    ) -> str:
        return await self._perception_layer.prepare_topic(
            target_topic_id, new_topic_title, new_topic_summary, identity
        )
