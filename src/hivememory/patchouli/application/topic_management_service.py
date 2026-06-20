from __future__ import annotations

from typing import Any

from hivememory.core.models import Identity


class TopicManagementService:
    """Patchouli application service for public topic management APIs."""

    def __init__(self, *, librarian_core: Any, retrieval_familiar: Any | None = None) -> None:
        self._librarian_core = librarian_core
        self._retrieval_familiar = retrieval_familiar

    async def list_active_topics(self, *, identity: Identity):
        if self._retrieval_familiar is None:
            return []
        return self._retrieval_familiar.list_active_topics(identity)

    async def archive_topic(self, *, topic_id: str | None = None) -> dict[str, Any]:
        return await self._librarian_core.manual_archive_topic(topic_id)

    async def evict_topic(self, *, topic_id: str) -> dict[str, Any]:
        removed = self._librarian_core.perception_layer.swap_out_topic(topic_id)
        if not removed:
            return {"success": False, "message": "话题不存在或已被驱逐"}
        return {"success": True, "message": f"话题 {topic_id} 已删除"}
