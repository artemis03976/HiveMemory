from __future__ import annotations

from typing import Any

from hivememory.core.models import Identity


class TopicManagementService:
    """Patchouli application service for public topic management APIs."""

    def __init__(self, *, librarian_core: Any) -> None:
        self._librarian_core = librarian_core

    async def list_active_topics(self, *, identity: Identity):
        return self._librarian_core.get_active_topics_snapshots(identity)

    async def archive_topic(self, *, topic_id: str | None = None) -> dict[str, Any]:
        return await self._librarian_core.manual_archive_topic(topic_id)

    async def evict_topic(self, *, topic_id: str) -> dict[str, Any]:
        buf = self._librarian_core.perception_layer.buffer_manager.pop_buffer(topic_id)
        if buf is None:
            return {"success": False, "message": "话题不存在或已被驱逐"}
        return {"success": True, "message": f"话题 {topic_id} 已删除"}
