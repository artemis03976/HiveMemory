from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import Identity

if TYPE_CHECKING:
    from hivememory.patchouli.services.perception import PerceptionFamiliar
    from hivememory.patchouli.services.retrieval import RetrievalFamiliar


class TopicManagementService:
    """Patchouli public topic management application service."""

    def __init__(
        self,
        *,
        perception_familiar: "PerceptionFamiliar",
        retrieval_familiar: "RetrievalFamiliar | None" = None,
    ) -> None:
        self._perception_familiar = perception_familiar
        self._retrieval_familiar = retrieval_familiar

    async def list_active_topics(self, *, identity: Identity):
        if self._retrieval_familiar is None:
            return []
        return self._retrieval_familiar.list_active_topics(identity)

    async def archive_topic(self, *, topic_id: str | None = None) -> dict:
        return await self._perception_familiar.manual_archive_topic(topic_id)

    async def evict_topic(self, *, topic_id: str) -> dict:
        return await self._perception_familiar.evict_topic(topic_id)

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity: Identity,
    ) -> str:
        return await self._perception_familiar.prepare_topic(
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity,
        )
