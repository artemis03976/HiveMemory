from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import Identity, TopicData, TopicSnapshot
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class TopicManagementService:
    """Patchouli public topic management application service."""

    def __init__(self, *, bus: PatchouliBus) -> None:
        # Topic public API 只通过 local bus 组合 topic primitives，不直接持有 familiar。
        self._bus = bus

    async def list_active_topics(
        self,
        *,
        identity: Identity,
        include_empty: bool = False,
    ) -> tuple[TopicSnapshot, ...]:
        kwargs = {"identity": identity}
        if include_empty:
            kwargs["include_empty"] = True
        snapshots = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            **kwargs,
        )
        return tuple(snapshots)

    async def get_topic_data(
        self,
        *,
        identity: Identity,
        topic_id: str,
    ) -> TopicData | None:
        """无副作用读取调用方可见的完整话题数据。"""

        topic_data = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_GET,
            topic_id,
            touch=False,
        )
        if topic_data is None or topic_data.user_id != identity.user_id:
            return None
        return topic_data

    async def settle_topic(self, *, topic_id: str | None = None) -> MemoryGenerationTask | None:
        return await self._bus.request(PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE, topic_id)

    async def evict_topic(self, *, topic_id: str) -> dict:
        return await self._bus.request(PatchouliLocalRoutes.TOPIC_EVICT, topic_id)

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity: Identity,
    ) -> str:
        return await self._bus.request(
            PatchouliLocalRoutes.TOPIC_PREPARE,
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity,
        )
