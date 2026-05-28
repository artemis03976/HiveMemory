from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import Identity
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class TopicApplicationService:
    """Topic API use-case service.

    Phase 1 only establishes the dependency boundary. Router behavior will be
    migrated into this service in later phases.
    """

    def __init__(
        self,
        global_bus: "GlobalSystemBus",
        config: "HiveMemoryConfig",
    ) -> None:
        self._global_bus = global_bus
        self._config = config

    @property
    def config(self) -> "HiveMemoryConfig":
        return self._config

    async def list_active_topics(self, *, user_id: str):
        identity = Identity(user_id=user_id)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            identity=identity,
        )

    async def archive_topic(self, *, topic_id: str | None = None) -> dict:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MANUAL_ARCHIVE_TOPIC,
            topic_id=topic_id,
        )

    async def evict_topic(self, *, topic_id: str) -> dict:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_EVICT_TOPIC,
            topic_id=topic_id,
        )
