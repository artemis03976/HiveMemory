from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from hivememory.core.models import Identity
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.patchouli.system import PatchouliSystem
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
        patchouli: Optional["PatchouliSystem"] = None,
    ) -> None:
        self._global_bus = global_bus
        self._config = config
        self._patchouli = patchouli

    @property
    def config(self) -> "HiveMemoryConfig":
        return self._config

    def bind_patchouli(self, patchouli: "PatchouliSystem") -> None:
        self._patchouli = patchouli

    @property
    def _librarian_core(self):
        if self._patchouli is None:
            raise RuntimeError("PatchouliSystem is not bound to TopicApplicationService")
        return self._patchouli.librarian_core

    def list_active_topics(self, *, user_id: str):
        identity = Identity(user_id=user_id)
        return self._librarian_core.get_active_topics_snapshots(identity)

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
