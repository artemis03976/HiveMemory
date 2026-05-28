from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        librarian_core: Any | None = None,
    ) -> None:
        self._global_bus = global_bus
        self._config = config
        self._librarian_core_backend = librarian_core

    @property
    def config(self) -> "HiveMemoryConfig":
        return self._config

    def bind_librarian_core(self, librarian_core: Any) -> None:
        self._librarian_core_backend = librarian_core

    @property
    def _librarian_core(self):
        if self._librarian_core_backend is None:
            raise RuntimeError("LibrarianCore is not bound to TopicApplicationService")
        return self._librarian_core_backend

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
