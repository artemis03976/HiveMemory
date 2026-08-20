from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import Identity, resolve_default_workspace_access
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
        access_context = resolve_default_workspace_access(identity)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            access_context=access_context,
        )

    async def settle_topic(self, *, user_id: str, topic_id: str | None = None) -> dict:
        from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
        access_context = resolve_default_workspace_access(
            Identity(user_id=user_id),
        )
        task: MemoryGenerationTask | None = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC,
            access_context=access_context,
            topic_id=topic_id,
        )
        if task is None:
            return {"success": False, "message": "话题为空，无需生成"}
        return {"success": True, "task_id": task.task_id, "topic_id": task.topic_id}

    async def evict_topic(self, *, user_id: str, topic_id: str) -> dict:
        access_context = resolve_default_workspace_access(
            Identity(user_id=user_id),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_EVICT_TOPIC,
            access_context=access_context,
            topic_id=topic_id,
        )
