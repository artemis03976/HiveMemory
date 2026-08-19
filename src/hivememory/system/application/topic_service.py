from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

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
        access_context = resolve_default_workspace_access(
            identity,
            f"topic_list_{uuid4().hex}",
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
            access_context=access_context,
        )

    async def settle_topic(self, *, topic_id: str | None = None) -> dict:
        from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
        task: MemoryGenerationTask | None = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC,
            topic_id=topic_id,
        )
        if task is None:
            return {"success": False, "message": "话题为空，无需生成"}
        return {"success": True, "task_id": task.task_id, "topic_id": task.topic_id}

    async def evict_topic(self, *, topic_id: str) -> dict:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_EVICT_TOPIC,
            topic_id=topic_id,
        )
