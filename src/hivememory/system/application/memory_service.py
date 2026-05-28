from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from hivememory.core.models import (
    Artifacts,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class MemoryLifecycleUnavailableError(RuntimeError):
    """Raised when lifecycle feedback operations are unavailable."""


class MemoryNotFoundError(ValueError):
    """Raised when the requested memory does not exist."""


class MemoryApplicationService:
    """Memory API use-case service.

    HTTP routers call this service instead of reaching into Patchouli internals.
    This phase preserves the existing storage behavior while narrowing the
    server-facing dependency boundary.
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

    async def create_memory(
        self,
        *,
        title: str,
        summary: str,
        content: str,
        memory_type: str,
        tags: list[str],
        alias: str | None = None,
    ) -> MemoryAtom:
        atom = MemoryAtom(
            meta=MetaData(source_agent_id="ui", user_id="default"),
            index=IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType(memory_type),
                alias=alias,
            ),
            payload=PayloadLayer(
                content=content,
                artifacts=Artifacts(),
            ),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_CREATE,
            atom,
        )

    async def list_memories(
        self,
        *,
        query: str | None = None,
        user_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryAtom]:
        filters = self._build_filters(user_id=user_id, memory_type=memory_type)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_LIST,
            query=query,
            filters=filters if filters else None,
            limit=limit,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
            refresh_vitality=True,
        )

    async def get_memory(self, memory_id: UUID) -> MemoryAtom:
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            memory_id,
            refresh_vitality=True,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def update_memory(
        self,
        memory_id: UUID,
        *,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom:
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_UPDATE,
            memory_id,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def record_feedback(
        self,
        memory_id: UUID,
        *,
        positive: bool,
        source: str,
    ):
        try:
            return await self._global_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RECORD_FEEDBACK,
                memory_id,
                positive=positive,
                source=source,
            )
        except RuntimeError as exc:
            raise MemoryLifecycleUnavailableError(
                "Memory lifecycle engine is unavailable"
            ) from exc
        except ValueError as exc:
            raise MemoryNotFoundError(str(exc)) from exc

    async def delete_memory(self, memory_id: UUID) -> bool:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_DELETE,
            memory_id,
        )

    @staticmethod
    def _build_filters(
        *,
        user_id: str | None,
        memory_type: str | None,
    ) -> dict[str, str]:
        filters = {}
        if user_id:
            filters["meta.user_id"] = user_id
        if memory_type:
            filters["index.memory_type"] = memory_type
        return filters
