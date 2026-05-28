from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from hivememory.core.models import (
    Artifacts,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)

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
        storage: Any | None = None,
        lifecycle_engine: Any | None = None,
    ) -> None:
        self._global_bus = global_bus
        self._config = config
        self._storage_backend = storage
        self._lifecycle_engine = lifecycle_engine

    @property
    def config(self) -> "HiveMemoryConfig":
        return self._config

    def bind_storage(
        self,
        storage: Any,
        *,
        lifecycle_engine: Any | None = None,
    ) -> None:
        self._storage_backend = storage
        self._lifecycle_engine = lifecycle_engine

    @property
    def _storage(self):
        if self._storage_backend is None:
            raise RuntimeError("Storage is not bound to MemoryApplicationService")
        return self._storage_backend

    def create_memory(
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
        self._storage.upsert_memory(atom)
        return atom

    def list_memories(
        self,
        *,
        query: str | None = None,
        user_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryAtom]:
        filters = self._build_filters(user_id=user_id, memory_type=memory_type)

        if query:
            results = self._storage.search_memories(
                query_text=query,
                top_k=limit,
                filters=filters if filters else None,
            )
            atoms = [
                r["memory"]
                for r in results
                if "memory" in r and r["memory"].index.memory_type != MemoryType.AGENT_PROFILE
            ]
        else:
            atoms = self._storage.get_all_memories(
                filters=filters if filters else None,
                limit=limit,
            )
            atoms = [
                atom for atom in atoms
                if atom.index.memory_type != MemoryType.AGENT_PROFILE
            ]

        self._refresh_vitality_for_response(atoms)
        return atoms

    def get_memory(self, memory_id: UUID) -> MemoryAtom:
        atom = self._storage.get_memory(memory_id)
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        self._refresh_vitality_for_response([atom])
        return atom

    def update_memory(
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
        atom = self._storage.get_memory(memory_id)
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")

        if title is not None:
            atom.index.title = title
        if summary is not None:
            atom.index.summary = summary
        if content is not None:
            atom.payload.content = content
        if alias is not None:
            atom.index.alias = alias or None
        if tags is not None:
            atom.index.tags = tags
        if agent_config is not None:
            atom.payload.artifacts.agent_config = agent_config
        atom.meta.updated_at = datetime.now(timezone.utc)

        self._storage.upsert_memory(atom)
        return atom

    def record_feedback(
        self,
        memory_id: UUID,
        *,
        positive: bool,
        source: str,
    ):
        lifecycle = self._get_lifecycle_engine()
        if lifecycle is None:
            raise MemoryLifecycleUnavailableError(
                "Memory lifecycle engine is unavailable"
            )
        try:
            return lifecycle.record_feedback(
                memory_id,
                positive=positive,
                source=source,
            )
        except ValueError as exc:
            raise MemoryNotFoundError(str(exc)) from exc

    def delete_memory(self, memory_id: UUID) -> bool:
        return self._storage.delete_memory(memory_id)

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

    def _get_lifecycle_engine(self):
        return self._lifecycle_engine

    def _refresh_vitality_for_response(self, atoms: list[MemoryAtom]) -> None:
        lifecycle = self._get_lifecycle_engine()
        if lifecycle is None or not atoms:
            return
        try:
            lifecycle.refresh_vitality_batch(atoms, persist=False)
        except Exception:
            return
