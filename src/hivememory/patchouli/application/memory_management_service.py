from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from hivememory.core.models import MemoryAtom, MemoryType


class MemoryManagementService:
    """Patchouli application service for public memory management APIs."""

    def __init__(
        self,
        *,
        storage: Any,
        lifecycle_engine: Any | None = None,
    ) -> None:
        self._storage = storage
        self._lifecycle_engine = lifecycle_engine

    async def create_memory(self, atom: MemoryAtom) -> MemoryAtom:
        await self._storage.upsert_memory(atom)
        return atom

    async def list_memories(
        self,
        *,
        query: str | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 20,
        exclude_types: list[str] | None = None,
        refresh_vitality: bool = True,
    ) -> list[MemoryAtom]:
        excluded = set(exclude_types or [])
        if query:
            results = await self._storage.search_memories(
                query_text=query,
                top_k=limit,
                filters=filters,
            )
            atoms = [
                result["memory"]
                for result in results
                if "memory" in result
            ]
        else:
            atoms = await self._storage.get_all_memories(
                filters=filters,
                limit=limit,
            )

        atoms = [
            atom for atom in atoms
            if self._memory_type_value(atom.index.memory_type) not in excluded
        ]
        if refresh_vitality:
            await self._refresh_vitality_for_response(atoms)
        return atoms

    async def get_memory(
        self,
        memory_id: UUID | str,
        *,
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        atom = await self._storage.get_memory(self._normalize_uuid(memory_id))
        if atom is not None and refresh_vitality:
            await self._refresh_vitality_for_response([atom])
        return atom

    async def update_memory(
        self,
        memory_id: UUID | str,
        *,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom | None:
        atom = await self._storage.get_memory(self._normalize_uuid(memory_id))
        if atom is None:
            return None

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

        await self._storage.upsert_memory(atom)
        return atom

    async def delete_memory(self, memory_id: UUID | str) -> bool:
        return await self._storage.delete_memory(self._normalize_uuid(memory_id))

    async def record_feedback(
        self,
        memory_id: UUID | str,
        *,
        positive: bool,
        source: str,
    ):
        if self._lifecycle_engine is None:
            raise RuntimeError("Memory lifecycle engine is unavailable")

        return await self._lifecycle_engine.record_feedback(
            self._normalize_uuid(memory_id),
            positive=positive,
            source=source,
        )

    @staticmethod
    def _normalize_uuid(memory_id: UUID | str) -> UUID:
        return memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))

    @staticmethod
    def _memory_type_value(memory_type: MemoryType | str) -> str:
        return memory_type.value if hasattr(memory_type, "value") else str(memory_type)

    async def _refresh_vitality_for_response(self, atoms: list[MemoryAtom]) -> None:
        if self._lifecycle_engine is None or not atoms:
            return
        try:
            await self._lifecycle_engine.refresh_vitality_batch(atoms, persist=False)
        except Exception:
            return
