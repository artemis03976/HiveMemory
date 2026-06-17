from __future__ import annotations

from typing import Any

from hivememory.core.models import MemoryAtom


class AgentProfileManagementService:
    """Patchouli application service for public agent profile management APIs."""

    def __init__(self, *, storage: Any) -> None:
        self._storage = storage

    async def create_agent_profile(self, atom: MemoryAtom) -> MemoryAtom:
        await self._storage.upsert_memory(atom)
        return atom

    async def list_agent_profiles(self, *, limit: int = 100) -> list[MemoryAtom]:
        return await self._storage.get_all_memories(
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )
