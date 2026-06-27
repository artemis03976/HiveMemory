from __future__ import annotations

from typing import Any

from hivememory.core.models import AgentProfile, MemoryAtom, MemoryType
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class AgentProfileManagementService:
    """Patchouli application service for public agent profile management APIs."""

    def __init__(self, *, bus: Any) -> None:
        self._bus = bus

    async def create_agent_profile(self, atom: MemoryAtom) -> MemoryAtom:
        atom.index.memory_type = MemoryType.AGENT_PROFILE
        await self._bus.request(PatchouliLocalRoutes.MEMORY_CREATE, atom)
        return atom

    async def list_agent_profiles(self, *, limit: int = 100) -> list[MemoryAtom]:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )

    async def get_agent_profile(self, agent_alias: str) -> AgentProfile:
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            agent_alias,
        )
