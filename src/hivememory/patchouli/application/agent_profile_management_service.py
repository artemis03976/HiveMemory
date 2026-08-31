from __future__ import annotations

from typing import Any

from hivememory.core.models import (
    AgentProfile,
    MemoryAtom,
    MemoryType,
    IdentityScope,
    require_identity_scope,
)
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class AgentProfileManagementService:
    """Patchouli application service for public agent profile management APIs."""

    def __init__(self, *, bus: Any) -> None:
        self._bus = bus

    async def create_agent_profile(
        self,
        identity_scope: IdentityScope,
        atom: MemoryAtom,
    ) -> MemoryAtom:
        identity_scope = require_identity_scope(identity_scope)
        if atom.workspace_identity != identity_scope.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        atom.index.memory_type = MemoryType.AGENT_PROFILE
        await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            identity_scope,
            atom,
        )
        return atom

    async def list_agent_profiles(
        self,
        *,
        identity_scope: IdentityScope,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            identity_scope=require_identity_scope(identity_scope),
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope,
    ) -> AgentProfile:
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            agent_alias,
            identity_scope=identity_scope,
        )
