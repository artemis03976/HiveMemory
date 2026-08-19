from __future__ import annotations

from typing import Any

from hivememory.core.models import (
    AgentProfile,
    MemoryAtom,
    MemoryType,
    WorkspaceAccessContext,
    require_workspace_access_context,
)
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class AgentProfileManagementService:
    """Patchouli application service for public agent profile management APIs."""

    def __init__(self, *, bus: Any) -> None:
        self._bus = bus

    async def create_agent_profile(
        self,
        access_context: WorkspaceAccessContext,
        atom: MemoryAtom,
    ) -> MemoryAtom:
        access_context = require_workspace_access_context(access_context)
        if atom.workspace_identity != access_context.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        atom.index.memory_type = MemoryType.AGENT_PROFILE
        await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            access_context,
            atom,
        )
        return atom

    async def list_agent_profiles(
        self,
        *,
        access_context: WorkspaceAccessContext,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            access_context=require_workspace_access_context(access_context),
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        access_context: WorkspaceAccessContext,
    ) -> AgentProfile:
        return await self._bus.request(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            agent_alias,
            access_context=access_context,
        )
