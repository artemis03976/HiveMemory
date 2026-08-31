from __future__ import annotations

from uuid import UUID

from hivememory.core.models import (
    MemoryAtom,
    MemoryType,
    IdentityScope,
    require_identity_scope,
)
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.utils.uuid import normalize_uuid


class MemoryManagementService:
    """Patchouli application service for public memory management APIs."""

    def __init__(
        self,
        *,
        bus,
    ) -> None:
        self._bus = bus

    async def create_memory(
        self,
        identity_scope: IdentityScope,
        atom: MemoryAtom,
    ) -> MemoryAtom:
        identity_scope = require_identity_scope(identity_scope)
        if atom.workspace_identity != identity_scope.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_CREATE,
            identity_scope,
            atom,
        )

    async def list_memories(
        self,
        *,
        identity_scope: IdentityScope,
        query: str | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 20,
        exclude_types: list[str] | None = None,
        refresh_vitality: bool = True,
    ) -> list[MemoryAtom]:
        excluded = set(exclude_types or [])
        atoms = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
            identity_scope=require_identity_scope(identity_scope),
            query=query,
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
        identity_scope: IdentityScope,
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            normalize_uuid(memory_id),
            identity_scope=require_identity_scope(identity_scope),
        )
        if atom is not None and refresh_vitality:
            await self._refresh_vitality_for_response([atom])
        return atom

    async def update_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom | None:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_UPDATE,
            normalize_uuid(memory_id),
            identity_scope=require_identity_scope(identity_scope),
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )

    async def delete_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
    ) -> bool:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_DELETE,
            require_identity_scope(identity_scope),
            normalize_uuid(memory_id),
        )

    async def record_feedback(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        positive: bool,
        source: str,
    ):
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
            normalize_uuid(memory_id),
            identity_scope=require_identity_scope(identity_scope),
            positive=positive,
            source=source,
        )

    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            request,
        )

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        identity_scope: IdentityScope,
    ) -> RetrievalResponse:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            aliases,
            identity_scope,
        )

    @staticmethod
    def _memory_type_value(memory_type: MemoryType | str) -> str:
        return memory_type.value if hasattr(memory_type, "value") else str(memory_type)

    async def _refresh_vitality_for_response(self, atoms: list[MemoryAtom]) -> None:
        if not atoms:
            return
        try:
            await self._bus.request(
                PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
                atoms,
                persist=False,
            )
        except Exception:
            return
