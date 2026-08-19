from __future__ import annotations

from uuid import UUID

from hivememory.core.models import MemoryAtom, MemoryType, WorkspaceAccessContext
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

    async def create_memory(self, atom: MemoryAtom) -> MemoryAtom:
        return await self._bus.request(PatchouliLocalRoutes.MEMORY_CREATE, atom)

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
        atoms = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_LIST,
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
        refresh_vitality: bool = True,
    ) -> MemoryAtom | None:
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            normalize_uuid(memory_id),
        )
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
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_UPDATE,
            normalize_uuid(memory_id),
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )

    async def delete_memory(self, memory_id: UUID | str) -> bool:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_DELETE,
            normalize_uuid(memory_id),
        )

    async def record_feedback(
        self,
        memory_id: UUID | str,
        *,
        positive: bool,
        source: str,
    ):
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
            normalize_uuid(memory_id),
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
        access_context: WorkspaceAccessContext,
    ) -> RetrievalResponse:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            aliases,
            access_context,
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
