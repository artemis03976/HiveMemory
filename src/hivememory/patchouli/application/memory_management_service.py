from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from hivememory.core.models import MemoryAtom, MemoryType
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class MemoryManagementService:
    """Patchouli application service for public memory management APIs."""

    def __init__(
        self,
        *,
        bus,
    ) -> None:
        self._bus = bus

    async def create_memory(self, atom: MemoryAtom) -> MemoryAtom:
        await self._bus.request(PatchouliLocalRoutes.MEMORY_CREATE, atom)
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
            self._normalize_uuid(memory_id),
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
        atom = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_GET,
            self._normalize_uuid(memory_id),
        )
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

        await self._bus.request(PatchouliLocalRoutes.MEMORY_UPDATE, atom)
        return atom

    async def delete_memory(self, memory_id: UUID | str) -> bool:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_DELETE,
            self._normalize_uuid(memory_id),
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
            self._normalize_uuid(memory_id),
            positive=positive,
            source=source,
        )

    async def retrieve(
        self,
        request: RetrievalRequest,
        mode: str = "active",
    ) -> RetrievalResponse:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            request,
        )

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        identity=None,
        mode: str = "active",
    ) -> RetrievalResponse:
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            aliases,
            identity,
        )

    @staticmethod
    def _normalize_uuid(memory_id: UUID | str) -> UUID:
        return memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))

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
