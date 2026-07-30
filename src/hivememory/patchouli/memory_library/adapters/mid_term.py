"""
QdrantStorageAdapter — MidTermStoragePort 的 Qdrant 实现

包装现有 QdrantMemoryStore，使中期存储操作通过 Port 接口发起。

实现阶段: Phase 1
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import MidTermStoragePort

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hivememory.infrastructure.storage import QdrantMemoryStore


class QdrantStorageAdapter(MidTermStoragePort):
    """Qdrant 向量存储的 MidTermStoragePort 适配器。"""

    def __init__(self, store: "QdrantMemoryStore") -> None:
        self._store = store

    async def upsert(self, memory: MemoryAtom) -> None:
        await self._store.upsert_memory(memory)

    async def get(self, memory_id: UUID) -> Optional[MemoryAtom]:
        return await self._store.get_memory(memory_id)

    async def get_by_alias(self, alias: str, user_id: Optional[str] = None) -> Optional[MemoryAtom]:
        return await self._store.get_memory_by_alias(alias, user_id)

    async def update_access_info(self, memory_id: UUID) -> None:
        await self._store.update_access_info(memory_id)

    async def delete(self, memory_id: UUID) -> bool:
        return await self._store.delete_memory(memory_id)

    async def batch_delete(self, ids: List[UUID]) -> int:
        return await self._store.batch_delete_memories(ids)

    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return await self._store.search_memories(
            query_text=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            mode=mode,
        )

    async def scroll(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        return await self._store.get_all_memories(filters=filters, limit=limit)

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        return await self._store.count_memories(filters)

    async def check_health(self) -> StorageHealthComponent:
        try:
            await self._store.client.get_collections()
            return StorageHealthComponent(name="mid_term", healthy=True)
        except Exception as exc:
            return StorageHealthComponent(
                name="mid_term",
                healthy=False,
                detail=str(exc),
            )


__all__ = ["QdrantStorageAdapter"]
