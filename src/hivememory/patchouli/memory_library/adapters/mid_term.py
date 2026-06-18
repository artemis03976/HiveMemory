"""
QdrantStorageAdapter — MidTermStoragePort 的 Qdrant 实现

包装现有 QdrantMemoryStore，使中期存储操作通过 Port 接口发起。

版本: 0.1.0 (Phase 1)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom
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
    ) -> List[Dict[str, Any]]:
        return await self._store.search_memories(
            query_text=query,
            top_k=top_k,
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


__all__ = ["QdrantStorageAdapter"]
