"""
QdrantStorageAdapter — MidTermStoragePort 的 Qdrant 实现

包装现有 QdrantMemoryStore，使中期存储操作通过 Port 接口发起。

实现阶段: Phase 1
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import (
    MemoryAtom,
    MemoryReadScope,
    WorkspaceAccessContext,
    WorkspaceMemoryKey,
)
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.engines.retrieval.policy import memory_is_readable
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import MidTermStoragePort

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hivememory.infrastructure.storage import QdrantMemoryStore


class QdrantStorageAdapter(MidTermStoragePort):
    """Qdrant 向量存储的 MidTermStoragePort 适配器。"""

    def __init__(
        self,
        store: "QdrantMemoryStore",
        *,
        use_sparse: bool = True,
    ) -> None:
        self._store = store
        self._use_sparse = use_sparse
        self._filter_converter = QdrantFilterConverter()

    async def upsert(self, memory: MemoryAtom) -> None:
        await self._store.upsert_memory(memory, use_sparse=self._use_sparse)

    async def get(
        self,
        scope: MemoryReadScope,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]:
        key = WorkspaceMemoryKey(
            workspace_identity=scope.workspace_identity,
            memory_id=memory_id,
        )
        atom = await self._store.get_memory(key)
        if atom is None:
            return None
        if not memory_is_readable(
            atom,
            workspace_identity=scope.workspace_identity,
            actor_identity=scope.actor_identity,
        ):
            return None
        return atom

    async def get_by_alias(
        self,
        scope: MemoryReadScope,
        alias: str,
    ) -> Optional[MemoryAtom]:
        query_filter = self._filter_converter.convert(QueryFilters(), scope)
        atom = await self._store.get_memory_by_alias(
            alias,
            query_filter=query_filter,
            workspace_identity=scope.workspace_identity,
        )
        if atom is None:
            return None
        if not memory_is_readable(
            atom,
            workspace_identity=scope.workspace_identity,
            actor_identity=scope.actor_identity,
        ):
            return None
        return atom

    async def get_for_mutation(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]:
        return await self.get_by_key(
            WorkspaceMemoryKey.from_access_context(access_context, memory_id)
        )

    async def get_by_key(self, key: WorkspaceMemoryKey) -> Optional[MemoryAtom]:
        return await self._store.get_memory(key)

    async def update_access_info(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> None:
        atom = await self.get(access_context, memory_id)
        if atom is None:
            return
        atom.meta.access_count += 1
        atom.meta.last_accessed_at = datetime.now()
        await self.upsert(atom)

    async def delete(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> bool:
        atom = await self.get_for_mutation(access_context, memory_id)
        if atom is None:
            return False
        return await self.delete_by_key(
            WorkspaceMemoryKey.from_access_context(access_context, memory_id)
        )

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        return await self._store.delete_memory(key)

    async def batch_delete(
        self,
        access_context: WorkspaceAccessContext,
        ids: List[UUID],
    ) -> int:
        existing = [
            memory_id
            for memory_id in ids
            if await self.get_for_mutation(access_context, memory_id) is not None
        ]
        keys = [
            WorkspaceMemoryKey.from_access_context(access_context, memory_id)
            for memory_id in existing
        ]
        return await self._store.batch_delete_memories(keys)

    async def search(
        self,
        scope: MemoryReadScope,
        query: str,
        top_k: int,
        filters: Optional[QueryFilters] = None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        query_filter = self._filter_converter.convert(filters or QueryFilters(), scope)
        hits = await self._store.search_memories(
            query_text=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=query_filter,
            mode=mode,
            workspace_identity=scope.workspace_identity,
        )
        # 存储预过滤不是授权事实；命中返回前仍以 canonical Memory 重验策略。
        return [
            hit
            for hit in hits
            if memory_is_readable(
                hit["memory"],
                workspace_identity=scope.workspace_identity,
                actor_identity=scope.actor_identity,
            )
        ]

    async def scroll(
        self,
        scope: MemoryReadScope,
        filters: Optional[QueryFilters] = None,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        query_filter = self._filter_converter.convert(filters or QueryFilters(), scope)
        memories = await self._store.get_all_memories(
            filters=query_filter,
            workspace_identity=scope.workspace_identity,
            limit=limit,
        )
        return [
            memory
            for memory in memories
            if memory_is_readable(
                memory,
                workspace_identity=scope.workspace_identity,
                actor_identity=scope.actor_identity,
            )
        ]

    async def count(
        self,
        scope: MemoryReadScope,
        filters: Optional[QueryFilters] = None,
    ) -> int:
        query_filter = self._filter_converter.convert(filters or QueryFilters(), scope)
        return await self._store.count_memories(query_filter)

    async def list_all_for_maintenance(self, limit: int = 10000) -> List[MemoryAtom]:
        return await self._store.get_all_memories_for_maintenance(limit=limit)

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
