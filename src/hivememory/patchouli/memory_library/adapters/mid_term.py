"""
QdrantStorageAdapter — MidTermStoragePort 的 Qdrant 实现

包装现有 QdrantMemoryStore，使中期存储操作通过 Port 接口发起。

实现阶段: Phase 1
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError

from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    WorkspaceMemoryKey,
)
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.engines.retrieval.policy import memory_is_readable
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import MidTermStoragePort

if TYPE_CHECKING:
    from hivememory.infrastructure.storage import QdrantMemoryStore


class QdrantStorageAdapter(MidTermStoragePort):
    """Qdrant 向量存储的 MidTermStoragePort 适配器。"""

    def __init__(
        self,
        store: QdrantMemoryStore,
        *,
        use_sparse: bool = True,
    ) -> None:
        self._store = store
        self._use_sparse = use_sparse
        self._filter_converter = QdrantFilterConverter()

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        """提交完整 canonical Memory；``recompute_vectors=False`` 保留既有向量。"""
        await self._store.upsert_memory(
            memory, use_sparse=self._use_sparse, recompute_vectors=recompute_vectors
        )

    # patch_payload 允许的 canonical dotted 字段白名单（A2-P §4.1 / MVL-0 M0.1）。
    _PATCH_ALLOWED_PATHS = frozenset(
        {
            "meta.lifecycle.access_count",
            "meta.lifecycle.last_accessed_at",
            "meta.lifecycle.event_vitality_boost",
            "meta.lifecycle.vitality_score",
            "meta.lifecycle.confidence_score",
            "meta.lifecycle.verification_status",
            "meta.lifecycle.decay_anchor_at",
            "meta.access_policy",
        }
    )

    async def patch_payload(
        self,
        key: WorkspaceMemoryKey,
        patch: Mapping[str, Any],
    ) -> MemoryAtom | None:
        """受限局部更新：只改白名单字段，保留向量与全部非目标字段。

        读取使用严格 schema（旧记录只读拒绝）；patch 值按字段赋值到领域对象，
        由模型的赋值校验拒绝非法值，再按 ``meta.lifecycle`` /
        ``meta.access_policy`` 两个嵌套键提交 Qdrant 局部 payload 更新，不重算
        向量、不写版本 Artifact。
        """
        if not patch:
            raise ValueError("patch_payload 不允许空 patch")
        unknown = set(patch) - self._PATCH_ALLOWED_PATHS
        if unknown:
            raise ValueError(f"patch_payload 不允许的字段路径: {sorted(unknown)}")

        atom = await self._store.get_memory(key, require_current_schema=True)
        if atom is None:
            return None

        updated = atom.model_copy(deep=True)
        try:
            for path, value in patch.items():
                if path == "meta.access_policy":
                    updated.meta.access_policy = value
                else:
                    setattr(updated.meta.lifecycle, path.rsplit(".", 1)[-1], value)
        except ValidationError as exc:
            raise ValueError(f"patch_payload 值未通过领域校验: {exc}") from exc

        patches_lifecycle = any(path != "meta.access_policy" for path in patch)
        await self._store.patch_memory_payload(
            key,
            lifecycle=(
                updated.meta.lifecycle.model_dump(mode="json") if patches_lifecycle else None
            ),
            access_policy=(
                updated.meta.access_policy.model_dump(mode="json")
                if "meta.access_policy" in patch
                else None
            ),
        )
        return updated

    async def get(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        key = WorkspaceMemoryKey(
            workspace_identity=identity_scope.workspace_identity,
            memory_id=memory_id,
        )
        atom = await self._store.get_memory(key)
        if atom is None:
            return None
        if not memory_is_readable(
            atom,
            workspace_identity=identity_scope.workspace_identity,
            actor_identity=identity_scope.actor_identity,
            enforce_actor_visibility=enforce_actor_visibility,
        ):
            return None
        return atom

    async def get_by_alias(
        self,
        identity_scope: IdentityScope,
        alias: str,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        query_filter = self._filter_converter.convert(QueryFilters(), identity_scope)
        atom = await self._store.get_memory_by_alias(
            alias,
            query_filter=query_filter,
            workspace_identity=identity_scope.workspace_identity,
        )
        if atom is None:
            return None
        if not memory_is_readable(
            atom,
            workspace_identity=identity_scope.workspace_identity,
            actor_identity=identity_scope.actor_identity,
            enforce_actor_visibility=enforce_actor_visibility,
        ):
            return None
        return atom

    async def get_for_mutation(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> MemoryAtom | None:
        """mutation 入口只接受当前 schema：兼容窗口内旧记录只读，拒绝读出后回写。"""
        key = WorkspaceMemoryKey.from_identity_scope(identity_scope, memory_id)
        return await self._store.get_memory(key, require_current_schema=True)

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        return await self._store.get_memory(key)

    async def delete(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> bool:
        atom = await self.get_for_mutation(identity_scope, memory_id)
        if atom is None:
            return False
        return await self.delete_by_key(
            WorkspaceMemoryKey.from_identity_scope(identity_scope, memory_id)
        )

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        return await self._store.delete_memory(key)

    async def batch_delete(
        self,
        identity_scope: IdentityScope,
        ids: list[UUID],
    ) -> int:
        existing = [
            memory_id
            for memory_id in ids
            if await self.get_for_mutation(identity_scope, memory_id) is not None
        ]
        keys = [
            WorkspaceMemoryKey.from_identity_scope(identity_scope, memory_id)
            for memory_id in existing
        ]
        return await self._store.batch_delete_memories(keys)

    async def search(
        self,
        identity_scope: IdentityScope,
        query: str,
        top_k: int,
        filters: QueryFilters | None = None,
        mode: str = "dense",
        *,
        enforce_actor_visibility: bool = True,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        query_filter = self._filter_converter.convert(filters or QueryFilters(), identity_scope)
        hits = await self._store.search_memories(
            query_text=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=query_filter,
            mode=mode,
            workspace_identity=identity_scope.workspace_identity,
        )
        # 存储预过滤不是授权事实；命中返回前仍以 canonical Memory 重验策略。
        return [
            hit
            for hit in hits
            if memory_is_readable(
                hit["memory"],
                workspace_identity=identity_scope.workspace_identity,
                actor_identity=identity_scope.actor_identity,
                enforce_actor_visibility=enforce_actor_visibility,
            )
        ]

    async def scroll(
        self,
        identity_scope: IdentityScope,
        filters: QueryFilters | None = None,
        limit: int = 100,
        *,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]:
        query_filter = self._filter_converter.convert(filters or QueryFilters(), identity_scope)
        memories = await self._store.get_all_memories(
            filters=query_filter,
            workspace_identity=identity_scope.workspace_identity,
            limit=limit,
        )
        return [
            memory
            for memory in memories
            if memory_is_readable(
                memory,
                workspace_identity=identity_scope.workspace_identity,
                actor_identity=identity_scope.actor_identity,
                enforce_actor_visibility=enforce_actor_visibility,
            )
        ]

    async def count(
        self,
        identity_scope: IdentityScope,
        filters: QueryFilters | None = None,
    ) -> int:
        query_filter = self._filter_converter.convert(filters or QueryFilters(), identity_scope)
        return await self._store.count_memories(query_filter)

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
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
