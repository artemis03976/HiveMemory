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
from hivememory.engines.retrieval.policy import memory_belongs_to_workspace, memory_is_readable
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

        patch 值按字段赋值到读取出的领域对象，由模型的赋值校验拒绝非法值，再按
        ``meta.lifecycle`` / ``meta.access_policy`` 两个嵌套键提交 Qdrant 局部
        payload 更新，不重算向量、不写版本 Artifact。
        """
        if not patch:
            raise ValueError("patch_payload 不允许空 patch")
        unknown = set(patch) - self._PATCH_ALLOWED_PATHS
        if unknown:
            raise ValueError(f"patch_payload 不允许的字段路径: {sorted(unknown)}")

        atom = await self._store.get_memory(key)
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
        atom = await self.get_by_key(
            WorkspaceMemoryKey.from_identity_scope(identity_scope, memory_id)
        )
        if atom is None or not _readable(atom, identity_scope, enforce_actor_visibility):
            return None
        return atom

    async def get_by_alias(
        self,
        identity_scope: IdentityScope,
        alias: str,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        atom = await self._store.get_memory_by_alias(
            alias,
            query_filter=self._filter_converter.convert(QueryFilters(), identity_scope),
        )
        if atom is None or not _readable(atom, identity_scope, enforce_actor_visibility):
            return None
        return atom

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        """内部可信路径按复合键读取：只校验 ownership，不做 actor 可见性过滤。"""
        atom = await self._store.get_memory(key)
        if atom is None or not memory_belongs_to_workspace(atom, key.workspace_identity):
            return None
        return atom

    async def delete(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> bool:
        key = WorkspaceMemoryKey.from_identity_scope(identity_scope, memory_id)
        if await self.get_by_key(key) is None:
            return False
        return await self.delete_by_key(key)

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        return await self._store.delete_memory(key)

    async def search(
        self,
        identity_scope: IdentityScope,
        query: str,
        top_k: int,
        filters: QueryFilters | None = None,
        mode: str = "dense",
        score_threshold: float = 0.0,
        *,
        enforce_actor_visibility: bool = True,
    ) -> list[dict[str, Any]]:
        hits = await self._store.search_memories(
            query_text=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=self._filter_converter.convert(filters or QueryFilters(), identity_scope),
            mode=mode,
        )
        # 存储预过滤不是授权事实；命中返回前仍以 canonical Memory 重验策略。
        return [
            hit
            for hit in hits
            if _readable(hit["memory"], identity_scope, enforce_actor_visibility)
        ]

    async def scroll(
        self,
        identity_scope: IdentityScope,
        filters: QueryFilters | None = None,
        limit: int = 100,
        *,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]:
        memories = await self._store.get_all_memories(
            filters=self._filter_converter.convert(filters or QueryFilters(), identity_scope),
            limit=limit,
        )
        return [
            memory
            for memory in memories
            if _readable(memory, identity_scope, enforce_actor_visibility)
        ]

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


def _readable(
    memory: MemoryAtom,
    identity_scope: IdentityScope,
    enforce_actor_visibility: bool,
) -> bool:
    """Workspace ownership 硬边界 + actor 读取策略（中期存储唯一的授权重验点）。"""
    return memory_is_readable(
        memory,
        workspace_identity=identity_scope.workspace_identity,
        actor_identity=identity_scope.actor_identity,
        enforce_actor_visibility=enforce_actor_visibility,
    )


__all__ = ["QdrantStorageAdapter"]
