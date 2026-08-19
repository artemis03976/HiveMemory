"""
MemoryLibrary 三层存储 Port 接口定义

定义短期 / 中期 / 长期三层存储的抽象契约，供各层 Store 通过 Port 多态实现。
实现类不应感知其他层的存储，跨层操作由 MemoryLibrary 编排。

实现阶段: Phase 1 骨架
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import (
    MemoryAtom,
    MemoryReadScope,
    WorkspaceAccessContext,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
    WorkspaceTopicKey,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType, BaseArtifact
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.engines.lifecycle.models import ArchiveRecord

if TYPE_CHECKING:
    from hivememory.engines.retrieval.models import QueryFilters


# ============ ShortTermStoragePort ============

class ShortTermStoragePort(ABC):
    """
    短期存储 Port — WorkspaceTopicKey → SemanticBuffer 的键值映射。

    ShortTermMemoryStore exposes synchronous APIs to the perception layer, so the
    short-term port is synchronous as well. Async backends should hide their I/O
    boundary behind an adapter instead of leaking await points into the store.

    实现：
        InMemoryShortTermStorage（内存态，Phase 1）
        RedisShortTermStorage（future）
    """

    @abstractmethod
    def get(self, key: WorkspaceTopicKey) -> Optional[SemanticBuffer]: ...

    @abstractmethod
    def put(self, key: WorkspaceTopicKey, buffer: SemanticBuffer) -> None: ...

    @abstractmethod
    def pop(self, key: WorkspaceTopicKey) -> Optional[SemanticBuffer]: ...

    @abstractmethod
    def list_by_workspace(self, workspace: WorkspaceIdentity) -> List[SemanticBuffer]: ...

    @abstractmethod
    def list_all(self) -> List[SemanticBuffer]: ...

    @abstractmethod
    def count(self, workspace: WorkspaceIdentity) -> int: ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="short_term", healthy=True)


# ============ MidTermStoragePort ============

class MidTermStoragePort(ABC):
    """
    中期存储 Port — 以 MemoryAtom 为边界的向量库操作。

    实现：
        QdrantStorageAdapter（Phase 1）
        GraphStorageAdapter（future）
    """

    @abstractmethod
    async def upsert(self, memory: MemoryAtom) -> None: ...

    @abstractmethod
    async def get(self, scope: MemoryReadScope, memory_id: UUID) -> Optional[MemoryAtom]: ...

    @abstractmethod
    async def get_by_alias(
        self,
        scope: MemoryReadScope,
        alias: str,
    ) -> Optional[MemoryAtom]: ...

    @abstractmethod
    async def get_for_mutation(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]: ...

    @abstractmethod
    async def get_by_key(self, key: WorkspaceMemoryKey) -> Optional[MemoryAtom]: ...

    @abstractmethod
    async def update_access_info(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> None: ...

    @abstractmethod
    async def delete(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> bool: ...

    @abstractmethod
    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool: ...

    @abstractmethod
    async def batch_delete(
        self,
        access_context: WorkspaceAccessContext,
        ids: List[UUID],
    ) -> int: ...

    @abstractmethod
    async def search(
        self,
        scope: MemoryReadScope,
        query: str,
        top_k: int,
        filters: Optional["QueryFilters"] = None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def scroll(
        self,
        scope: MemoryReadScope,
        filters: Optional["QueryFilters"] = None,
        limit: int = 100,
    ) -> List[MemoryAtom]: ...

    @abstractmethod
    async def count(
        self,
        scope: MemoryReadScope,
        filters: Optional["QueryFilters"] = None,
    ) -> int: ...

    @abstractmethod
    async def list_all_for_maintenance(self, limit: int = 10000) -> List[MemoryAtom]: ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="mid_term", healthy=True)


# ============ LongTermStoragePort ============

class LongTermStoragePort(ABC):
    """
    长期存储 Port — 冷存储读写，不感知中期存储。

    跨层状态转移（archive / revive）由 MemoryLibrary 编排，不在此 Port 内实现。

    实现：
        FileBasedStorageAdapter（Phase 1）
        DBBasedStorageAdapter（future）
    """

    @abstractmethod
    async def persist(self, memory: MemoryAtom) -> None: ...

    @abstractmethod
    async def load(self, key: WorkspaceMemoryKey) -> MemoryAtom: ...

    @abstractmethod
    async def remove(self, key: WorkspaceMemoryKey) -> None: ...

    @abstractmethod
    async def is_archived(self, key: WorkspaceMemoryKey) -> bool: ...

    @abstractmethod
    async def query(
        self,
        limit: int = 100,
        vitality_threshold: Optional[float] = None,
    ) -> List[ArchiveRecord]: ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="long_term", healthy=True)


# ============ ArtifactStoragePort ============

class ArtifactStoragePort(ABC):
    """
    Artifact 记忆附属资产仓库 Port — append-only 持久化存储。

    实现：
        FilesystemArtifactStorageAdapter（Phase 2，从 infrastructure/storage 迁入）
        SQLArtifactStorageAdapter（future）
    """

    @abstractmethod
    async def put(self, artifact: BaseArtifact) -> ArtifactRef: ...

    @abstractmethod
    async def get(self, ref_or_id: "ArtifactRef | str") -> Dict[str, Any]: ...

    @abstractmethod
    async def exists(self, artifact_id: str) -> bool: ...

    @abstractmethod
    async def list_by_memory(
        self,
        memory_id: str,
        artifact_type: Optional[ArtifactType] = None,
    ) -> List[ArtifactRef]: ...

    @abstractmethod
    async def verify(self, ref: ArtifactRef) -> "ArtifactIntegrityResult": ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="artifact", healthy=True, required=False)


__all__ = [
    "ShortTermStoragePort",
    "MidTermStoragePort",
    "LongTermStoragePort",
    "ArtifactStoragePort",
]
