"""
MemoryLibrary 三层存储 Port 接口定义

定义短期 / 中期 / 长期三层存储的抽象契约，供各层 Store 通过 Port 多态实现。
实现类不应感知其他层的存储，跨层操作由 MemoryLibrary 编排。

实现阶段: Phase 1 骨架
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    TopicData,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType, BaseArtifact
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)

if TYPE_CHECKING:
    from hivememory.engines.retrieval.models import QueryFilters


# ============ ShortTermStoragePort ============


class ShortTermStoragePort(ABC):
    """
    短期存储 Port。

    Port 的稳定契约使用 WorkspaceIdentity + topic_id 和不可变 TopicData 快照；
    复合物理键 WorkspaceTopicKey 只属于具体 adapter 的内部实现。

    ShortTermMemoryStore exposes synchronous APIs to the perception layer, so the
    short-term port is synchronous as well. Async backends should hide their I/O
    boundary behind an adapter instead of leaking await points into the store.

    实现：
        InMemoryShortTermStorage（内存态，Phase 1）
        RedisShortTermStorage（future）
    """

    @abstractmethod
    def get(
        self,
        workspace: WorkspaceIdentity,
        topic_id: str,
    ) -> TopicData | None: ...

    @abstractmethod
    def put(self, topic: TopicData) -> None: ...

    @abstractmethod
    def delete(self, workspace: WorkspaceIdentity, topic_id: str) -> bool: ...

    @abstractmethod
    def list_by_workspace(self, workspace: WorkspaceIdentity) -> list[TopicData]: ...

    @abstractmethod
    def list_all(self) -> list[TopicData]: ...

    @abstractmethod
    def count(self, workspace: WorkspaceIdentity) -> int: ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="short_term", healthy=True)


# ============ MidTermStoragePort ============


class MidTermStoragePort(ABC):
    """
    中期存储 Port — 以 MemoryAtom 为边界的向量库操作。

    授权重验在 Port 实现内完成，存储预过滤不是授权事实：
        - 带 ``IdentityScope`` 的读取（get/get_by_alias/search/scroll）校验
          Workspace ownership 与 actor 读取策略；``enforce_actor_visibility=False``
          仅供管理读取跳过 actor 策略，ownership 仍然生效；
        - 按 ``WorkspaceMemoryKey`` 的读取与删除是内部可信路径（编辑、强化、
          归档），只校验 ownership；
        - 存储失败以 ``StorageOfflineError`` / ``StorageReadError`` /
          ``StorageWriteError`` 传播，不以空结果或 ``False`` 掩盖。

    实现：
        QdrantStorageAdapter（Phase 1）
        GraphStorageAdapter（future）
    """

    @abstractmethod
    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        """提交完整 canonical Memory；``recompute_vectors=False`` 时保留既有向量。"""

    @abstractmethod
    async def get(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None: ...

    @abstractmethod
    async def get_by_alias(
        self,
        identity_scope: IdentityScope,
        alias: str,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None: ...

    @abstractmethod
    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None: ...

    @abstractmethod
    async def patch_payload(
        self,
        key: WorkspaceMemoryKey,
        patch: Mapping[str, Any],
    ) -> MemoryAtom | None:
        """原子地更新允许的持久化字段并返回更新后的 Memory。

        ``patch`` 是 canonical dotted field path 到完整替换值的 mapping；
        只允许 ``meta.lifecycle.*`` 白名单字段与 ``meta.access_policy`` 整体
        替换。资源不存在返回 ``None``；未知路径、类型错误或 Workspace 不
        匹配直接拒绝。
        """

    @abstractmethod
    async def delete(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> bool: ...

    @abstractmethod
    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool: ...

    @abstractmethod
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
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def scroll(
        self,
        identity_scope: IdentityScope,
        filters: QueryFilters | None = None,
        limit: int = 100,
        *,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]: ...

    @abstractmethod
    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]: ...

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
        vitality_threshold: float | None = None,
    ) -> list[ArchiveRecord]: ...

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
    async def get(
        self,
        identity_scope: IdentityScope,
        ref_or_id: ArtifactRef | str,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def exists(
        self,
        identity_scope: IdentityScope,
        artifact_id: str,
    ) -> bool: ...

    @abstractmethod
    async def list_by_memory(
        self,
        identity_scope: IdentityScope,
        memory_id: str,
        artifact_type: ArtifactType | None = None,
    ) -> list[ArtifactRef]: ...

    @abstractmethod
    async def verify(
        self,
        identity_scope: IdentityScope,
        ref: ArtifactRef,
    ) -> ArtifactIntegrityResult: ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="artifact", healthy=True, required=False)


__all__ = [
    "ShortTermStoragePort",
    "MidTermStoragePort",
    "LongTermStoragePort",
    "ArtifactStoragePort",
]
