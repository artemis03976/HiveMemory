"""
MemoryLibrary 三层存储 Port 接口定义

定义短期 / 中期 / 长期三层存储的抽象契约，供各层 Store 通过 Port 多态实现。
实现类不应感知其他层的存储，跨层操作由 MemoryLibrary 编排。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from hivememory.core.models import (
    ActorIdentity,
    MemoryAtom,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType, BaseArtifact
from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)

if TYPE_CHECKING:
    from hivememory.core.models.query import QueryFilters


# ============ ShortTermStoragePort ============


class ShortTermStoragePort(ABC):
    """
    短期存储 Port：每个话题一条只追加的日志，按话题 ID 存取。

    一行是一条已经编码的记录（不含换行），记录的含义与编解码属于 ``ShortTermMemoryStore``；
    Port 只保证一行要么完整写下、要么不留痕迹。话题 ID 全局唯一，Workspace 归属写在日志的
    第一条记录里。

    感知热路径使用同步接口，Port 也是同步契约：写入在返回之前交给操作系统，进程崩溃不丢；
    不保证断电耐久。

    实现：
        JsonlShortTermStorage（每个话题一个 JSON Lines 文件）
    """

    @abstractmethod
    def append(self, topic_id: str, line: bytes) -> None:
        """在话题日志末尾追加一行，日志不存在时新建。

        失败时日志回到追加之前的长度，错误照常传播。
        """

    @abstractmethod
    def read_all(self) -> dict[str, list[bytes]]:
        """读回全部话题日志的完整行，按话题 ID 分组；启动时调用。

        最后一行不完整（写入中途进程退出）时丢弃，并从日志中截掉。
        """

    @abstractmethod
    def reject(self, topic_id: str) -> str:
        """把读不回的话题日志移到一旁保留，返回它的新位置；之后不再读回它。"""

    @abstractmethod
    def delete(self, topic_id: str) -> bool:
        """删除话题日志；日志不存在时返回 ``False``。"""

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="short_term", healthy=True)


# ============ MidTermStoragePort ============


class MemoryChangePublisherPort(Protocol):
    """中期库注入的内联失效通知端口；Store 不持有总线。"""

    async def publish_change(self, payload: MemoryChangeEvent) -> None:
        """等待本次失效通知的订阅者执行完毕。"""
        ...


class MidTermStoragePort(ABC):
    """
    中期存储 Port — 以 MemoryAtom 为边界的向量库操作。

    授权重验在 Port 实现内完成，存储预过滤不是授权事实：
        - 归属与发起者分别传入的读取（get/get_by_alias/search/scroll）校验
          Workspace ownership 与 actor 读取策略；``enforce_actor_visibility=False``
          仅供管理读取跳过 actor 策略，ownership 仍然生效；
        - 按 ``WorkspaceMemoryKey`` 的读取与删除是内部可信路径（编辑、强化、
          归档），只校验 ownership；
        - 存储失败以 ``StorageOfflineError`` / ``StorageReadError`` /
          ``StorageWriteError`` 传播，不以空结果或 ``False`` 掩盖。

    实现：
        QdrantStorageAdapter
        GraphStorageAdapter（future）
    """

    @abstractmethod
    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        """提交完整 canonical Memory；``recompute_vectors=False`` 时保留既有向量。"""

    @abstractmethod
    async def get(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: UUID,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None: ...

    @abstractmethod
    async def get_by_alias(
        self,
        belong_to: WorkspaceIdentity,
        alias: str,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None: ...

    @abstractmethod
    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None: ...

    @abstractmethod
    async def list_alias_holders(
        self,
        workspace_identity: WorkspaceIdentity,
        alias: str,
        *,
        limit: int,
    ) -> list[UUID]:
        """返回在该 Workspace 中期库内占用 ``alias`` 的 memory_id，最多 ``limit`` 条。

        内部可信路径：只校验 Workspace ownership，不叠加 actor 读取策略——
        alias 唯一性不变量作用于 Workspace 全体 Memory，而不是某个 Actor
        可见的子集。存储失败以结构化错误传播，不以空列表掩盖。
        """

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
        belong_to: WorkspaceIdentity,
        memory_id: UUID,
    ) -> bool: ...

    @abstractmethod
    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool: ...

    @abstractmethod
    async def search(
        self,
        belong_to: WorkspaceIdentity,
        query: str,
        top_k: int,
        filters: QueryFilters | None = None,
        mode: str = "dense",
        score_threshold: float = 0.0,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def scroll(
        self,
        belong_to: WorkspaceIdentity,
        filters: QueryFilters | None = None,
        limit: int = 100,
        *,
        cursor: str | None = None,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> tuple[list[MemoryAtom], str | None]:
        """按存储顺序读取一页，返回本页记忆与下一页的游标（取完时为 None）。"""

    @abstractmethod
    async def count(
        self,
        belong_to: WorkspaceIdentity,
        filters: QueryFilters | None = None,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> int:
        """按与 ``scroll`` 相同的归属、可见性与业务过滤条件计数。"""

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
        FileBasedStorageAdapter
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
        FilesystemArtifactStorageAdapter
        SQLArtifactStorageAdapter（future）
    """

    @abstractmethod
    async def put(self, artifact: BaseArtifact) -> ArtifactRef: ...

    @abstractmethod
    async def get(
        self,
        belong_to: WorkspaceIdentity,
        ref_or_id: ArtifactRef | str,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def exists(
        self,
        belong_to: WorkspaceIdentity,
        artifact_id: str,
    ) -> bool: ...

    @abstractmethod
    async def list_by_memory(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: str,
        artifact_type: ArtifactType | None = None,
    ) -> list[ArtifactRef]: ...

    @abstractmethod
    async def verify(
        self,
        belong_to: WorkspaceIdentity,
        ref: ArtifactRef,
    ) -> ArtifactIntegrityResult: ...

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="artifact", healthy=True, required=False)


__all__ = [
    "MemoryChangePublisherPort",
    "ShortTermStoragePort",
    "MidTermStoragePort",
    "LongTermStoragePort",
    "ArtifactStoragePort",
]
