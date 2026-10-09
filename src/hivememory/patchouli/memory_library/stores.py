"""Storage facades for Patchouli memory layers.

短期存储只暴露持久化事实（CRUD）。Topic 状态转换、compact、settle、驱逐与
路由策略不属于本模块；占用权（lease）由 ``TopicWorkingSet`` 管理，编排由
Perception Familiar 承担。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.core.models import (
    ActorIdentity,
    MemoryAtom,
    TopicData,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import ArtifactRef
from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)
from hivememory.patchouli.memory_library.ports import (
    ArtifactStoragePort,
    LongTermStoragePort,
    MemoryChangePublisherPort,
    MidTermStoragePort,
    ShortTermStoragePort,
)

logger = logging.getLogger(__name__)


class ShortTermMemoryStore:
    """短期 Topic 持久化 facade（纯 CRUD）。

    职责：封装 Port（get/put/delete/list）、全局 ID 唯一性检查（Port 内执行）、
    返回不可变 ``TopicData`` 业务快照。

    不职责：不持有驻留容量与访问时间（``TopicWorkingSet`` 的工作集索引）、
    不持有执行占用状态（lease 表）、不执行 compact/settle 等组合操作
    （Familiar 的编排）。
    """

    def __init__(self, port: ShortTermStoragePort | None = None) -> None:
        self._port = port or InMemoryShortTermStorage()
        # 只保护 Port 读写的原子性；工作集与生命周期决策不在本模块。
        self._lock = threading.RLock()

    def get(self, belong_to: WorkspaceIdentity, topic_id: str) -> TopicData | None:
        """读取不可变业务快照；访问追踪在 WorkingSet，不在此处。"""
        with self._lock:
            return self._port.get(belong_to, topic_id)

    def put(self, topic: TopicData) -> None:
        """写入或替换话题快照；全局 ID 唯一性检查在 Port 内部。"""
        if not isinstance(topic, TopicData):
            raise TypeError("short-term store accepts TopicData snapshots")
        with self._lock:
            self._port.put(topic)

    def create(
        self,
        belong_to: WorkspaceIdentity,
        topic_title: str = "新建话题",
        topic_summary: str = "",
        *,
        topic_id: str | None = None,
    ) -> TopicData:
        """创建新话题并返回初始快照。"""
        topic = TopicData(
            topic_id=topic_id or str(uuid4()),
            workspace_identity=belong_to,
            topic_title=topic_title,
            topic_summary=topic_summary,
            last_update=datetime.now().timestamp(),
        )
        self.put(topic)
        return topic

    def delete(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """删除话题；是否允许删除（占用检查）由调用方持有 lease 判断。"""
        with self._lock:
            return self._port.delete(belong_to, topic_id)

    def list_by_workspace(
        self,
        belong_to: WorkspaceIdentity,
        *,
        include_empty: bool = True,
    ) -> list[TopicData]:
        """列出 Workspace 内的话题快照。"""
        with self._lock:
            topics = self._port.list_by_workspace(belong_to)
            if not include_empty:
                topics = [topic for topic in topics if topic.has_content]
            return topics

    def list_all(self) -> list[TopicData]:
        """列出全部 Workspace 的话题快照（进程级维护路径使用）。"""
        with self._lock:
            return self._port.list_all()

    def count(self, belong_to: WorkspaceIdentity) -> int:
        """统计 Workspace 内话题数量。"""
        with self._lock:
            return self._port.count(belong_to)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


class MidTermMemoryStore:
    """中期记忆存储（向量库）。

    alias 唯一性不变量（A2 §8 D-4）：同一 Workspace 的中期库内，一个 alias
    至多被一条 Memory 占用。``upsert`` 是全部完整写入（生成、手工编辑、
    Profile 管理、revive）的汇聚点，也是唯一能覆盖所有路径的检查点，因此
    在主后端写入前校验；归档即释放 alias，revive 撞名显式失败。
    ``patch_payload`` 白名单不含 ``index.alias``，该路径无需校验。

    生产 Runtime 注入变更发布端口；四个提交入口都在 finally 中内联等待
    失效通知，提交失败与删除未命中同样通知，不回滚已完成的后端写入。
    """

    # 唯一性判定只需确认"除自身外是否还有其他占用者"：自身至多一条，
    # 因此取两条即可覆盖。
    _ALIAS_HOLDER_PROBE_LIMIT = 2

    def __init__(
        self,
        primary: MidTermStoragePort,
        secondary: list[MidTermStoragePort] | None = None,
        *,
        change_publisher: MemoryChangePublisherPort | None = None,
    ) -> None:
        self._primary = primary
        self._secondary: list[MidTermStoragePort] = secondary or []
        self._change_publisher = change_publisher

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        """提交完整 canonical Memory；primary 写入后沿顺序同步 secondary。

        写入前校验 alias 唯一性，冲突时抛 ``MemoryAliasConflictError`` 且不
        产生任何写入。首版沿用 A2-P 的串行写入假设：并发写入者之间"检查→
        写入"的竞态不在保证范围内。
        """
        try:
            await self.ensure_alias_available(memory)
            await self._primary.upsert(memory, recompute_vectors=recompute_vectors)
            for secondary in self._secondary:
                await secondary.upsert(memory, recompute_vectors=recompute_vectors)
        finally:
            # primary 失败或 secondary 部分提交后也不能继续信任读取投影。
            await self._publish_change(memory.workspace_identity, memory.id, "upsert")

    async def list_alias_holders(
        self,
        workspace_identity: WorkspaceIdentity,
        alias: str,
        *,
        limit: int = _ALIAS_HOLDER_PROBE_LIMIT,
    ) -> list[UUID]:
        """返回 Workspace 中期库内占用 ``alias`` 的 memory_id（primary 为准）。"""
        return await self._primary.list_alias_holders(workspace_identity, alias, limit=limit)

    async def ensure_alias_available(self, memory: MemoryAtom) -> None:
        """校验 ``memory`` 的 alias 未被同 Workspace 的其他 Memory 占用。

        无 alias 的原子不参与唯一性约束；自身已持有该 alias（如保留原 alias
        的内容更新）视为可用。内容提交路径可在写版本 Artifact 之前调用本方法
        提前失败，避免冲突时留下孤立记录；``upsert`` 仍会再次校验。
        """
        alias = memory.index.alias
        if not alias:
            return
        holders = await self.list_alias_holders(memory.workspace_identity, alias)
        others = [holder for holder in holders if holder != memory.id]
        if others:
            raise MemoryAliasConflictError(
                "alias 已被同一 Workspace 内的其他记忆占用",
                details={
                    "alias": alias,
                    "memory_id": str(memory.id),
                    "conflicting_memory_id": str(others[0]),
                    "reason": "alias_occupied",
                },
            )

    async def get(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: UUID,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return await self._primary.get(
            belong_to,
            memory_id,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def get_by_alias(
        self,
        belong_to: WorkspaceIdentity,
        alias: str,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return await self._primary.get_by_alias(
            belong_to,
            alias,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        return await self._primary.get_by_key(key)

    async def patch_payload(
        self,
        key: WorkspaceMemoryKey,
        patch: Mapping[str, Any],
    ) -> MemoryAtom | None:
        """受限局部更新：primary 提交并返回结果，同一 patch 沿顺序同步 secondary。

        任一存储失败按顺序直接传播，不回滚已成功的写入（首版串行假设）。
        """
        try:
            result = await self._primary.patch_payload(key, patch)
            for secondary in self._secondary:
                await secondary.patch_payload(key, patch)
            return result
        finally:
            await self._publish_change(key.workspace_identity, key.memory_id, "patch")

    async def delete(self, belong_to: WorkspaceIdentity, memory_id: UUID) -> bool:
        """删除 canonical；未命中同样失效可能残留的读取投影。"""
        try:
            result = await self._primary.delete(belong_to, memory_id)
            for secondary in self._secondary:
                await secondary.delete(belong_to, memory_id)
            return result
        finally:
            await self._publish_change(belong_to, memory_id, "delete")

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        """按复合归属键删除；通知不依赖后端是否命中。"""
        try:
            result = await self._primary.delete_by_key(key)
            for secondary in self._secondary:
                await secondary.delete_by_key(key)
            return result
        finally:
            await self._publish_change(key.workspace_identity, key.memory_id, "delete")

    async def _publish_change(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: UUID,
        operation: Literal["upsert", "patch", "delete"],
    ) -> None:
        """通过注入端口内联通知，等待失效完成后才结束提交调用。"""
        if self._change_publisher is None:
            return
        await self._change_publisher.publish_change(
            MemoryChangeEvent(belong_to=belong_to, memory_id=memory_id, operation=operation)
        )

    async def search(
        self,
        belong_to: WorkspaceIdentity,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ):
        return await self._primary.search(
            belong_to,
            query,
            top_k,
            filters=filters,
            mode=mode,
            score_threshold=score_threshold,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def scroll(
        self,
        belong_to: WorkspaceIdentity,
        filters=None,
        limit: int = 100,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]:
        return await self._primary.scroll(
            belong_to,
            filters,
            limit,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return await self._primary.list_all_for_maintenance(limit)

    async def check_health(self) -> StorageHealthComponent:
        primary_health = await self._primary.check_health()
        if not primary_health.healthy:
            return primary_health
        for index, secondary in enumerate(self._secondary):
            health = await secondary.check_health()
            if not health.healthy and health.required:
                return StorageHealthComponent(
                    name=f"mid_term.secondary.{index}",
                    healthy=False,
                    required=True,
                    detail=health.detail,
                )
        return primary_health


class LongTermMemoryStore:
    """长期记忆存储（冷存储）。"""

    def __init__(self, port: LongTermStoragePort) -> None:
        self._port = port

    async def persist(self, memory: MemoryAtom) -> None:
        await self._port.persist(memory)

    async def load(self, key: WorkspaceMemoryKey) -> MemoryAtom:
        return await self._port.load(key)

    async def remove(self, key: WorkspaceMemoryKey) -> None:
        await self._port.remove(key)

    async def is_archived(self, key: WorkspaceMemoryKey) -> bool:
        return await self._port.is_archived(key)

    async def query(
        self,
        limit: int = 100,
        vitality_threshold: float | None = None,
    ) -> list[ArchiveRecord]:
        return await self._port.query(limit=limit, vitality_threshold=vitality_threshold)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


class ArtifactStore:
    """Artifact 附属资产仓库。"""

    def __init__(self, port: ArtifactStoragePort) -> None:
        self._port = port

    async def put(self, artifact) -> ArtifactRef:
        return await self._port.put(artifact)

    async def get(self, belong_to: WorkspaceIdentity, ref_or_id) -> dict:
        return await self._port.get(belong_to, ref_or_id)

    async def exists(self, belong_to: WorkspaceIdentity, artifact_id: str) -> bool:
        return await self._port.exists(belong_to, artifact_id)

    async def list_by_memory(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: str,
        artifact_type=None,
    ) -> list:
        return await self._port.list_by_memory(
            belong_to,
            memory_id,
            artifact_type,
        )

    async def verify(self, belong_to: WorkspaceIdentity, ref) -> ArtifactIntegrityResult:
        return await self._port.verify(belong_to, ref)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


__all__ = ["ShortTermMemoryStore", "MidTermMemoryStore", "LongTermMemoryStore", "ArtifactStore"]
