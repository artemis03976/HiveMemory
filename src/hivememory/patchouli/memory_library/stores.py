"""
MemoryLibrary 三层 Store

每层 Store 持有对应的 StoragePort，封装上层调度逻辑。

ShortTermMemoryStore:
    - 持有 InMemoryShortTermStorage（Phase 1），future 可替换为 RedisShortTermStorage
    - 承接短期话题调度方法（add_block / update_summary / LRU 等）
    - 新增命名写入方法：clear_blocks / update_summary / update_title
    - 所有 buffer 字段写操作必须通过命名方法，不允许调用方直接写字段

MidTermMemoryStore:
    - 持有 primary（向量库）和 optional secondary（图库等）Port
    - 写入时同步到所有后端

LongTermMemoryStore:
    - 持有 LongTermStoragePort
    - 不负责跨层状态转移，由 MemoryLibrary 编排

实现阶段: Phase 1
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import (
    BufferState,
    LogicalBlock,
    MemoryAtom,
    MemoryReadScope,
    TopicData,
    WorkspaceAccessContext,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
    WorkspaceTopicKey,
    require_memory_read_scope,
    require_workspace_access_context,
)
from hivememory.core.models.artifact import ArtifactRef
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)
from hivememory.patchouli.memory_library.ports import (
    ArtifactStoragePort,
    LongTermStoragePort,
    MidTermStoragePort,
    ShortTermStoragePort,
)

logger = logging.getLogger(__name__)


# ============ ShortTermMemoryStore ============

class ShortTermMemoryStore:
    """
    短期记忆存储（MMU）

    持有 ShortTermStoragePort 实现，提供 buffer CRUD 与上层调度方法。
    Port contract is synchronous because the perception layer uses this store on
    its hot path without await points.
    """

    def __init__(
        self,
        port: Optional[ShortTermStoragePort] = None,
        max_resident_topics: int = 5,
    ) -> None:
        self._port: ShortTermStoragePort = port or InMemoryShortTermStorage()
        self.max_resident_topics = max_resident_topics
        self._last_active_topic_keys: dict[tuple[str, str], WorkspaceTopicKey] = {}
        logger.info(f"ShortTermMemoryStore 初始化, max_resident={max_resident_topics}")

    # ========== 最后活跃话题记录 ==========

    @staticmethod
    def _scope(workspace: WorkspaceIdentity) -> tuple[str, str]:
        return workspace.owner_user_id, workspace.workspace_id

    @staticmethod
    def _key(
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> WorkspaceTopicKey:
        access_context = require_workspace_access_context(access_context)
        return WorkspaceTopicKey.from_access_context(access_context, topic_id)

    def get_last_active_topic(
        self,
        access_context: WorkspaceAccessContext,
    ) -> Optional[str]:
        """返回当前 Workspace 最后活跃的 topic ID。"""
        access_context = require_workspace_access_context(access_context)
        key = self._last_active_topic_keys.get(self._scope(access_context.workspace_identity))
        return key.topic_id if key is not None else None

    def set_last_active_topic(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> None:
        key = self._key(access_context, topic_id)
        if self._port.get(key) is None:
            raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
        self._last_active_topic_keys[self._scope(access_context.workspace_identity)] = key

    def _require_buffer(self, key: WorkspaceTopicKey) -> SemanticBuffer:
        """返回必须存在的话题 buffer；写命令不得静默忽略缺失 topic。"""
        buf = self._port.get(key)
        if buf is None:
            raise KeyError(f"topic '{key.topic_id}' does not exist in requested Workspace")
        return buf

    # ========== CRUD ==========

    def get_topic_data(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
        *,
        touch: bool = True,
    ) -> Optional[TopicData]:
        """Return an immutable topic read view without exposing SemanticBuffer."""
        key = self._key(access_context, topic_id)
        buf = self._port.get(key)
        if buf is None:
            return None
        if touch:
            buf.last_accessed_at = datetime.now().timestamp()
            self._last_active_topic_keys[self._scope(access_context.workspace_identity)] = key
        return self._to_topic_data(buf)

    def get_topic_data_by_key(
        self,
        key: WorkspaceTopicKey,
        *,
        touch: bool = True,
    ) -> Optional[TopicData]:
        """由持有已验证复合键的内部协调器读取 Topic。"""
        buf = self._port.get(key)
        if buf is None:
            return None
        if touch:
            buf.last_accessed_at = datetime.now().timestamp()
            workspace = buf.workspace_identity
            self._last_active_topic_keys[self._scope(workspace)] = key
        return self._to_topic_data(buf)

    def list_topic_data(
        self,
        access_context: WorkspaceAccessContext,
        *,
        include_empty: bool = True,
    ) -> List[TopicData]:
        """Return immutable read views for active topics."""
        access_context = require_workspace_access_context(access_context)
        buffers = self._port.list_by_workspace(access_context.workspace_identity)
        if not include_empty:
            buffers = [buf for buf in buffers if buf.has_content]
        return [self._to_topic_data(buf) for buf in buffers]

    def topic_exists(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
        *,
        touch: bool = True,
    ) -> bool:
        return self.get_topic_data(access_context, topic_id, touch=touch) is not None

    def has_blocks(self, access_context: WorkspaceAccessContext, topic_id: str) -> bool:
        data = self.get_topic_data(access_context, topic_id, touch=False)
        return bool(data and data.blocks)

    def create_buffer(
        self,
        access_context: WorkspaceAccessContext,
        topic_title: str = "新建话题",
        topic_summary: str = "",
    ) -> SemanticBuffer:
        access_context = require_workspace_access_context(access_context)
        buf = SemanticBuffer(
            workspace_identity=access_context.workspace_identity,
            topic_title=topic_title,
            topic_summary=topic_summary,
        )
        self._port.put(buf.topic_key, buf)
        logger.debug(
            "创建话题段: topic_id=%s, owner=%s, workspace=%s",
            buf.topic_id,
            buf.workspace_identity.owner_user_id,
            buf.workspace_identity.workspace_id,
        )
        return buf

    def pop_buffer(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> Optional[SemanticBuffer]:
        key = self._key(access_context, topic_id)
        buf = self._port.pop(key)
        if buf is not None:
            logger.info(f"移除话题段: topic_id={topic_id}")
            scope = self._scope(access_context.workspace_identity)
            if self._last_active_topic_keys.get(scope) == key:
                self._last_active_topic_keys.pop(scope, None)
        return buf

    def pop_buffer_by_key(self, key: WorkspaceTopicKey) -> Optional[SemanticBuffer]:
        """由持有已验证复合键的内部结算流程驱逐 Topic。"""
        buf = self._port.pop(key)
        if buf is not None:
            scope = self._scope(buf.workspace_identity)
            if self._last_active_topic_keys.get(scope) == key:
                self._last_active_topic_keys.pop(scope, None)
        return buf

    # ========== 写操作（命名方法，禁止调用方直接写 buffer 字段）==========

    def add_block(self, key: WorkspaceTopicKey, block: LogicalBlock) -> None:
        buf = self._require_buffer(key)
        buf.blocks.append(block)
        buf.total_tokens += block.total_tokens
        buf.last_update = datetime.now().timestamp()

    def clear_blocks(self, key: WorkspaceTopicKey) -> None:
        """清空 blocks 并重置 token 计数（替代 buffer.blocks.clear() + total_tokens=0）。"""
        buf = self._require_buffer(key)
        buf.blocks.clear()
        buf.total_tokens = 0
        buf.last_update = datetime.now().timestamp()

    def update_summary(
        self,
        key: WorkspaceTopicKey,
        summary: str,
    ) -> None:
        """只更新 state summary，不改变当前 blocks。"""
        buf = self._require_buffer(key)
        buf.state_summary = summary
        buf.last_update = datetime.now().timestamp()

    def apply_compaction(
        self,
        key: WorkspaceTopicKey,
        summary: str,
        *,
        retain_count: int,
    ) -> int:
        """写入摘要并保留最近 N 个 blocks，返回被裁剪的 block 数。

        所有 compact 路径都必须保证至少保留一个最新 block；传入小于 1 的
        ``retain_count`` 在输入边界以具体异常拒绝，不静默提升。
        """
        if retain_count < 1:
            raise ValueError("retain_count must be >= 1")

        buf = self._require_buffer(key)
        buf.state_summary = summary
        buf.last_update = datetime.now().timestamp()
        if len(buf.blocks) <= retain_count:
            return 0
        folded = len(buf.blocks) - retain_count
        buf.blocks = buf.blocks[-retain_count:]
        buf.total_tokens = sum(b.total_tokens for b in buf.blocks)
        return folded

    def update_title(self, key: WorkspaceTopicKey, title: str) -> None:
        """写入 topic_title（替代 buffer.topic_title = title）。"""
        buf = self._require_buffer(key)
        buf.topic_title = title

    def update_metadata(self, key: WorkspaceTopicKey, state: Optional[BufferState] = None) -> None:
        buf = self._require_buffer(key)
        if state is not None:
            buf.state = state
        buf.last_update = datetime.now().timestamp()

    def update_model_used(self, key: WorkspaceTopicKey, model_used: str) -> None:
        """写入最近一次 run 使用的模型展示名。"""
        buf = self._require_buffer(key)
        buf.model_used = model_used

    # ========== LRU ==========

    def get_lru_topic(self, access_context: WorkspaceAccessContext) -> Optional[str]:
        """返回访问时间最久远的话题 topic_id，无话题时返回 None。"""
        access_context = require_workspace_access_context(access_context)
        bufs = self._port.list_by_workspace(access_context.workspace_identity)
        if not bufs:
            return None
        return min(bufs, key=lambda b: b.last_accessed_at).topic_id

    def needs_eviction(self, access_context: WorkspaceAccessContext) -> bool:
        access_context = require_workspace_access_context(access_context)
        return self._port.count(access_context.workspace_identity) >= self.max_resident_topics

    def get_active_topic_buffer_count(self, access_context: WorkspaceAccessContext) -> int:
        access_context = require_workspace_access_context(access_context)
        return self._port.count(access_context.workspace_identity)

    # ========== info ==========

    def get_buffer_info(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> Dict[str, Any]:
        buf = self._port.get(self._key(access_context, topic_id))
        if buf:
            return {
                "exists": True,
                "topic_id": buf.topic_id,
                "block_count": len(buf.blocks),
                "total_tokens": buf.total_tokens,
                "has_content": buf.has_content,
                "state": buf.state.value if hasattr(buf.state, "value") else buf.state,
            }
        return {"exists": False}

    def _to_topic_data(self, buf: SemanticBuffer) -> TopicData:
        return TopicData(
            topic_id=buf.topic_id,
            workspace_identity=buf.workspace_identity,
            current_agent_id=buf.current_agent_id,
            topic_title=buf.topic_title,
            topic_summary=buf.topic_summary,
            state_summary=buf.state_summary,
            blocks=tuple(buf.blocks),
            state=buf.state,
            last_update=buf.last_update,
            last_accessed_at=buf.last_accessed_at,
            total_tokens=buf.total_tokens,
            model_used=buf.model_used,
        )

    def list_all_topic_data_for_maintenance(self) -> List[TopicData]:
        """供进程级 idle/shutdown 协调器遍历，不作为用户授权入口。"""
        return [self._to_topic_data(buf) for buf in self._port.list_all()]

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


# ============ MidTermMemoryStore ============

class MidTermMemoryStore:
    """
    中期记忆存储（向量库）

    持有 primary Port（向量库）和 optional secondary Ports（图库等）。
    写入时同步到所有后端；查询仅走 primary。
    """

    def __init__(
        self,
        primary: MidTermStoragePort,
        secondary: Optional[List[MidTermStoragePort]] = None,
    ) -> None:
        self._primary = primary
        self._secondary: List[MidTermStoragePort] = secondary or []

    async def upsert(self, memory: MemoryAtom) -> None:
        """仅持久化已通过 v2 领域校验的 canonical Memory。"""
        await self._primary.upsert(memory)
        for s in self._secondary:
            await s.upsert(memory)

    async def get(
        self,
        scope: MemoryReadScope,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]:
        return await self._primary.get(require_memory_read_scope(scope), memory_id)

    async def get_by_alias(
        self,
        scope: MemoryReadScope,
        alias: str,
    ) -> Optional[MemoryAtom]:
        return await self._primary.get_by_alias(require_memory_read_scope(scope), alias)

    async def get_for_mutation(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]:
        access_context = require_workspace_access_context(access_context)
        return await self._primary.get_for_mutation(access_context, memory_id)

    async def get_by_key(self, key: WorkspaceMemoryKey) -> Optional[MemoryAtom]:
        return await self._primary.get_by_key(key)

    async def update_access_info(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> None:
        access_context = require_workspace_access_context(access_context)
        await self._primary.update_access_info(access_context, memory_id)

    async def delete(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> bool:
        access_context = require_workspace_access_context(access_context)
        result = await self._primary.delete(access_context, memory_id)
        for s in self._secondary:
            await s.delete(access_context, memory_id)
        return result

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        result = await self._primary.delete_by_key(key)
        for secondary in self._secondary:
            await secondary.delete_by_key(key)
        return result

    async def batch_delete(
        self,
        access_context: WorkspaceAccessContext,
        ids: List[UUID],
    ) -> int:
        access_context = require_workspace_access_context(access_context)
        count = await self._primary.batch_delete(access_context, ids)
        for s in self._secondary:
            await s.batch_delete(access_context, ids)
        return count

    async def search(
        self,
        scope: MemoryReadScope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ):
        scope = require_memory_read_scope(scope)
        return await self._primary.search(
            scope,
            query,
            top_k,
            filters,
            mode,
            score_threshold,
        )

    async def scroll(
        self,
        scope: MemoryReadScope,
        filters=None,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        return await self._primary.scroll(
            require_memory_read_scope(scope),
            filters,
            limit,
        )

    async def count(self, scope: MemoryReadScope, filters=None) -> int:
        return await self._primary.count(require_memory_read_scope(scope), filters)

    async def list_all_for_maintenance(self, limit: int = 10000) -> List[MemoryAtom]:
        """供进程级生命周期维护遍历，不作为用户授权入口。"""
        return await self._primary.list_all_for_maintenance(limit)

    async def check_health(self) -> StorageHealthComponent:
        primary_health = await self._primary.check_health()
        if not primary_health.healthy:
            return primary_health

        for index, secondary in enumerate(self._secondary):
            secondary_health = await secondary.check_health()
            if not secondary_health.healthy and secondary_health.required:
                return StorageHealthComponent(
                    name=f"mid_term.secondary.{index}",
                    healthy=False,
                    required=True,
                    detail=secondary_health.detail,
                )

        return primary_health


# ============ LongTermMemoryStore ============

class LongTermMemoryStore:
    """
    长期记忆存储（冷存储）

    持有 LongTermStoragePort 实现，只负责读写冷存储。
    archive / revive 跨层操作由 MemoryLibrary 编排。
    """

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
        vitality_threshold: Optional[float] = None,
    ) -> List[ArchiveRecord]:
        return await self._port.query(limit=limit, vitality_threshold=vitality_threshold)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


# ============ ArtifactStore ============

class ArtifactStore:
    """Artifact 附属资产仓库（书库隐喻的附属档案室）。"""

    def __init__(self, port: ArtifactStoragePort) -> None:
        self._port = port

    async def put(self, artifact) -> ArtifactRef:
        return await self._port.put(artifact)

    async def get(self, access_context: WorkspaceAccessContext, ref_or_id) -> dict:
        access_context = require_workspace_access_context(access_context)
        return await self._port.get(access_context, ref_or_id)

    async def exists(
        self,
        access_context: WorkspaceAccessContext,
        artifact_id: str,
    ) -> bool:
        access_context = require_workspace_access_context(access_context)
        return await self._port.exists(access_context, artifact_id)

    async def list_by_memory(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: str,
        artifact_type=None,
    ) -> list:
        access_context = require_workspace_access_context(access_context)
        return await self._port.list_by_memory(access_context, memory_id, artifact_type)

    async def verify(
        self,
        access_context: WorkspaceAccessContext,
        ref,
    ) -> ArtifactIntegrityResult:
        access_context = require_workspace_access_context(access_context)
        return await self._port.verify(access_context, ref)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


__all__ = [
    "ShortTermMemoryStore",
    "MidTermMemoryStore",
    "LongTermMemoryStore",
    "ArtifactStore",
]
