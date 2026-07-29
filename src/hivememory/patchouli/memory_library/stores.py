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

from hivememory.core.models import BufferState, LogicalBlock, MemoryAtom, TopicData
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.models import (
    StorageHealthComponent,
)
from hivememory.patchouli.memory_library.ports import (
    LongTermStoragePort,
    MidTermStoragePort,
    ShortTermStoragePort,
    ArtifactStoragePort,
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
        self._last_active_topic_id: Optional[str] = None
        logger.info(f"ShortTermMemoryStore 初始化, max_resident={max_resident_topics}")

    # ========== 最后活跃话题记录 ==========

    def get_last_active_topic(self) -> Optional[str]:
        return self._last_active_topic_id

    def set_last_active_topic(self, topic_id: str) -> None:
        self._last_active_topic_id = topic_id

    # ========== CRUD ==========

    def get_topic_data(
        self,
        topic_id: str,
        *,
        touch: bool = True,
    ) -> Optional[TopicData]:
        """Return an immutable topic read view without exposing SemanticBuffer."""
        buf = self._port.get(topic_id)
        if buf is None:
            return None
        if touch:
            buf.last_accessed_at = datetime.now().timestamp()
            self._last_active_topic_id = topic_id
        return self._to_topic_data(buf)

    def list_topic_data(
        self,
        user_id: Optional[str] = None,
        *,
        include_empty: bool = True,
    ) -> List[TopicData]:
        """Return immutable read views for active topics."""
        if user_id is None:
            buffers = self._port.list_all()
        else:
            buffers = self._port.list_by_user(user_id)
        if not include_empty:
            buffers = [buf for buf in buffers if buf.blocks]
        return [self._to_topic_data(buf) for buf in buffers]

    def topic_exists(self, topic_id: str, *, touch: bool = True) -> bool:
        return self.get_topic_data(topic_id, touch=touch) is not None

    def has_blocks(self, topic_id: str) -> bool:
        data = self.get_topic_data(topic_id, touch=False)
        return bool(data and data.blocks)

    def create_buffer(
        self,
        user_id: str,
        topic_title: str = "新建话题",
        topic_summary: str = "",
    ) -> SemanticBuffer:
        buf = SemanticBuffer(user_id=user_id, topic_title=topic_title, topic_summary=topic_summary)
        self._port.put(buf.topic_id, buf)
        logger.debug(f"创建话题段: topic_id={buf.topic_id}, owner={user_id}")
        return buf

    def pop_buffer(self, topic_id: str) -> Optional[SemanticBuffer]:
        buf = self._port.pop(topic_id)
        if buf is not None:
            logger.info(f"移除话题段: topic_id={topic_id}")
        return buf

    # ========== 写操作（命名方法，禁止调用方直接写 buffer 字段）==========

    def add_block(self, topic_id: str, block: LogicalBlock) -> None:
        buf = self._port.get(topic_id)
        if buf is None:
            logger.error(f"add_block: topic_id={topic_id} 不存在")
            return
        buf.blocks.append(block)
        buf.total_tokens += block.total_tokens
        buf.last_update = datetime.now().timestamp()

    def clear_blocks(self, topic_id: str) -> None:
        """清空 blocks 并重置 token 计数（替代 buffer.blocks.clear() + total_tokens=0）。"""
        buf = self._port.get(topic_id)
        if buf is None:
            return
        buf.blocks.clear()
        buf.total_tokens = 0
        buf.last_update = datetime.now().timestamp()

    def update_summary(
        self,
        topic_id: str,
        summary: str,
        *,
        retain_count: Optional[int] = None,
    ) -> int:
        """写入 state_summary，并可选保留最近 N 个 blocks。"""
        buf = self._port.get(topic_id)
        if buf is None:
            return 0
        buf.state_summary = summary
        buf.last_update = datetime.now().timestamp()
        if retain_count is None or len(buf.blocks) <= retain_count:
            return 0
        folded = len(buf.blocks) - retain_count
        buf.blocks = buf.blocks[-retain_count:]
        buf.total_tokens = sum(b.total_tokens for b in buf.blocks)
        return folded

    def update_title(self, topic_id: str, title: str) -> None:
        """写入 topic_title（替代 buffer.topic_title = title）。"""
        buf = self._port.get(topic_id)
        if buf is None:
            return
        buf.topic_title = title

    def clear_buffer(self, topic_id: str) -> List[LogicalBlock]:
        """清空话题段内容，保留在活跃池中。"""
        buf = self._port.get(topic_id)
        if buf is None:
            return []
        cleared = buf.blocks.copy()
        buf.blocks.clear()
        buf.total_tokens = 0
        buf.state_summary = ""
        buf.last_update = datetime.now().timestamp()
        return cleared

    def update_metadata(self, topic_id: str, state: Optional[BufferState] = None) -> None:
        buf = self._port.get(topic_id)
        if buf is None:
            return
        if state is not None:
            buf.state = state
        buf.last_update = datetime.now().timestamp()

    def update_model_used(self, topic_id: str, model_used: str) -> None:
        """写入最近一次 run 使用的模型展示名。"""
        buf = self._port.get(topic_id)
        if buf is None:
            return
        buf.model_used = model_used

    # ========== LRU ==========

    def get_lru_topic(self) -> Optional[str]:
        """返回访问时间最久远的话题 topic_id，无话题时返回 None。"""
        bufs = self._port.list_all()
        if not bufs:
            return None
        return min(bufs, key=lambda b: b.last_accessed_at).topic_id

    def needs_eviction(self) -> bool:
        return self._port.count() >= self.max_resident_topics

    def get_active_topic_buffer_count(self) -> int:
        return self._port.count()

    # ========== info ==========

    def get_buffer_info(self, topic_id: str) -> Dict[str, Any]:
        buf = self._port.get(topic_id)
        if buf:
            return {
                "exists": True,
                "topic_id": buf.topic_id,
                "block_count": len(buf.blocks),
                "total_tokens": buf.total_tokens,
                "state": buf.state.value if hasattr(buf.state, "value") else buf.state,
            }
        return {"exists": False}

    def _to_topic_data(self, buf: SemanticBuffer) -> TopicData:
        return TopicData(
            topic_id=buf.topic_id,
            user_id=buf.user_id,
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
        await self._primary.upsert(memory)
        for s in self._secondary:
            await s.upsert(memory)

    async def get(self, memory_id: UUID) -> Optional[MemoryAtom]:
        return await self._primary.get(memory_id)

    async def get_by_alias(self, alias: str, user_id: Optional[str] = None) -> Optional[MemoryAtom]:
        return await self._primary.get_by_alias(alias, user_id)

    async def update_access_info(self, memory_id: UUID) -> None:
        await self._primary.update_access_info(memory_id)

    async def delete(self, memory_id: UUID) -> bool:
        result = await self._primary.delete(memory_id)
        for s in self._secondary:
            await s.delete(memory_id)
        return result

    async def batch_delete(self, ids: List[UUID]) -> int:
        count = await self._primary.batch_delete(ids)
        for s in self._secondary:
            await s.batch_delete(ids)
        return count

    async def search(
        self,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ):
        return await self._primary.search(query, top_k, filters, mode, score_threshold)

    async def scroll(self, filters=None, limit: int = 100) -> List[MemoryAtom]:
        return await self._primary.scroll(filters, limit)

    async def count(self, filters=None) -> int:
        return await self._primary.count(filters)

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

    async def load(self, memory_id: UUID) -> MemoryAtom:
        return await self._port.load(memory_id)

    async def remove(self, memory_id: UUID) -> None:
        await self._port.remove(memory_id)

    async def is_archived(self, memory_id: UUID) -> bool:
        return await self._port.is_archived(memory_id)

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

    async def put(self, artifact) -> "ArtifactRef":
        return await self._port.put(artifact)

    async def get(self, ref_or_id) -> dict:
        return await self._port.get(ref_or_id)

    async def exists(self, artifact_id: str) -> bool:
        return await self._port.exists(artifact_id)

    async def list_by_memory(self, memory_id: str, artifact_type=None) -> list:
        return await self._port.list_by_memory(memory_id, artifact_type)

    async def verify(self, ref) -> "ArtifactIntegrityResult":
        return await self._port.verify(ref)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


__all__ = [
    "ShortTermMemoryStore",
    "MidTermMemoryStore",
    "LongTermMemoryStore",
    "ArtifactStore",
]
