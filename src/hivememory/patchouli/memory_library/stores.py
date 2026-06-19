"""
MemoryLibrary 三层 Store

每层 Store 持有对应的 StoragePort，封装上层调度逻辑。

ShortTermMemoryStore:
    - 持有 InMemoryShortTermStorage（Phase 1），future 可替换为 RedisShortTermStorage
    - 保留 SemanticBufferManager 的全部调度方法（add_block / fold_blocks / LRU 等）
    - 新增命名写入方法：clear_blocks / update_summary / update_title
    - 所有 buffer 字段写操作必须通过命名方法，不允许调用方直接写字段

MidTermMemoryStore:
    - 持有 primary（向量库）和 optional secondary（图库等）Port
    - 写入时同步到所有后端

LongTermMemoryStore:
    - 持有 LongTermStoragePort
    - 不负责跨层状态转移，由 MemoryLibrary 编排

版本: 0.1.0 (Phase 1)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.engines.perception.models import BufferState, LogicalBlock, SemanticBuffer
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.models import TopicData
from hivememory.patchouli.memory_library.ports import (
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
    Phase 1 使用 InMemoryShortTermStorage，其 _get_sync / _put_sync 等 sync
    快捷方法供本 Store 直接调用，保持所有公开方法同步以与感知层兼容。
    """

    def __init__(
        self,
        port: Optional[ShortTermStoragePort] = None,
        max_resident_topics: int = 5,
    ) -> None:
        self._port: InMemoryShortTermStorage = port or InMemoryShortTermStorage()  # type: ignore[assignment]
        self.max_resident_topics = max_resident_topics
        self._last_active_topic_id: Optional[str] = None
        logger.info(f"ShortTermMemoryStore 初始化, max_resident={max_resident_topics}")

    # ── last_active tracking ──

    def get_last_active_topic(self) -> Optional[str]:
        return self._last_active_topic_id

    def set_last_active_topic(self, topic_id: str) -> None:
        self._last_active_topic_id = topic_id

    # ── CRUD ──

    def get_buffer(self, topic_id: str) -> Optional[SemanticBuffer]:
        buf = self._port._get_sync(topic_id)
        if buf is not None:
            buf.last_accessed_at = datetime.now().timestamp()
            self._last_active_topic_id = topic_id
        return buf

    def get_all_buffers(self) -> List[SemanticBuffer]:
        return self._port._list_all_sync()

    def get_buffers_by_owner(self, user_id: str) -> List[SemanticBuffer]:
        return self._port._list_by_user_sync(user_id)

    def get_topic_data(self, topic_id: str, *, touch: bool = True) -> Optional[TopicData]:
        """Return an immutable topic read view without exposing SemanticBuffer."""
        buf = self._port._get_sync(topic_id)
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
            buffers = self._port._list_all_sync()
        else:
            buffers = self._port._list_by_user_sync(user_id)
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
        self._port._put_sync(buf.topic_id, buf)
        logger.debug(f"创建话题段: topic_id={buf.topic_id}, owner={user_id}")
        return buf

    def pop_buffer(self, topic_id: str) -> Optional[SemanticBuffer]:
        buf = self._port._pop_sync(topic_id)
        if buf is not None:
            logger.info(f"移除话题段: topic_id={topic_id}")
        return buf

    # ── 写操作（命名方法，禁止调用方直接写 buffer 字段）──

    def add_block(self, topic_id: str, block: LogicalBlock) -> None:
        buf = self._port._get_sync(topic_id)
        if buf is None:
            logger.error(f"add_block: topic_id={topic_id} 不存在")
            return
        buf.blocks.append(block)
        buf.total_tokens += block.total_tokens
        buf.last_update = datetime.now().timestamp()

    def clear_blocks(self, topic_id: str) -> None:
        """清空 blocks 并重置 token 计数（替代 buffer.blocks.clear() + total_tokens=0）。"""
        buf = self._port._get_sync(topic_id)
        if buf is None:
            return
        buf.blocks.clear()
        buf.total_tokens = 0
        buf.last_update = datetime.now().timestamp()

    def update_summary(self, topic_id: str, summary: str) -> None:
        """写入 state_summary（替代 buffer.state_summary = summary）。"""
        buf = self._port._get_sync(topic_id)
        if buf is None:
            return
        buf.state_summary = summary
        buf.last_update = datetime.now().timestamp()

    def update_title(self, topic_id: str, title: str) -> None:
        """写入 topic_title（替代 buffer.topic_title = title）。"""
        buf = self._port._get_sync(topic_id)
        if buf is None:
            return
        buf.topic_title = title

    def clear_buffer(self, topic_id: str) -> List[LogicalBlock]:
        """清空话题段内容，保留在活跃池中（向后兼容 SemanticBufferManager.clear_buffer）。"""
        buf = self._port._get_sync(topic_id)
        if buf is None:
            return []
        cleared = buf.blocks.copy()
        buf.blocks.clear()
        buf.total_tokens = 0
        buf.state_summary = ""
        buf.last_update = datetime.now().timestamp()
        return cleared

    def fold_blocks(self, topic_id: str, state_summary: str, retain_count: int) -> int:
        buf = self._port._get_sync(topic_id)
        if buf is None:
            return 0
        buf.state_summary = state_summary
        buf.last_update = datetime.now().timestamp()
        if len(buf.blocks) <= retain_count:
            return 0
        folded = len(buf.blocks) - retain_count
        buf.blocks = buf.blocks[-retain_count:]
        buf.total_tokens = sum(b.total_tokens for b in buf.blocks)
        return folded

    def update_metadata(self, topic_id: str, state: Optional[BufferState] = None) -> None:
        buf = self._port._get_sync(topic_id)
        if buf is None:
            return
        if state is not None:
            buf.state = state
        buf.last_update = datetime.now().timestamp()

    # ── LRU ──

    def get_lru_buffer(self) -> Optional[SemanticBuffer]:
        bufs = self._port._list_all_sync()
        if not bufs:
            return None
        return min(bufs, key=lambda b: b.last_accessed_at)

    def needs_eviction(self) -> bool:
        return self._port._count() >= self.max_resident_topics

    def get_active_topic_buffer_count(self) -> int:
        return self._port._count()

    # ── info ──

    def get_buffer_info(self, topic_id: str) -> Dict[str, Any]:
        buf = self._port._get_sync(topic_id)
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
            blocks=tuple(block.model_copy(deep=True) for block in buf.blocks),
            state=buf.state,
            last_update=buf.last_update,
            last_accessed_at=buf.last_accessed_at,
            total_tokens=buf.total_tokens,
        )


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
    ):
        return await self._primary.search(query, top_k, filters, mode)

    async def scroll(self, filters=None, limit: int = 100) -> List[MemoryAtom]:
        return await self._primary.scroll(filters, limit)

    async def count(self, filters=None) -> int:
        return await self._primary.count(filters)


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


__all__ = [
    "ShortTermMemoryStore",
    "MidTermMemoryStore",
    "LongTermMemoryStore",
]
