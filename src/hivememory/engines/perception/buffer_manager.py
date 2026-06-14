"""
HiveMemory Buffer Manager

纯状态管理器，管理 buffer 池的 CRUD 操作。

职责:
    - 管理 buffer 池 (Dict[str, SemanticBuffer])，key 为 topic_id
    - 提供基于 topic_id 的 CRUD 操作接口
    - 提供基于 owner (user_id:agent_id) 的索引查询
    - LRU 驱逐判定

注意:
    Trigger 调度逻辑已迁移至 TriggerManager。

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 5.0.0 (Phase 4.5 重构)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from hivememory.core.models import Identity
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
)

logger = logging.getLogger(__name__)


class SemanticBufferManager:
    """
    话题管理器 / MMU (TopicManager / Memory Management Unit)

    短期记忆的中央调度器，管理活跃话题池的生命周期。
    类似操作系统的 MMU，负责话题的换入(Swap-in)、换出(Swap-out)和 LRU 驱逐。

    映射关系 (ShortTermMemory.md §2.1):
        TopicManager = SemanticBufferManager
        active_topics = _buffers

    架构变更 (Phase 4.5):
        - _buffers 的 key 改为 topic_id
        - 新增 _user_index 索引：user_id:agent_id -> Set[topic_id]
        - 所有 CRUD 操作通过 topic_id 完成

    职责:
        - 管理活跃话题池 (Dict[str, SemanticBuffer/TopicSegment])，key 为 topic_id
        - 提供线程安全的 CRUD 操作
        - 提供话题路由 (route) 与换出 (swap_out)
        - LRU 驱逐判定

    Examples:
        >>> manager = SemanticBufferManager(max_resident_topics=5)
        >>> buffer = manager.get_buffer("topic_123")
    """

    def __init__(self, max_resident_topics: int = 5) -> None:
        """
        初始化 SemanticBufferManager (MMU)

        Args:
            max_resident_topics: 最大驻留话题数，超过此数量将触发 LRU 驱逐
        """
        # 活跃话题池 (L1 Cache / 驻留内存): key -> SemanticBuffer (TopicSegment)
        # Phase 4.5: key 使用 topic_id
        self._buffers: Dict[str, SemanticBuffer] = {}

        # 用户索引: user_id:agent_id -> Set[topic_id]
        # 用于查询特定用户的所有话题
        self._user_index: Dict[str, Set[str]] = {}

        # 最大驻留限制
        self.max_resident_topics = max_resident_topics

        # 线程安全
        self._lock = threading.RLock()

        # 最后活跃话题 ID（用于 manual_trigger 的默认回退）
        self._last_active_topic_id: Optional[str] = None

        logger.info(f"SemanticBufferManager 初始化完成, max_resident={max_resident_topics}")

    # ========== 最后活跃话题跟踪 ==========

    def get_last_active_topic(self) -> Optional[str]:
        """
        获取最后活跃的话题 ID

        用于 manual_trigger 的默认回退目标。

        Returns:
            最后活跃的 topic_id，如果没有活跃话题则返回 None
        """
        with self._lock:
            return self._last_active_topic_id

    def set_last_active_topic(self, topic_id: str) -> None:
        """
        设置最后活跃的话题 ID

        在 route_and_ingest 中自动调用。

        Args:
            topic_id: 话题 ID
        """
        with self._lock:
            self._last_active_topic_id = topic_id
            logger.debug(f"更新 last_active_topic_id: {topic_id}")

    # ========== 内部辅助方法 ==========

    def _get_owner_key(self, user_id: str) -> str:
        """获取 owner key"""
        return user_id

    def _update_user_index(self, user_id: str, topic_id: str, remove: bool = False) -> None:
        """更新用户索引"""
        if user_id not in self._user_index:
            self._user_index[user_id] = set()

        if remove:
            self._user_index[user_id].discard(topic_id)
            if not self._user_index[user_id]:
                del self._user_index[user_id]
        else:
            self._user_index[user_id].add(topic_id)

    # ========== 基于 topic_id 的 CRUD (Phase 4.5 新接口) ==========

    def get_buffer(self, topic_id: str) -> Optional[SemanticBuffer]:
        """
        通过 topic_id 获取话题段

        如果存在，更新 last_accessed_at 和 last_active_topic_id，然后返回 buffer；
        否则返回 None。

        Args:
            topic_id: 话题 ID

        Returns:
            SemanticBuffer or None
        """
        with self._lock:
            if topic_id in self._buffers:
                buf = self._buffers[topic_id]
                buf.last_accessed_at = datetime.now().timestamp()
                # 自动更新最后活跃话题
                self._last_active_topic_id = topic_id
                return buf
            return None

    def create_buffer(self, user_id: str, topic_title: str = "新建话题", topic_summary: str = "") -> SemanticBuffer:
        """
        创建新话题段

        唯一的 Create 方法，创建时生成新的 topic_id。

        Args:
            user_id: 用户标识
            topic_title: 话题标题
            topic_summary: 话题展示摘要（由 Gateway 生成，创建后不再修改）

        Returns:
            新创建的 SemanticBuffer（包含新生成的 topic_id）
        """
        with self._lock:
            buf = SemanticBuffer(user_id=user_id, topic_title=topic_title, topic_summary=topic_summary)
            topic_id = buf.topic_id

            self._buffers[topic_id] = buf
            self._update_user_index(user_id, topic_id)

            logger.debug(f"创建新话题段: topic_id={topic_id}, topic_title='{topic_title}', owner={user_id}")
            return buf

    def pop_buffer(self, topic_id: str) -> Optional[SemanticBuffer]:
        """
        通过 topic_id 移除并返回话题段

        Args:
            topic_id: 话题 ID

        Returns:
            被移除的 SemanticBuffer，不存在则返回 None
        """
        with self._lock:
            if topic_id not in self._buffers:
                return None

            buf = self._buffers.pop(topic_id)
            self._update_user_index(buf.user_id, topic_id, remove=True)

            logger.info(f"移除话题段: topic_id={topic_id}, topic_title='{buf.topic_title}'")
            return buf

    def clear_buffer(self, topic_id: str) -> List[LogicalBlock]:
        """
        清空话题段内容，保留在活跃池中

        Args:
            topic_id: 话题 ID

        Returns:
            被清除的 blocks 列表
        """
        with self._lock:
            buffer = self._buffers.get(topic_id)
            if not buffer:
                return []

            cleared_blocks = buffer.blocks.copy()
            buffer.clear()

            # 重置元数据
            buffer.state_summary = ""

            logger.debug(f"清空话题段内容: topic_id={topic_id}, 返回 {len(cleared_blocks)} 个 blocks")
            return cleared_blocks

    def add_block(self, topic_id: str, block: LogicalBlock) -> None:
        """
        将完成的 block 添加到 buffer

        Args:
            topic_id: 话题 ID
            block: 要添加的完成 block
        """
        with self._lock:
            buffer = self._buffers.get(topic_id)
            if not buffer:
                logger.error(f"尝试向不存在的 buffer 添加 block: topic_id={topic_id}")
                return

            buffer.blocks.append(block)
            buffer.total_tokens += block.total_tokens
            buffer.last_update = datetime.now().timestamp()

            logger.debug(f"将 block {block.block_id} 添加到 buffer topic_id={topic_id}")

    def fold_blocks(
        self,
        topic_id: str,
        state_summary: str,
        retain_count: int,
    ) -> int:
        """
        原子化页折叠操作 (Page Folding)

        保留最近 retain_count 个 blocks，丢弃其余旧 blocks，
        将 state_summary 写入 buffer.state_summary，重新计算 total_tokens。

        Args:
            topic_id: 话题 ID
            state_summary: 折叠摘要文本（累积拼接由调用方负责）
            retain_count: 保留的最近 block 数量

        Returns:
            int: 被折叠（丢弃）的 block 数量
        """
        with self._lock:
            buffer = self._buffers.get(topic_id)
            if not buffer:
                logger.error(f"尝试折叠不存在的 buffer: topic_id={topic_id}")
                return 0

            # 写入 state_summary（无论是否删除 blocks）
            buffer.state_summary = state_summary
            buffer.last_update = datetime.now().timestamp()

            # 如果 blocks 数量不足，只更新 summary，不删除 blocks
            if len(buffer.blocks) <= retain_count:
                logger.info(
                    f"Page fold (summary only): topic_id={topic_id}, "
                    f"blocks={len(buffer.blocks)} (retained all), "
                    f"summary_len={len(state_summary)}"
                )
                return 0

            # blocks 数量充足，执行正常折叠
            folded_count = len(buffer.blocks) - retain_count
            # 保留最近 retain_count 个 blocks
            buffer.blocks = buffer.blocks[-retain_count:]
            # 重新计算 total_tokens（仅基于保留的 blocks）
            buffer.total_tokens = sum(b.total_tokens for b in buffer.blocks)

            logger.info(
                f"Page fold: topic_id={topic_id}, "
                f"folded={folded_count}, retained={retain_count}, "
                f"new_total_tokens={buffer.total_tokens}"
            )
            return folded_count

    def update_metadata(
        self,
        topic_id: str,
        state: Optional[BufferState] = None,
    ) -> None:
        """
        更新缓冲区元数据

        Args:
            topic_id: 话题 ID
            state: 新的状态（None 表示不更新）
        """
        with self._lock:
            buffer = self._buffers.get(topic_id)
            if not buffer:
                logger.error(f"尝试更新不存在的 buffer 元数据: topic_id={topic_id}")
                return

            if state is not None:
                buffer.state = state

            buffer.last_update = datetime.now().timestamp()

    # ========== 所有者查询 ==========

    def get_buffers_by_owner(self, user_id: str) -> List[SemanticBuffer]:
        """
        获取指定用户的所有活跃话题

        Args:
            user_id: 用户标识

        Returns:
            话题列表
        """
        with self._lock:
            topic_ids = self._user_index.get(user_id, set())
            return [self._buffers[tid] for tid in topic_ids if tid in self._buffers]

    # ========== LRU 与生命周期管理 ==========

    def get_lru_buffer(self) -> Optional[SemanticBuffer]:
        """
        获取最近最少访问的话题段 (LRU)

        Returns:
            SemanticBuffer 或 None
        """
        with self._lock:
            if not self._buffers:
                return None
            lru_key = min(
                self._buffers.keys(),
                key=lambda k: self._buffers[k].last_accessed_at
            )
            return self._buffers[lru_key]

    def get_active_topic_buffer_count(self) -> int:
        """
        获取活跃话题数量

        Returns:
            int: 活跃话题数量
        """
        with self._lock:
            return len(self._buffers)

    def get_all_buffers(self) -> List[SemanticBuffer]:
        """
        获取所有活跃话题段

        用于遍历检查（如空闲超时）。

        Returns:
            List[SemanticBuffer]
        """
        with self._lock:
            return list(self._buffers.values())

    def needs_eviction(self) -> bool:
        """检查是否需要 LRU 驱逐"""
        with self._lock:
            return len(self._buffers) >= self.max_resident_topics

    def get_buffer_info(self, topic_id: str) -> Dict[str, Any]:
        """
        获取 buffer 信息

        Args:
            topic_id: 话题 ID

        Returns:
            buffer 信息字典
        """
        with self._lock:
            buffer = self._buffers.get(topic_id)

            if buffer:
                return {
                    "exists": True,
                    "topic_id": buffer.topic_id,
                    "block_count": len(buffer.blocks),
                    "total_tokens": buffer.total_tokens,
                    "state": buffer.state.value if hasattr(buffer.state, 'value') else buffer.state,
                }
            return {"exists": False}


__all__ = [
    "SemanticBufferManager",
]
