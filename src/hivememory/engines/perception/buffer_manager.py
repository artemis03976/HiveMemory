"""
HiveMemory Buffer Manager

纯状态管理器，管理 buffer 池的 CRUD 操作。

职责:
    - 管理 buffer 池 (Dict[str, SemanticBuffer])
    - 提供 CRUD 操作接口

不负责:
    - Flush 条件检测（由 PerceptionLayer 编排）
    - Flush 执行（由 PerceptionLayer 编排）
    - 话题核心更新（由 PerceptionLayer 编排）

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 3.1.0
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
    SimpleBuffer,
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

    职责:
        - 管理活跃话题池 (Dict[str, SemanticBuffer/TopicSegment])
        - 提供线程安全的 CRUD 操作
        - 提供话题路由 (route) 与换出 (swap_out)
        - 提供活跃话题菜单供 TheEye 路由决策
        - LRU 驱逐判定

    Examples:
        >>> manager = SemanticBufferManager(max_resident_topics=5)
        >>> buffer = manager.get_buffer(identity)
        >>> menu = manager.get_active_topics_menu()
        >>> manager.route(topic_id)
    """

    def __init__(self, max_resident_topics: int = 5) -> None:
        """
        初始化 SemanticBufferManager (MMU)

        Args:
            max_resident_topics: 最大驻留话题数，超过此数量将触发 LRU 驱逐
        """
        # 活跃话题池 (L1 Cache / 驻留内存): key -> SemanticBuffer (TopicSegment)
        self._buffers: Dict[str, SemanticBuffer] = {}

        # 最大驻留限制
        self.max_resident_topics = max_resident_topics

        # 线程安全
        self._lock = threading.RLock()

        logger.info(f"SemanticBufferManager (MMU) 初始化完成, max_resident={max_resident_topics}")

    # ========== Buffer CRUD ==========

    def get_buffer(self, identity: Identity) -> SemanticBuffer:
        """
        获取或创建话题段 (TopicSegment)

        Args:
            identity: 用于查找 buffer 的身份标识

        Returns:
            该身份对应的 SemanticBuffer (TopicSegment)
        """
        key = identity.buffer_key

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SemanticBuffer(
                    identity=identity,
                )
                logger.debug(f"创建新话题段 (TopicSegment): {key}")

            buf = self._buffers[key]
            buf.last_accessed_at = datetime.now().timestamp()
            return buf

    def add_block_to_buffer(
        self,
        identity: Identity,
        block: LogicalBlock,
    ) -> None:
        """
        将完成的 block 添加到 buffer

        Args:
            identity: 身份标识
            block: 要添加的完成 block
        """
        with self._lock:
            buffer = self.get_buffer(identity)
            buffer.blocks.append(block)
            buffer.total_tokens += block.total_tokens
            buffer.last_update = datetime.now().timestamp()
            logger.debug(f"将 block {block.block_id} 添加到 buffer {buffer.buffer_id}")

    def clear_buffer(self, identity: Identity) -> List[LogicalBlock]:
        """
        清空 buffer 并返回被清除的 blocks

        Args:
            identity: 身份标识

        Returns:
            被清除的 blocks 列表
        """
        with self._lock:
            key = identity.buffer_key
            if key not in self._buffers:
                return []

            buffer = self._buffers[key]
            cleared_blocks = buffer.blocks.copy()

            buffer.blocks.clear()
            buffer.total_tokens = 0
            buffer.last_update = datetime.now().timestamp()

            logger.debug(f"清空 buffer {key}, 返回 {len(cleared_blocks)} 个 blocks")
            return cleared_blocks

    def update_buffer_metadata(
        self,
        identity: Identity,
        topic_kernel_vector: Optional[List[float]] = None,
        relay_summary: Optional[str] = None,
        state: Optional[BufferState] = None,
        reset_topic_kernel: bool = False,
        reset_relay_summary: bool = False,
    ) -> None:
        """
        更新 buffer 元数据

        Args:
            identity: 身份标识
            topic_kernel_vector: 新的话题核心向量（None 表示不更新，除非 reset_topic_kernel=True）
            relay_summary: 新的接力摘要（None 表示不更新，除非 reset_relay_summary=True）
            state: 新的状态（None 表示不更新）
            reset_topic_kernel: 是否重置话题核心向量为 None
            reset_relay_summary: 是否重置接力摘要为 None
        """
        with self._lock:
            buffer = self.get_buffer(identity)

            if topic_kernel_vector is not None:
                buffer.topic_kernel_vector = topic_kernel_vector
            elif reset_topic_kernel:
                buffer.topic_kernel_vector = None

            if relay_summary is not None:
                buffer.relay_summary = relay_summary
            elif reset_relay_summary:
                buffer.relay_summary = None

            if state is not None:
                buffer.state = state

            buffer.last_update = datetime.now().timestamp()

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的话题 keys

        Returns:
            buffer key 列表
        """
        with self._lock:
            return list(self._buffers.keys())

    # ========== MMU 路由与生命周期 (Phase 4.5) ==========

    def get_active_topics_menu(self) -> List[Dict[str, str]]:
        """
        获取活跃话题菜单，供 TheEye 路由决策使用

        Returns:
            List[Dict]: [{"topic_id": buffer_id, "title": title, "buffer_key": key}, ...]
        """
        with self._lock:
            menu = []
            for key, buf in self._buffers.items():
                if buf.blocks:  # 只返回有内容的话题
                    menu.append({
                        "topic_id": buf.buffer_id,
                        "title": buf.title,
                        "buffer_key": key,
                    })
            return menu

    def route(self, topic_id: str) -> Optional[SemanticBuffer]:
        """
        根据 topic_id 换入(Swap-in)目标话题

        更新 last_accessed_at 时间戳，用于 LRU 驱逐判定。

        Args:
            topic_id: 目标话题的 buffer_id

        Returns:
            SemanticBuffer 如果找到，否则 None
        """
        with self._lock:
            for key, buf in self._buffers.items():
                if buf.buffer_id == topic_id:
                    buf.last_accessed_at = datetime.now().timestamp()
                    return buf
            return None

    def create_new_topic(self, identity: Identity, title: str = "新建话题") -> SemanticBuffer:
        """
        创建新话题段

        Args:
            identity: 身份标识
            title: 话题标题

        Returns:
            新创建的 SemanticBuffer (TopicSegment)
        """
        with self._lock:
            buf = SemanticBuffer(identity=identity, title=title)
            key = identity.buffer_key
            self._buffers[key] = buf
            logger.debug(f"创建新话题段: key={key}, title='{title}'")
            return buf

    def find_lru_topic(self) -> Optional[Tuple[str, SemanticBuffer]]:
        """
        找到最近最少访问的话题 (LRU)

        Returns:
            (buffer_key, SemanticBuffer) 或 None
        """
        with self._lock:
            if not self._buffers:
                return None
            lru_key = min(
                self._buffers.keys(),
                key=lambda k: self._buffers[k].last_accessed_at
            )
            return (lru_key, self._buffers[lru_key])

    def swap_out(self, buffer_key: str) -> Optional[SemanticBuffer]:
        """
        换出(Swap-out)指定话题，从活跃池中移除并返回

        调用方负责将返回的 TopicSegment 移交给 LibrarianCore 进行 MTM 归档。

        Args:
            buffer_key: 要换出的话题 key

        Returns:
            被换出的 SemanticBuffer，不存在则返回 None
        """
        with self._lock:
            evicted = self._buffers.pop(buffer_key, None)
            if evicted:
                logger.info(f"话题换出: key={buffer_key}, title='{evicted.title}'")
            return evicted

    def needs_eviction(self) -> bool:
        """检查是否需要 LRU 驱逐"""
        with self._lock:
            return len(self._buffers) >= self.max_resident_topics

    def get_active_count(self) -> int:
        """获取当前活跃话题数量"""
        with self._lock:
            return len(self._buffers)

    # ========== Info ==========

    def get_buffer_info(self, identity: Identity) -> Dict[str, Any]:
        """
        获取 buffer 信息

        Args:
            identity: 身份标识

        Returns:
            buffer 信息字典
        """
        with self._lock:
            buffer = self._buffers.get(identity.buffer_key)

            if buffer:
                return {
                    "exists": True,
                    "buffer_id": buffer.buffer_id,
                    "block_count": len(buffer.blocks),
                    "total_tokens": buffer.total_tokens,
                    "state": buffer.state.value if hasattr(buffer.state, 'value') else buffer.state,
                    "relay_summary": buffer.relay_summary,
                    "has_topic_kernel": buffer.topic_kernel_vector is not None,
                }
            return {"exists": False}


class SimpleBufferManager:
    """
    Simple Buffer 管理器 - 纯状态容器

    管理简单缓冲区 (SimpleBuffer) 的生命周期。
    仅提供 CRUD 操作，不包含业务逻辑。

    职责:
        - 管理 buffer 池 (Dict[str, SimpleBuffer])
        - 提供线程安全的 CRUD 操作

    Examples:
        >>> manager = SimpleBufferManager()
        >>> buffer = manager.get_buffer(identity)
        >>> manager.add_message(identity, message)
    """

    def __init__(self) -> None:
        """初始化 SimpleBufferManager"""
        # Buffer 池: key -> SimpleBuffer
        self._buffers: Dict[str, SimpleBuffer] = {}

        # 线程安全
        self._lock = threading.RLock()

        logger.info("SimpleBufferManager 初始化完成")

    # ========== Buffer CRUD ==========

    def get_buffer(self, identity: Identity) -> SimpleBuffer:
        """
        获取或创建 buffer

        Args:
            identity: 用于查找 buffer 的身份标识

        Returns:
            该身份对应的 SimpleBuffer
        """
        key = identity.buffer_key

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SimpleBuffer(
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    session_id=identity.session_id or key.split(":")[-1],
                )
                logger.debug(f"创建新 simple buffer: {key}")

            return self._buffers[key]

    def add_message(
        self,
        identity: Identity,
        message: StreamMessage,
    ) -> None:
        """
        添加消息到 buffer

        Args:
            identity: 身份标识
            message: 要添加的消息
        """
        with self._lock:
            buffer = self.get_buffer(identity)
            buffer.add_message(message)
            logger.debug(f"将消息添加到 simple buffer {buffer.buffer_id}")

    def clear_buffer(self, identity: Identity) -> bool:
        """
        清空 buffer

        Args:
            identity: 身份标识

        Returns:
            是否成功清空（如果 buffer 存在则返回 True）
        """
        with self._lock:
            key = identity.buffer_key
            if key in self._buffers:
                self._buffers[key].clear()
                logger.debug(f"清空 simple buffer {key}")
                return True
            return False

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 buffer keys

        Returns:
            buffer key 列表
        """
        with self._lock:
            return list(self._buffers.keys())

    def get_buffer_info(self, identity: Identity) -> Dict[str, Any]:
        """
        获取 buffer 信息

        Args:
            identity: 身份标识

        Returns:
            buffer 信息字典
        """
        with self._lock:
            buffer = self._buffers.get(identity.buffer_key)
            if buffer:
                return {
                    "exists": True,
                    "buffer_id": buffer.buffer_id,
                    "message_count": buffer.message_count,
                    "user_id": buffer.user_id,
                    "agent_id": buffer.agent_id,
                    "session_id": buffer.session_id,
                }
            return {"exists": False}


__all__ = [
    "SemanticBufferManager", 
    "SimpleBufferManager"
]
