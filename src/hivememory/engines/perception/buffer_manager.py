"""
HiveMemory Buffer Manager

纯状态管理器，管理 buffer 池和 builder 池的 CRUD 操作。

职责:
    - 管理 buffer 池 (Dict[str, SemanticBuffer])
    - 管理 builder 池 (Dict[str, LogicalBlockBuilder])
    - 提供 CRUD 操作接口

不负责:
    - Flush 条件检测（由 PerceptionLayer 编排）
    - Flush 执行（由 PerceptionLayer 编排）
    - 话题核心更新（由 PerceptionLayer 编排）

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 3.0.0
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.block_builder import LogicalBlockBuilder
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
    SimpleBuffer,
)

logger = logging.getLogger(__name__)


class SemanticBufferManager:
    """
    语义 Buffer 管理器 - 纯状态容器

    管理语义缓冲区和逻辑块构建器的生命周期。
    仅提供 CRUD 操作，不包含业务逻辑。

    职责:
        - 管理 buffer 池 (Dict[str, SemanticBuffer])
        - 管理 builder 池 (Dict[str, LogicalBlockBuilder])
        - 提供线程安全的 CRUD 操作

    Examples:
        >>> manager = SemanticBufferManager()
        >>> buffer = manager.get_buffer(identity)
        >>> manager.add_block_to_buffer(identity, block)
    """

    def __init__(self) -> None:
        """初始化 SemanticBufferManager"""
        # Buffer 池: key -> SemanticBuffer
        self._buffers: Dict[str, SemanticBuffer] = {}

        # Builder 池: key -> LogicalBlockBuilder
        self._builders: Dict[str, LogicalBlockBuilder] = {}

        # 线程安全
        self._lock = threading.RLock()

        logger.info("SemanticBufferManager 初始化完成")

    # ========== Buffer CRUD ==========

    def get_buffer(self, identity: Identity) -> SemanticBuffer:
        """
        获取或创建 buffer

        Args:
            identity: 用于查找 buffer 的身份标识

        Returns:
            该身份对应的 SemanticBuffer
        """
        key = identity.buffer_key

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SemanticBuffer(
                    identity=identity,
                )
                logger.debug(f"创建新 buffer: {key}")

            return self._buffers[key]

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
        列出所有活跃的 buffer keys

        Returns:
            buffer key 列表
        """
        with self._lock:
            return list(self._buffers.keys())

    # ========== Builder CRUD ==========

    def get_builder(self, identity: Identity) -> LogicalBlockBuilder:
        """
        获取或创建 builder

        Args:
            identity: 用于查找 builder 的身份标识

        Returns:
            该身份对应的 LogicalBlockBuilder
        """
        key = identity.buffer_key

        with self._lock:
            if key not in self._builders:
                self._builders[key] = LogicalBlockBuilder()
                logger.debug(f"创建新 builder: {key}")

            return self._builders[key]

    def reset_builder(self, identity: Identity) -> None:
        """
        重置 builder 到初始状态

        Args:
            identity: 身份标识
        """
        with self._lock:
            key = identity.buffer_key
            if key in self._builders:
                self._builders[key]._reset()
                logger.debug(f"重置 builder: {key}")

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
            builder = self._builders.get(identity.buffer_key)

            if buffer:
                return {
                    "exists": True,
                    "buffer_id": buffer.buffer_id,
                    "block_count": len(buffer.blocks),
                    "total_tokens": buffer.total_tokens,
                    "state": buffer.state.value if hasattr(buffer.state, 'value') else buffer.state,
                    "has_building_block": builder.is_started if builder else False,
                    "building_block_complete": builder.is_complete if builder else False,
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
