"""
HiveMemory - 简单感知层 (Simple Perception Layer)

职责:
    整合原有的 ConversationBuffer 与 TriggerManager 逻辑，作为低级感知层策略。

特性:
    - 三重触发机制（消息数、超时、语义边界）
    - 使用 ConversationMessage 数据结构
    - 简单直接的消息累积
    - 线程安全

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
import threading
import time
from typing import List, Optional, Dict, Any, Callable

from hivememory.core.models import ConversationMessage, FlushReason
from hivememory.perception.interfaces import BasePerceptionLayer
from hivememory.perception.models import SimpleBuffer  # 从 models.py 导入
from hivememory.perception.trigger_strategies import (
    TriggerManager,
    create_default_trigger_manager,
)

logger = logging.getLogger(__name__)


class SimplePerceptionLayer(BasePerceptionLayer):
    """
    简单感知层

    整合原有的 ConversationBuffer 与 TriggerManager 逻辑：
        - 三重触发机制（消息数、超时、语义边界）
        - 使用 ConversationMessage 数据结构
        - 简单直接的消息累积，复写消息流

    Examples:
        >>> def on_flush(messages, reason):
        ...     print(f"Flush: {reason}, Messages: {len(messages)}")
        >>>
        >>> perception = SimplePerceptionLayer(on_flush_callback=on_flush)
        >>>
        >>> perception.add_message("user", "hello", "user1", "agent1", "sess1")
        >>> messages = perception.flush_buffer("user1", "agent1", "sess1")
    """

    def __init__(
        self,
        trigger_manager: Optional[TriggerManager] = None,
        on_flush_callback: Optional[
            Callable[[List[ConversationMessage], FlushReason], None]
        ] = None,
    ):
        """
        初始化简单感知层

        Args:
            trigger_manager: 触发管理器
            on_flush_callback: Flush 回调函数
        """
        self.trigger_manager = trigger_manager or create_default_trigger_manager()
        self.on_flush_callback = on_flush_callback

        # Buffer 池管理
        self._buffers: Dict[str, SimpleBuffer] = {}
        # 触发上下文追踪（每个 Buffer 独立的 timing state）
        self._trigger_context: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

        logger.info("SimplePerceptionLayer 初始化完成")

    def _get_buffer_key(self, user_id: str, agent_id: str, session_id: str) -> str:
        """生成缓冲区唯一键"""
        return f"{user_id}:{agent_id}:{session_id}"

    # ========== 内部方法 ==========

    def _get_trigger_context(self, buffer_key: str) -> Dict[str, float]:
        """
        获取或创建触发上下文

        Args:
            buffer_key: Buffer 唯一键

        Returns:
            Dict: 触发上下文 {
                "last_trigger_time": float,
                "last_message_time": float
            }
        """
        current_time = time.time()

        if buffer_key not in self._trigger_context:
            self._trigger_context[buffer_key] = {
                "last_trigger_time": current_time,
                "last_message_time": current_time,
            }

        return self._trigger_context[buffer_key]

    def _check_and_flush(
        self,
        buffer: SimpleBuffer,
        buffer_key: str
    ) -> Optional[List[ConversationMessage]]:
        """
        检查并触发 Flush

        Args:
            buffer: SimpleBuffer 实例
            buffer_key: Buffer 唯一键

        Returns:
            Optional[List[ConversationMessage]]: Flush 的消息列表，未触发返回 None
        """
        if not buffer.messages:
            return None

        # 获取触发上下文
        trigger_context = self._get_trigger_context(buffer_key)

        # 检查是否触发
        should_trigger, flush_reason = self.trigger_manager.should_trigger(
            messages=buffer.messages,
            context=trigger_context,
        )

        if should_trigger:
            # 执行 Flush
            return self._flush(buffer, buffer_key, flush_reason)

        return None

    def _flush(
        self,
        buffer: SimpleBuffer,
        buffer_key: str,
        reason: FlushReason,
    ) -> List[ConversationMessage]:
        """
        执行 Flush

        Args:
            buffer: SimpleBuffer 实例
            buffer_key: Buffer 唯一键
            reason: Flush 原因

        Returns:
            List[ConversationMessage]: Flush 的消息列表
        """
        messages_to_process = buffer.messages.copy()

        logger.info(
            f"触发 Flush: {buffer.buffer_id}, "
            f"原因: {reason.value}, "
            f"消息数量: {len(messages_to_process)}"
        )

        # 清空 Buffer
        buffer.clear()

        # 更新触发上下文
        trigger_context = self._get_trigger_context(buffer_key)
        trigger_context["last_trigger_time"] = time.time()

        # 调用回调
        if self.on_flush_callback:
            try:
                self.on_flush_callback(messages_to_process, reason)
            except Exception as e:
                logger.error(f"Flush 回调执行失败: {e}", exc_info=True)

        return messages_to_process

    # ========== BasePerceptionLayer 接口实现 ==========

    def add_message(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        """
        添加消息到感知层

        处理流程（匹配 SemanticFlowPerceptionLayer）：
            1. 获取或创建 SimpleBuffer
            2. 创建消息并添加到 Buffer
            3. 更新触发上下文
            4. 检查触发条件
            5. 如果触发，执行 Flush

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID
        """
        with self._lock:
            # 1. 获取或创建 Buffer
            buffer = self.get_buffer(user_id, agent_id, session_id)
            if not buffer:
                logger.debug(f"Buffer 创建失败: {user_id}:{agent_id}:{session_id}")
                return

            # 2. 创建消息并添加
            message = ConversationMessage(
                role=role,
                content=content,
                user_id=user_id,
                session_id=session_id,
            )
            buffer.add_message(message)

            logger.debug(f"添加消息: {role} - {content[:50]}...")

            # 3. 检查触发条件
            buffer_key = self._get_buffer_key(user_id, agent_id, session_id)
            self._check_and_flush(buffer, buffer_key)

            # 4. 更新触发上下文（在检查之后更新时间）
            trigger_context = self._get_trigger_context(buffer_key)
            trigger_context["last_message_time"] = time.time()

    def flush_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> None:
        """
        手动刷新 Buffer

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID
            reason: 刷新原因

        Returns:
            None
        """
        with self._lock:
            buffer = self.get_buffer(user_id, agent_id, session_id)
            if not buffer:
                logger.debug(f"Buffer 不存在: {user_id}:{agent_id}:{session_id}")
                return

            buffer_key = self._get_buffer_key(user_id, agent_id, session_id)
            self._flush(buffer, buffer_key, reason)

    def get_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> Optional[SimpleBuffer]:
        """
        获取缓冲区对象

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            SimpleBuffer: 缓冲区对象，不存在返回 None
        """
        key = self._get_buffer_key(user_id, agent_id, session_id)

        with self._lock:
            if key not in self._buffers:
                # 创建新 Buffer
                self._buffers[key] = SimpleBuffer(
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                )
                logger.debug(f"创建新 Buffer: {key}")

            return self._buffers.get(key)

    def clear_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> bool:
        """
        清理指定的 Buffer

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            bool: 是否成功清理

        Examples:
            >>> success = perception.clear_buffer("user1", "agent1", "sess1")
            >>> print(f"清理{'成功' if success else '失败'}")
        """
        with self._lock:
            buffer = self.get_buffer(user_id, agent_id, session_id)
            if buffer:
                buffer.clear()

                # 清理触发上下文
                buffer_key = self._get_buffer_key(user_id, agent_id, session_id)
                if buffer_key in self._trigger_context:
                    del self._trigger_context[buffer_key]

                logger.info(f"清理 Buffer: {user_id}:{agent_id}:{session_id}")
                return True
            else:
                logger.debug(f"Buffer 不存在，无需清理: {user_id}:{agent_id}:{session_id}")
                return False

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表
        """
        with self._lock:
            return list(self._buffers.keys())

    def get_buffer_info(
        self,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            Dict: 缓冲区信息字典
        """
        buffer = self.get_buffer(user_id, agent_id, session_id)
        if buffer:
            return {
                "exists": True,
                "mode": "simple",
                "message_count": buffer.message_count,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
            }
        else:
            return {
                "exists": False,
                "mode": "simple",
                "message_count": 0,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
            }


__all__ = [
    "SimpleBuffer",
    "SimplePerceptionLayer",
]
