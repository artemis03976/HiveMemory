"""
HiveMemory - 简单感知层 (Simple Perception Layer)

职责:
    整合原有的 ConversationBuffer 与 TriggerManager 逻辑，作为低级感知层策略。

特性:
    - 三重触发机制（消息数、超时、语义边界）
    - 使用 StreamMessage 数据结构
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

from hivememory.core.models import FlushReason, Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import SimpleBuffer  # 从 models.py 导入
from hivememory.engines.perception.trigger_strategies import (
    TriggerManager,
    create_default_trigger_manager,
)

logger = logging.getLogger(__name__)


class SimplePerceptionLayer(BasePerceptionLayer):
    """
    简单感知层

    整合原有的 ConversationBuffer 与 TriggerManager 逻辑：
        - 三重触发机制（消息数、超时、语义边界）
        - 使用 StreamMessage 数据结构
        - 简单直接的消息累积，复写消息流

    Examples:
        >>> def on_flush(messages, reason):
        ...     print(f"Flush: {reason}, Messages: {len(messages)}")
        >>>
        >>> perception = SimplePerceptionLayer(on_flush_callback=on_flush)
        >>>
        >>> perception.add_message("user", "hello", identity)
        >>> messages = perception.flush_buffer(identity)
    """

    def __init__(
        self,
        trigger_manager: Optional[TriggerManager] = None,
        on_flush_callback: Optional[
            Callable[[List[StreamMessage], FlushReason], None]
        ] = None,
    ):
        """
        初始化简单感知层

        Args:
            trigger_manager: 触发管理器
            on_flush_callback: Flush 回调函数
        """
        super().__init__()

        self.trigger_manager = trigger_manager or create_default_trigger_manager()
        self.on_flush_callback = on_flush_callback

        # Buffer 池管理
        self._buffers: Dict[str, SimpleBuffer] = {}
        # 触发上下文追踪（每个 Buffer 独立的 timing state）
        self._trigger_context: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

        logger.info("SimplePerceptionLayer 初始化完成")

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
    ) -> Optional[List[StreamMessage]]:
        """
        检查并触发 Flush

        Args:
            buffer: SimpleBuffer 实例
            buffer_key: Buffer 唯一键

        Returns:
            Optional[List[StreamMessage]]: Flush 的消息列表，未触发返回 None
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
    ) -> List[StreamMessage]:
        """
        执行 Flush

        Args:
            buffer: SimpleBuffer 实例
            buffer_key: Buffer 唯一键
            reason: Flush 原因

        Returns:
            List[StreamMessage]: Flush 的消息列表
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
        identity: Identity,
        rewritten_query: Optional[str] = None,
        gateway_intent: Optional[str] = None,
        worth_saving: Optional[bool] = None,
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
            identity: 身份标识对象
            rewritten_query: Gateway 重写后的查询（可选）
            gateway_intent: Gateway 意图分类结果（可选）
            worth_saving: Gateway 价值判断（可选）
        """
        with self._lock:
            buffer_key = identity.buffer_key

            # 1. 获取或创建 Buffer
            buffer = self.get_buffer(identity)
            if not buffer:
                logger.debug(f"Buffer 创建失败: {buffer_key}")
                return

            # 2. 创建消息并添加
            try:
                msg_type = StreamMessageType(role)
            except ValueError:
                msg_type = StreamMessageType.ASSISTANT
            
            message = StreamMessage(
                message_type=msg_type,
                content=content,
                identity=Identity(
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    session_id=identity.session_id or buffer_key.split(":")[-1]
                ),
                rewritten_query=rewritten_query,
                gateway_intent=gateway_intent,
                worth_saving=worth_saving,
            )
            
            buffer.add_message(message)

            logger.debug(f"添加消息: {role} - {content[:50]}...")

            # 3. 检查触发条件
            self._check_and_flush(buffer, buffer_key)

            # 4. 更新触发上下文（在检查之后更新时间）
            trigger_context = self._get_trigger_context(buffer_key)
            trigger_context["last_message_time"] = time.time()

    def flush_buffer(
        self,
        identity: Identity,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> List[StreamMessage]:
        """
        手动刷新 Buffer

        Args:
            identity: 身份标识对象
            reason: 刷新原因

        Returns:
            List[StreamMessage]: 被 Flush 的消息列表，如果 Buffer 不存在或为空则返回空列表
        """
        with self._lock:
            buffer_key = identity.buffer_key
            buffer = self.get_buffer(identity)
            if not buffer:
                logger.debug(f"Buffer 不存在: {buffer_key}")
                return []

            return self._flush(buffer, buffer_key, reason)

    def get_buffer(
        self,
        identity: Identity,
    ) -> Optional[SimpleBuffer]:
        """
        获取缓冲区对象

        Args:
            identity: 身份标识对象

        Returns:
            SimpleBuffer: 缓冲区对象，不存在返回 None
        """
        key = identity.buffer_key

        with self._lock:
            if key not in self._buffers:
                # 创建新 Buffer
                self._buffers[key] = SimpleBuffer(
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    session_id=identity.session_id or key.split(":")[-1],
                )
                logger.debug(f"创建新 Buffer: {key}")

            return self._buffers.get(key)

    def clear_buffer(
        self,
        identity: Identity,
    ) -> bool:
        """
        清理指定的 Buffer

        Args:
            identity: 身份标识对象

        Returns:
            bool: 是否成功清理
        """
        with self._lock:
            buffer_key = identity.buffer_key
            buffer = self.get_buffer(identity)
            if buffer:
                buffer.clear()

                # 清理触发上下文
                if buffer_key in self._trigger_context:
                    del self._trigger_context[buffer_key]

                logger.info(f"清理 Buffer: {buffer_key}")
                return True
            else:
                logger.debug(f"Buffer 不存在，无需清理: {buffer_key}")
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
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            identity: 身份标识对象

        Returns:
            Dict: 缓冲区信息字典
        """
        buffer = self.get_buffer(identity)
        if buffer:
            return {
                "exists": True,
                "mode": "simple",
                "message_count": buffer.message_count,
                "user_id": identity.user_id,
                "agent_id": identity.agent_id,
                "session_id": identity.session_id,
            }
        else:
            return {
                "exists": False,
                "mode": "simple",
                "message_count": 0,
                "user_id": identity.user_id,
                "agent_id": identity.agent_id,
                "session_id": identity.session_id,
            }


__all__ = [
    "SimplePerceptionLayer",
]
