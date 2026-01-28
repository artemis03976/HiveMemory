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
from hivememory.core.models import Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import SimpleBuffer, FlushReason
from hivememory.engines.perception.buffer_manager import SimpleBufferManager
from hivememory.patchouli.config import SimplePerceptionConfig
from hivememory.engines.perception.trigger_strategies import (
    TriggerManager,
)

logger = logging.getLogger(__name__)


class SimplePerceptionLayer(BasePerceptionLayer):
    """
    简单感知层

    整合原有的 ConversationBuffer 与 TriggerManager 逻辑：
        - 三重触发机制（消息数、超时、语义边界）
        - 简单直接的消息累积，复写消息流

    Examples:
        >>> def on_flush(messages, reason):
        ...     print(f"Flush: {reason}, Messages: {len(messages)}")
        >>>
        >>> config = SimplePerceptionConfig(message_threshold=6)
        >>> perception = SimplePerceptionLayer(config=config, on_flush_callback=on_flush)
        >>>
        >>> perception.add_message("user", "hello", identity)
        >>> messages = perception.flush_buffer(identity)
    """

    def __init__(
        self,
        config: SimplePerceptionConfig,
        trigger_manager: TriggerManager,
        on_flush_callback: Optional[
            Callable[[List[StreamMessage], FlushReason], None]
        ] = None,
    ):
        """
        初始化简单感知层

        Args:
            config: SimplePerceptionConfig 配置对象
            trigger_manager: 触发管理器
            on_flush_callback: Flush 回调函数
        """
        super().__init__()
        
        self.config = config
        self.trigger_manager = trigger_manager
        self.on_flush_callback = on_flush_callback

        # BufferManager 管理
        self._buffer_manager = SimpleBufferManager()

        logger.info("SimplePerceptionLayer 初始化完成")

    # ========== 内部方法 ==========

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

        # 检查是否触发
        should_trigger, flush_reason = self.trigger_manager.should_trigger(
            messages=buffer.messages,
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

        # 调用回调
        if self.on_flush_callback:
            try:
                self.on_flush_callback(messages_to_process, reason)
            except Exception as e:
                logger.error(f"Flush 回调执行失败: {e}", exc_info=True)

        return messages_to_process

    # ========== BasePerceptionLayer 接口实现 ==========

    def perceive(
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
            1. 创建消息并添加到 Buffer
            2. 检查触发条件
            3. 如果触发，执行 Flush

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            identity: 身份标识对象
            rewritten_query: Gateway 重写后的查询（可选）
            gateway_intent: Gateway 意图分类结果（可选）
            worth_saving: Gateway 价值判断（可选）
        """
        buffer_key = identity.buffer_key

        # 1. 创建消息
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
            )
        )
        
        # 2. 添加到 Buffer (通过 Manager)
        self._buffer_manager.add_message(identity, message)

        logger.debug(f"添加消息: {role} - {content[:50]}...")

        # 3. 检查触发条件
        # 获取 buffer 对象用于检查
        buffer = self._buffer_manager.get_buffer(identity)
        self._check_and_flush(buffer, buffer_key)

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
        buffer_key = identity.buffer_key
        buffer = self._buffer_manager.get_buffer(identity)
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
        return self._buffer_manager.get_buffer(identity)

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
        return self._buffer_manager.clear_buffer(identity)

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表
        """
        return self._buffer_manager.list_active_buffers()

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
        info = self._buffer_manager.get_buffer_info(identity)
        info["mode"] = "simple"
        info["identity"] = identity
        return info


__all__ = [
    "SimplePerceptionLayer",
]
