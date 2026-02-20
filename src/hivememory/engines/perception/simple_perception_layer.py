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
from hivememory.engines.perception.models import SimpleBuffer, FlushReason, InteractionPayload
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
        write_focus=None,
        update_focus=None,
    ) -> List[StreamMessage]:
        """
        执行 Flush

        Args:
            buffer: SimpleBuffer 实例
            buffer_key: Buffer 唯一键
            reason: Flush 原因
            write_focus: WRITE 指令控制信号 (仅 MTP_WRITE 时传入)
            update_focus: UPDATE 指令控制信号 (仅 MTP_UPDATE 时传入)

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
                self.on_flush_callback(
                    messages_to_process, reason,
                    write_focus=write_focus,
                    update_focus=update_focus,
                )
            except Exception as e:
                logger.error(f"Flush 回调执行失败: {e}", exc_info=True)

        return messages_to_process

    # ========== BasePerceptionLayer 接口实现 ==========

    def ingest_payload(self, payload: InteractionPayload) -> None:
        """
        摄入 Kernel 递归循环的完整交互载荷

        处理流程:
            1. MTPLogParser 清洗 assistant 文本
            2. 将 user + clean assistant 添加到 buffer
            3. 信号检查: 携带 write_focus/update_focus → 立即 flush
            4. 否则走常规触发检查

        Args:
            payload: Kernel → Perception 的原子传输包
        """
        from hivememory.patchouli.protocol.mtp_log_parser import MTPLogParser

        clean_text, _ = MTPLogParser.parse(payload.assistant_message)
        buffer_key = payload.identity.buffer_key

        # 添加 user message
        user_msg = StreamMessage(
            message_type=StreamMessageType.USER,
            content=payload.user_message,
            identity=payload.identity,
        )
        self._buffer_manager.add_message(payload.identity, user_msg)

        # 添加 clean assistant message
        assistant_msg = StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content=clean_text,
            identity=payload.identity,
        )
        self._buffer_manager.add_message(payload.identity, assistant_msg)

        # 信号检查: WRITE/UPDATE → 立即 flush
        if payload.write_focus is not None or payload.update_focus is not None:
            reason = (
                FlushReason.MTP_WRITE if payload.write_focus is not None
                else FlushReason.MTP_UPDATE
            )
            buffer = self._buffer_manager.get_buffer(payload.identity)
            if buffer:
                self._flush(
                    buffer, buffer_key, reason,
                    write_focus=payload.write_focus,
                    update_focus=payload.update_focus,
                )
            return

        # 常规触发检查
        buffer = self._buffer_manager.get_buffer(payload.identity)
        if buffer:
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
