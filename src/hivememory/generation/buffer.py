"""
HiveMemory - 对话缓冲器 (Conversation Buffer)

职责:
    累积对话消息，管理刷新逻辑。

特性:
    - 线程安全 (threading.Lock)
    - 支持手动刷新
    - 回调函数机制
    - 集成触发策略管理

作者: HiveMemory Team
版本: 0.1.0
"""

import logging
import threading
import time
from typing import List, Optional, Callable
from uuid import uuid4

from hivememory.core.models import ConversationMessage, MemoryAtom
from hivememory.generation.triggers import TriggerManager, create_default_trigger_manager

logger = logging.getLogger(__name__)


class ConversationBuffer:
    """
    对话缓冲器

    职责:
        1. 累积对话消息
        2. 判断触发条件
        3. 调用编排器处理
        4. 触发回调函数

    Examples:
        >>> from hivememory.generation import MemoryOrchestrator
        >>> orchestrator = MemoryOrchestrator(storage)
        >>> buffer = ConversationBuffer(
        ...     orchestrator=orchestrator,
        ...     user_id="user123",
        ...     agent_id="agent456"
        ... )
        >>> buffer.add_message("user", "帮我写快排")
        >>> buffer.add_message("assistant", "好的...")
        >>> buffer.flush()  # 手动触发
    """

    def __init__(
        self,
        orchestrator,  # MemoryOrchestrator
        user_id: str,
        agent_id: str,
        session_id: Optional[str] = None,
        trigger_manager: Optional[TriggerManager] = None,
        on_flush_callback: Optional[Callable[[List[ConversationMessage], List[MemoryAtom]], None]] = None,
    ):
        """
        初始化对话缓冲器

        Args:
            orchestrator: 记忆编排器实例
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID（可选，自动生成）
            trigger_manager: 触发管理器（可选，使用默认配置）
            on_flush_callback: 刷新回调函数（可选）
        """
        self.orchestrator = orchestrator
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id or str(uuid4())

        # 消息缓冲区
        self.messages: List[ConversationMessage] = []
        self._lock = threading.RLock()  # 使用可重入锁避免死锁

        # 触发管理
        self.trigger_manager = trigger_manager or create_default_trigger_manager()

        # 上下文信息
        self.last_trigger_time = time.time()
        self.last_message_time = time.time()

        # 回调函数
        self.on_flush_callback = on_flush_callback

        logger.info(
            f"ConversationBuffer 初始化完成 "
            f"(user={user_id}, agent={agent_id}, session={self.session_id})"
        )

    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到缓冲区

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容

        Examples:
            >>> buffer.add_message("user", "你好")
            >>> buffer.add_message("assistant", "你好！有什么可以帮助你的吗？")
        """
        with self._lock:
            # 创建消息对象
            message = ConversationMessage(
                role=role,
                content=content,
                user_id=self.user_id,
                session_id=self.session_id,
            )

            # 添加到缓冲区
            self.messages.append(message)
            self.last_message_time = time.time()

            logger.debug(f"添加消息: {role} - {content[:50]}...")

            # 检查触发条件
            self._check_and_trigger()

    def flush(self) -> List[MemoryAtom]:
        """
        手动刷新缓冲区

        触发记忆处理并清空缓冲区。

        Returns:
            List[MemoryAtom]: 提取的记忆列表

        Examples:
            >>> memories = buffer.flush()
            >>> print(f"提取了 {len(memories)} 条记忆")
        """
        with self._lock:
            if not self.messages:
                logger.debug("缓冲区为空，跳过处理")
                return []

            logger.info(f"手动刷新缓冲区 ({len(self.messages)} 条消息)")

            # 调用编排器处理
            memories = self._process_messages()

            # 清空缓冲区
            self.messages.clear()
            self.last_trigger_time = time.time()

            return memories

    def _check_and_trigger(self) -> None:
        """
        检查触发条件并自动刷新

        注意:
            该方法在 add_message 中自动调用。
        """
        context = {
            "last_trigger_time": self.last_trigger_time,
            "last_message_time": self.last_message_time,
        }

        should_trigger, reason = self.trigger_manager.should_trigger(
            messages=self.messages,
            context=context
        )

        if should_trigger:
            logger.info(f"自动触发处理 (原因: {reason.value})")
            self.flush()

    def _process_messages(self) -> List[MemoryAtom]:
        """
        处理消息并提取记忆

        Returns:
            List[MemoryAtom]: 提取的记忆列表
        """
        try:
            # 调用编排器
            memories = self.orchestrator.process(
                messages=list(self.messages),
                user_id=self.user_id,
                agent_id=self.agent_id,
            )

            # 触发回调
            if self.on_flush_callback:
                try:
                    self.on_flush_callback(self.messages.copy(), memories)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {e}")

            return memories

        except Exception as e:
            logger.error(f"消息处理失败: {e}", exc_info=True)
            return []

    def clear(self) -> None:
        """
        清空缓冲区（不处理）

        Examples:
            >>> buffer.clear()
        """
        with self._lock:
            self.messages.clear()
            logger.debug("缓冲区已清空")

    def get_message_count(self) -> int:
        """
        获取当前消息数量

        Returns:
            int: 消息数量
        """
        with self._lock:
            return len(self.messages)


__all__ = [
    "ConversationBuffer",
]
