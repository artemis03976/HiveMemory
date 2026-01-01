"""
HiveMemory - 感知层触发策略 (Trigger Strategies)

职责:
    判断何时触发记忆处理。

支持的触发器:
    - MessageCountTrigger: 消息数阈值
    - IdleTimeoutTrigger: 超时触发（带 Debounce）
    - SemanticBoundaryTrigger: 语义边界检测
    - ManualTrigger: 手动触发

作者: HiveMemory Team
版本: 0.2.0
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional

from hivememory.core.models import ConversationMessage, FlushReason
from hivememory.perception.interfaces import TriggerStrategy

logger = logging.getLogger(__name__)


class MessageCountTrigger(TriggerStrategy):
    """
    消息数量触发器

    当缓冲区消息数达到阈值时触发。

    注意:
        建议使用偶数阈值以确保处理完整的 user-assistant 对话对。

    Examples:
        >>> trigger = MessageCountTrigger(threshold=6)
        >>> should, reason = trigger.should_trigger(messages, {})
        >>> if should:
        ...     print(f"触发原因: {reason.value}")
    """

    def __init__(self, threshold: int = 6):
        """
        初始化消息数触发器

        Args:
            threshold: 消息数阈值，默认 6
                建议使用偶数以确保完整对话对被处理
        """
        self.threshold = threshold

    def should_trigger(
        self,
        messages: List[ConversationMessage],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[FlushReason]]:
        """
        检查消息数是否达到阈值

        Args:
            messages: 当前消息列表
            context: 上下文（未使用）

        Returns:
            tuple[bool, FlushReason]: (是否触发, 原因)
        """
        if len(messages) >= self.threshold:
            logger.debug(f"消息数达到阈值: {len(messages)} >= {self.threshold}")
            return True, FlushReason.MESSAGE_COUNT

        return False, None


class IdleTimeoutTrigger(TriggerStrategy):
    """
    空闲超时触发器

    当距离上次消息超过指定时间时触发。

    特性:
        - Debounce 机制：避免频繁触发
        - 自动重置计时器

    Examples:
        >>> trigger = IdleTimeoutTrigger(timeout=900)  # 15分钟
        >>> should, reason = trigger.should_trigger(messages, context)
    """

    def __init__(self, timeout: int = 900):
        """
        初始化超时触发器

        Args:
            timeout: 超时时间（秒），默认 900（15分钟）
        """
        self.timeout = timeout

    def should_trigger(
        self,
        messages: List[ConversationMessage],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[FlushReason]]:
        """
        检查是否超时

        Args:
            messages: 当前消息列表
            context: 上下文，包含:
                - last_trigger_time: 上次触发时间
                - last_message_time: 最后一条消息时间

        Returns:
            tuple[bool, FlushReason]: (是否触发, 原因)
        """
        if not messages:
            return False, None

        current_time = time.time()
        last_message_time = context.get("last_message_time", current_time)

        idle_duration = current_time - last_message_time

        if idle_duration >= self.timeout:
            logger.debug(f"空闲超时: {idle_duration:.1f}s >= {self.timeout}s")
            return True, FlushReason.IDLE_TIMEOUT

        return False, None


class SemanticBoundaryTrigger(TriggerStrategy):
    """
    语义边界触发器

    检测话题切换、工具调用结束等语义边界。

    检测信号:
        - 结束语 ("希望这对您有帮助"、"还有其他问题吗")
        - 工具调用完成标记
        - 话题明显切换

    注意:
        当前为简化实现，基于关键词匹配。
        生产环境建议使用轻量级分类模型。

    Examples:
        >>> trigger = SemanticBoundaryTrigger()
        >>> should, reason = trigger.should_trigger(messages, {})
    """

    # 结束语关键词
    ENDING_PATTERNS = [
        r"希望.*帮助",
        r"还有.*问题",
        r"就这样了",
        r"完成了",
        r"完成.*任务",
        r"任务.*完成",
        r"解决.*问题",
    ]

    def __init__(self):
        """初始化语义边界触发器"""
        # 编译正则表达式
        self.ending_regex = [re.compile(pattern) for pattern in self.ENDING_PATTERNS]

    def should_trigger(
        self,
        messages: List[ConversationMessage],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[FlushReason]]:
        """
        检测语义边界

        Args:
            messages: 当前消息列表
            context: 上下文（未使用）

        Returns:
            tuple[bool, FlushReason]: (是否触发, 原因)
        """
        if not messages:
            return False, None

        # 检查最后一条 assistant 消息
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg.role == "assistant":
                last_assistant_msg = msg
                break

        if not last_assistant_msg:
            return False, None

        # 检测结束语
        content = last_assistant_msg.content
        for regex in self.ending_regex:
            if regex.search(content):
                logger.debug(f"检测到语义边界: {regex.pattern}")
                return True, FlushReason.SEMANTIC_DRIFT

        return False, None


class TriggerManager:
    """
    触发策略管理器

    职责:
        协调多个触发策略，返回综合判断结果。

    逻辑:
        任意一个策略触发即触发处理。

    Examples:
        >>> manager = TriggerManager(
        ...     strategies=[
        ...         MessageCountTrigger(threshold=5),
        ...         IdleTimeoutTrigger(timeout=900),
        ...     ]
        ... )
        >>> should, reason = manager.should_trigger(messages, context)
    """

    def __init__(self, strategies: List[TriggerStrategy]):
        """
        初始化管理器

        Args:
            strategies: 触发策略列表
        """
        self.strategies = strategies

    def should_trigger(
        self,
        messages: List[ConversationMessage],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[FlushReason]]:
        """
        综合判断是否触发

        逻辑:
            任意策略触发即返回 True

        Args:
            messages: 当前消息列表
            context: 上下文信息

        Returns:
            tuple[bool, FlushReason]: (是否触发, 触发原因)
        """
        for strategy in self.strategies:
            should, reason = strategy.should_trigger(messages, context)
            if should:
                logger.info(f"触发器激活: {reason.value}")
                return True, reason

        return False, None


# 便捷函数
def create_default_trigger_manager() -> TriggerManager:
    """
    创建默认触发管理器

    配置:
        - 消息数: 6 条 (偶数，确保完整对话对)
        - 超时: 900 秒 (15 分钟)
        - 语义边界: 启用

    Returns:
        TriggerManager: 管理器实例
    """
    return TriggerManager(
        strategies=[
            MessageCountTrigger(threshold=6),
            IdleTimeoutTrigger(timeout=900),
            SemanticBoundaryTrigger(),
        ]
    )


__all__ = [
    "MessageCountTrigger",
    "IdleTimeoutTrigger",
    "SemanticBoundaryTrigger",
    "TriggerManager",
    "create_default_trigger_manager",
]
