"""
HiveMemory 感知层抽象接口

定义感知层各组件的抽象接口，遵循依赖倒置原则。

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict

from hivememory.core.models import FlushReason, ConversationMessage
from hivememory.perception.models import (
    StreamMessage,
    LogicalBlock,
    SemanticBuffer,
)


class TriggerStrategy(ABC):
    """
    触发策略接口

    职责:
        判断是否应该触发记忆处理。

    实现策略:
        - MessageCountTrigger: 消息数阈值
        - IdleTimeoutTrigger: 超时触发
        - SemanticBoundaryTrigger: 语义边界检测
    """

    @abstractmethod
    def should_trigger(
        self,
        messages: List[ConversationMessage],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[FlushReason]]:
        """
        判断是否应该触发

        Args:
            messages: 当前缓冲区的消息列表
            context: 上下文信息 (last_trigger_time, buffer_size 等)

        Returns:
            tuple[bool, FlushReason]: (是否触发, 触发原因)

        Examples:
            >>> trigger = MessageCountTrigger(threshold=5)
            >>> should, reason = trigger.should_trigger(messages, {})
            >>> if should:
            ...     print(f"触发原因: {reason.value}")
        """
        pass



class StreamParser(ABC):
    """
    流式解析器接口

    职责：
        - 将原始消息流转换为 StreamMessage
        - 构建 LogicalBlock
        - 判断是否需要创建新的 LogicalBlock

    Examples:
        >>> parser = UnifiedStreamParser()
        >>> message = parser.parse_message({"role": "user", "content": "hello"})
        >>> print(message.message_type)  # StreamMessageType.USER_QUERY
    """

    @abstractmethod
    def parse_message(self, raw_message: Any) -> StreamMessage:
        """
        解析原始消息为统一格式

        支持的格式：
            - LangChain: AIMessage, HumanMessage, ToolMessage
            - OpenAI: {"role": "...", "content": "...", "tool_calls": ...}
            - 简单文本: 默认为 user 消息

        Args:
            raw_message: 原始消息对象

        Returns:
            StreamMessage: 统一格式的流式消息
        """
        pass

    @abstractmethod
    def should_create_new_block(self, message: StreamMessage) -> bool:
        """
        判断是否需要创建新的 LogicalBlock

        规则：
            - User Query 永远开启新 Block
            - 其他消息延续当前 Block

        Args:
            message: 流式消息

        Returns:
            bool: 是否需要创建新 Block
        """
        pass


class SemanticAdsorber(ABC):
    """
    语义吸附器接口

    职责：
        - 计算语义相似度
        - 判断是吸附还是漂移
        - 更新话题核心向量

    判定流程：
        1. 短文本强吸附（< 50 tokens）
        2. Token 溢出检测
        3. 空闲超时检测
        4. 语义相似度判定（阈值 0.6）

    Examples:
        >>> adsorber = SemanticBoundaryAdsorber()
        >>> should_adsorb, reason = adsorber.should_adsorb(new_block, buffer)
        >>> if not should_adsorb:
        ...     print(f"触发 Flush: {reason}")
    """

    @abstractmethod
    def compute_similarity(
        self,
        anchor_text: str,
        topic_kernel: Optional[List[float]]
    ) -> float:
        """
        计算语义相似度

        Args:
            anchor_text: 锚点文本（新 Block 的 User Query）
            topic_kernel: 话题核心向量

        Returns:
            float: 相似度 (0-1)，如果 topic_kernel 为 None 返回 0
        """
        pass

    @abstractmethod
    def should_adsorb(
        self,
        new_block: LogicalBlock,
        buffer: SemanticBuffer
    ) -> Tuple[bool, Optional[FlushReason]]:
        """
        判断是否吸附

        Args:
            new_block: 新的 LogicalBlock
            buffer: 当前语义缓冲区

        Returns:
            Tuple[bool, Optional[FlushReason]]:
                - 是否吸附
                - 漂移原因（如果不吸附）
        """
        pass

    @abstractmethod
    def update_topic_kernel(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> None:
        """
        更新话题核心向量

        策略：指数移动平均 (EMA)
            new_kernel = alpha * new_vector + (1 - alpha) * old_kernel

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock
        """
        pass


class RelayController(ABC):
    """
    接力控制器接口

    职责：
        - 检测 Token 溢出
        - 生成中间态摘要
        - 维护跨 Block 的上下文连贯性

    Examples:
        >>> controller = TokenOverflowRelayController()
        >>> if controller.should_trigger_relay(buffer, new_block):
        ...     summary = controller.generate_summary(buffer.blocks)
    """

    @abstractmethod
    def should_trigger_relay(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> bool:
        """
        检测是否需要接力（Token 溢出）

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock

        Returns:
            bool: 是否需要触发接力
        """
        pass

    @abstractmethod
    def generate_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成中间态摘要

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 生成的摘要文本
        """
        pass


class BasePerceptionLayer(ABC):
    """
    感知层抽象基类

    定义所有类型的 PerceptionLayer 的统一接口。

    两种策略实现：
        - SimplePerceptionLayer: 简单缓冲策略（ConversationMessage + 三重触发）
        - SemanticFlowPerceptionLayer: 语义流策略（LogicalBlock + 语义吸附）

    Examples:
        >>> perception = SimplePerceptionLayer()
        >>> perception.add_message("user", "hello", "user1", "agent1", "sess1")
        >>> messages = perception.flush_buffer("user1", "agent1", "sess1")
    """

    @abstractmethod
    def add_message(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> None:
        """
        添加消息到感知层

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID
        """
        pass

    @abstractmethod
    def flush_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        reason: FlushReason = FlushReason.MANUAL
    ) -> List["ConversationMessage"]:
        """
        手动刷新缓冲区，返回消息列表

        注意：统一的返回类型，便于 orchestrator 处理

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID
            reason: 刷新原因

        Returns:
            List[ConversationMessage]: 缓冲区的消息列表
        """
        pass

    @abstractmethod
    def get_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        注意：返回类型取决于具体实现
            - SimplePerceptionLayer: 返回 SimpleBuffer
            - SemanticFlowPerceptionLayer: 返回 SemanticBuffer

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            缓冲区对象，不存在返回 None
        """
        pass

    @abstractmethod
    def clear_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> bool:
        """清理指定的缓冲区"""
        pass

    @abstractmethod
    def list_active_buffers(self) -> List[str]:
        """列出所有活跃的缓冲区 key"""
        pass

    @abstractmethod
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
        pass


__all__ = [
    "StreamParser",
    "SemanticAdsorber",
    "RelayController",
    "BasePerceptionLayer",  # 新增
]
