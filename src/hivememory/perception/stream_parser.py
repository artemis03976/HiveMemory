"""
HiveMemory 统一流式解析器

抹平不同 Agent 框架的消息格式差异，封装为标准化的 StreamMessage。

支持格式：
    - LangChain: AIMessage, HumanMessage, ToolMessage
    - OpenAI: {"role": "...", "content": "...", "tool_calls": ...}
    - 简单文本: 默认为 user 消息

参考: PROJECT.md 4.1.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from typing import Any, Dict, Optional

from hivememory.perception.interfaces import StreamParser
from hivememory.perception.models import (
    StreamMessage,
    StreamMessageType,
)

logger = logging.getLogger(__name__)


class UnifiedStreamParser(StreamParser):
    """
    统一流式解析器

    职责：
        - 解析 LangChain 消息格式
        - 解析 OpenAI 格式
        - 解析简单文本
        - 构建 LogicalBlock

    Examples:
        >>> parser = UnifiedStreamParser()
        >>>
        >>> # 解析简单文本
        >>> msg = parser.parse_message("你好")
        >>> print(msg.message_type)  # StreamMessageType.USER_QUERY
        >>>
        >>> # 解析 OpenAI 格式
        >>> msg = parser.parse_message({"role": "user", "content": "hello"})
        >>> print(msg.message_type)  # StreamMessageType.USER_QUERY
    """

    def __init__(self, enable_thought_extraction: bool = False):
        """
        初始化流式解析器

        Args:
            enable_thought_extraction: 是否提取思考过程（Claude thinking）
        """
        self.enable_thought_extraction = enable_thought_extraction
        logger.debug("UnifiedStreamParser 初始化完成")

    def parse_message(self, raw_message: Any) -> StreamMessage:
        """
        解析原始消息

        支持的格式：
            - LangChain: AIMessage, HumanMessage, ToolMessage
            - OpenAI: {"role": "...", "content": "...", "tool_calls": ...}
            - 简单文本: 默认为 user 消息

        Args:
            raw_message: 原始消息对象

        Returns:
            StreamMessage: 统一格式的流式消息
        """
        # LangChain 格式
        if self._is_langchain_message(raw_message):
            return self._parse_langchain_message(raw_message)

        # OpenAI 格式字典
        elif isinstance(raw_message, dict):
            return self._parse_openai_format(raw_message)

        # 简单文本（默认为用户消息）
        elif isinstance(raw_message, str):
            return StreamMessage(
                message_type=StreamMessageType.USER_QUERY,
                content=raw_message
            )

        else:
            raise ValueError(
                f"不支持的消息类型: {type(raw_message)}。"
                f"支持的类型: LangChain Message, dict, str"
            )

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
        return message.message_type == StreamMessageType.USER_QUERY

    def _is_langchain_message(self, message: Any) -> bool:
        """检查是否为 LangChain 消息"""
        try:
            from langchain_core.messages import BaseMessage
            return isinstance(message, BaseMessage)
        except ImportError:
            return False

    def _parse_langchain_message(self, message: Any) -> StreamMessage:
        """解析 LangChain 消息"""
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                ToolMessage,
                SystemMessage,
            )
        except ImportError:
            raise ImportError(
                "langchain-core 未安装。如需 LangChain 支持，"
                "请运行: pip install langchain-core"
            )

        if isinstance(message, HumanMessage):
            return StreamMessage(
                message_type=StreamMessageType.USER_QUERY,
                content=message.content
            )

        elif isinstance(message, AIMessage):
            return self._parse_ai_message(message)

        elif isinstance(message, ToolMessage):
            return StreamMessage(
                message_type=StreamMessageType.TOOL_OUTPUT,
                content=message.content,
                tool_result=message.content
            )

        elif isinstance(message, SystemMessage):
            return StreamMessage(
                message_type=StreamMessageType.SYSTEM_MESSAGE,
                content=message.content
            )

        else:
            # 默认作为 assistant 消息处理
            return StreamMessage(
                message_type=StreamMessageType.ASSISTANT_MESSAGE,
                content=str(message.content) if hasattr(message, 'content') else str(message)
            )

    def _parse_ai_message(self, ai_message: Any) -> StreamMessage:
        """解析 LangChain AI 消息"""
        # 检查是否有工具调用
        tool_calls = getattr(ai_message, 'tool_calls', None)

        if tool_calls and len(tool_calls) > 0:
            tool_call = tool_calls[0]
            return StreamMessage(
                message_type=StreamMessageType.TOOL_CALL,
                content=ai_message.content or f"调用工具: {tool_call.get('name', 'unknown')}",
                tool_name=tool_call.get('name'),
                tool_args=tool_call.get('args', {})
            )

        # 普通 AI 响应
        return StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content=ai_message.content or ""
        )

    def _parse_openai_format(self, message: Dict[str, Any]) -> StreamMessage:
        """解析 OpenAI 格式消息"""
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            return StreamMessage(
                message_type=StreamMessageType.USER_QUERY,
                content=content
            )

        elif role == "assistant":
            # 检查 tool_calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                function = tool_call.get("function", {})
                import json
                args = function.get("arguments", "{}")
                try:
                    args_dict = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args_dict = {}

                return StreamMessage(
                    message_type=StreamMessageType.TOOL_CALL,
                    content=content or f"调用工具: {function.get('name', 'unknown')}",
                    tool_name=function.get('name'),
                    tool_args=args_dict
                )

            return StreamMessage(
                message_type=StreamMessageType.ASSISTANT_MESSAGE,
                content=content or ""
            )

        elif role == "tool":
            return StreamMessage(
                message_type=StreamMessageType.TOOL_OUTPUT,
                content=content,
                tool_result=content
            )

        elif role == "system":
            return StreamMessage(
                message_type=StreamMessageType.SYSTEM_MESSAGE,
                content=content
            )

        else:
            # 默认作为 assistant 消息处理
            return StreamMessage(
                message_type=StreamMessageType.ASSISTANT_MESSAGE,
                content=content
            )

    def parse_conversation_messages(
        self,
        messages: list,
        session_id: str,
        user_id: str
    ) -> list:
        """
        批量解析对话消息列表

        Args:
            messages: 原始消息列表
            session_id: 会话ID
            user_id: 用户ID

        Returns:
            List[ConversationMessage]: 转换后的消息列表
        """
        from hivememory.generation.models import ConversationMessage

        result = []
        for msg in messages:
            stream_msg = self.parse_message(msg)
            result.append(stream_msg.to_conversation_message(session_id, user_id))
        return result


__all__ = [
    "UnifiedStreamParser",
]
