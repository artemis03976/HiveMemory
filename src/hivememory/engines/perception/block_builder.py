"""
HiveMemory LogicalBlock Builder

封装 LogicalBlock 构建的状态机逻辑，遵循 Builder 模式。

职责:
    - 管理 Block 构建的状态转换
    - 管理 Triplet 构建
    - 判断是否需要创建新 Block（基于闭合状态）

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 2.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.engines.perception.models import LogicalBlock, Triplet

logger = logging.getLogger(__name__)


class LogicalBlockBuilder:
    """
    LogicalBlock 构建器

    封装状态机逻辑:
        1. IDLE -> 收到 User Message -> 开始构建新 Block
        2. BUILDING -> 收到 Thought/Tool Call/Tool Output -> 添加到执行链
        3. BUILDING -> 收到 Assistant Message -> Block 闭合
        4. Block 闭合 (is_complete = True)

    Examples:
        >>> builder = LogicalBlockBuilder()
        >>> builder.start(rewritten_query="What is Python?")
        >>> builder.add_message(user_message)
        >>> builder.add_message(thought_message)
        >>> builder.add_message(tool_call_message)
        >>> builder.add_message(tool_output_message)
        >>> builder.add_message(assistant_message)
        >>> block = builder.build()
        >>> assert block.is_complete
    """

    def __init__(self) -> None:
        """初始化构建器为空闲状态"""
        self._reset()

    def _reset(self) -> None:
        """重置构建器到初始状态"""
        self._block_id: str = str(uuid4())
        self._user_block: Optional[StreamMessage] = None
        self._execution_chain: List[Triplet] = []
        self._response_block: Optional[StreamMessage] = None
        self._created_at: float = datetime.now().timestamp()
        self._rewritten_query: Optional[str] = None
        self._gateway_intent: Optional[str] = None
        self._worth_saving: Optional[bool] = None

    def start(
        self,
        rewritten_query: Optional[str] = None,
        gateway_intent: Optional[str] = None,
        worth_saving: Optional[bool] = None,
    ) -> LogicalBlockBuilder:
        """
        开始构建新的 Block

        Args:
            rewritten_query: Gateway 重写后的查询（指代消解）
            gateway_intent: Gateway 意图分类
            worth_saving: Gateway 记忆价值信号

        Returns:
            Self，支持链式调用
        """
        self._reset()
        self._rewritten_query = rewritten_query
        self._gateway_intent = gateway_intent
        self._worth_saving = worth_saving
        logger.debug(f"开始构建新 Block: {self._block_id}")
        return self

    def add_message(self, message: StreamMessage) -> LogicalBlockBuilder:
        """
        添加消息到正在构建的 Block

        状态转换:
            - USER -> 设置 user_block
            - THOUGHT -> 开始或更新当前 triplet
            - TOOL_CALL -> 添加工具调用到当前 triplet
            - TOOL -> 完成当前 triplet（添加 observation）
            - ASSISTANT -> 设置 response_block（Block 闭合）

        Args:
            message: 流式消息

        Returns:
            Self，支持链式调用
        """
        if message.message_type == StreamMessageType.USER:
            self._user_block = message
            self._execution_chain.clear()
            self._response_block = None

        elif message.message_type == StreamMessageType.THOUGHT:
            self._add_thought(message.content)

        elif message.message_type == StreamMessageType.TOOL_CALL:
            self._add_tool_call(message.tool_name, message.tool_args)

        elif message.message_type == StreamMessageType.TOOL:
            self._add_observation(message.content)

        elif message.message_type == StreamMessageType.ASSISTANT:
            self._response_block = message

        return self

    def _add_thought(self, thought: str) -> None:
        """添加思考到执行链"""
        if not self._execution_chain or self._execution_chain[-1].is_complete:
            self._execution_chain.append(Triplet(thought=thought))
        else:
            self._execution_chain[-1].thought = thought

    def _add_tool_call(
        self,
        tool_name: Optional[str],
        tool_args: Optional[Dict[str, Any]],
    ) -> None:
        """添加工具调用到执行链"""
        if not self._execution_chain or self._execution_chain[-1].is_complete:
            self._execution_chain.append(
                Triplet(tool_name=tool_name, tool_args=tool_args)
            )
        else:
            self._execution_chain[-1].tool_name = tool_name
            self._execution_chain[-1].tool_args = tool_args

    def _add_observation(self, observation: str) -> None:
        """添加工具输出，闭合当前 triplet"""
        if self._execution_chain:
            self._execution_chain[-1].observation = observation

    def _calculate_tokens(self) -> int:
        """计算 Block 的总 Token 数"""
        tokens = 0
        if self._user_block:
            tokens += self._user_block.token_count
        for triplet in self._execution_chain:
            tokens += triplet.total_tokens
        if self._response_block:
            tokens += self._response_block.token_count
        return tokens

    @property
    def is_complete(self) -> bool:
        """检查正在构建的 Block 是否已闭合"""
        return self._user_block is not None and self._response_block is not None

    @property
    def is_started(self) -> bool:
        """检查是否已开始构建（有 user_block）"""
        return self._user_block is not None

    @property
    def is_empty(self) -> bool:
        """检查构建器是否处于空闲状态"""
        return self._user_block is None

    def should_create_new_block(self, message: StreamMessage) -> bool:
        """
        判断是否需要为此消息创建新的 Block

        逻辑:
            - 只有 USER 消息才可能开启新 Block
            - 当前为空状态时，USER 消息开启新 Block
            - 当前 Block 已闭合时，USER 消息开启新 Block
            - 否则继续使用当前 Block

        Args:
            message: 传入的消息

        Returns:
            是否需要创建新 Block
        """
        if message.message_type != StreamMessageType.USER:
            return False

        # 空状态或已闭合时，开启新 Block
        return self.is_empty or self.is_complete

    def build(self) -> LogicalBlock:
        """
        构建并返回 LogicalBlock

        Returns:
            构建完成的 LogicalBlock

        Note:
            不会重置构建器。调用 start() 开始新的 Block。
        """
        block = LogicalBlock(
            block_id=self._block_id,
            user_block=self._user_block,
            execution_chain=self._execution_chain.copy(),
            response_block=self._response_block,
            created_at=self._created_at,
            total_tokens=self._calculate_tokens(),
            rewritten_query=self._rewritten_query,
            gateway_intent=self._gateway_intent,
            worth_saving=self._worth_saving,
        )
        return block


__all__ = ["LogicalBlockBuilder"]
