"""
HiveMemory 感知层数据模型

定义统一语义流架构中的核心数据结构：
- StreamMessage: 统一流式消息模型
- Triplet: 执行链三元组 (Thought -> Tool Call -> Observation)
- LogicalBlock: 逻辑原子块（最小语义单元）
- SemanticBuffer: 语义缓冲区

参考: PROJECT.md 4.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from hivememory.core.models import FlushReason
from hivememory.generation.models import ConversationMessage


def estimate_tokens(text: str) -> int:
    """
    估算文本的 Token 数量

    规则：
    - 中文 1 token ≈ 2 字符
    - 英文 1 token ≈ 4 字符
    - 这是一个粗略估算，仅供测试使用
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return (chinese_chars // 2) + (other_chars // 4)


# ============ 枚举定义 ============

class StreamMessageType(str, Enum):
    """流式消息类型枚举"""
    USER_QUERY = "user_query"  # 用户查询
    THOUGHT = "thought"  # 思考过程
    TOOL_CALL = "tool_call"  # 工具调用
    TOOL_OUTPUT = "tool_output"  # 工具输出
    ASSISTANT_MESSAGE = "assistant_message"  # 助手消息
    SYSTEM_MESSAGE = "system_message"  # 系统消息


class BufferState(str, Enum):
    """Buffer 状态枚举"""
    IDLE = "idle"  # 空闲，等待用户输入
    PROCESSING = "processing"  # 处理中，等待 Block 闭合
    FLUSHING = "flushing"  # 刷新中，触发记忆处理


# ============ 流式消息模型 ============

class StreamMessage(BaseModel):
    """
    统一流式消息模型

    职责：抹平不同 Agent 框架的消息格式差异

    Attributes:
        message_type: 消息类型
        content: 消息内容
        timestamp: 时间戳
        tool_name: 工具名称（可选，用于 TOOL_CALL）
        tool_args: 工具参数（可选，用于 TOOL_CALL）
        tool_result: 工具结果（可选，用于 TOOL_OUTPUT）
    """
    message_type: StreamMessageType
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    # 工具调用相关字段（可选）
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None

    def to_conversation_message(self, user_id: str, agent_id: str, session_id: str) -> "ConversationMessage":
        """
        转换为现有的 ConversationMessage 模型（向后兼容）

        Args:
            user_id: 用户ID
            agent_id: 智能体ID
            session_id: 会话ID

        Returns:
            ConversationMessage: 转换后的消息对象
        """
        # from hivememory.generation.models import ConversationMessage (removed local import)


        return ConversationMessage(
            role=self._get_role(),
            content=self.content,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            timestamp=datetime.fromtimestamp(self.timestamp),
        )

    def _get_role(self) -> str:
        """映射消息类型到角色"""
        mapping = {
            StreamMessageType.USER_QUERY: "user",
            StreamMessageType.ASSISTANT_MESSAGE: "assistant",
            StreamMessageType.SYSTEM_MESSAGE: "system",
            StreamMessageType.THOUGHT: "assistant",
            StreamMessageType.TOOL_CALL: "assistant",
            StreamMessageType.TOOL_OUTPUT: "system",
        }
        return mapping.get(self.message_type, "assistant")

    @property
    def token_count(self) -> int:
        """估算消息的 Token 数量"""
        return estimate_tokens(self.content)

    class Config:
        use_enum_values = True


# ============ 执行链三元组 ============

class Triplet(BaseModel):
    """
    执行链三元组：Thought -> Tool Call -> Observation

    约束：三个元素中 tool_name 和 observation 必须存在才算完整

    Attributes:
        thought: 思考过程（可选）
        tool_name: 工具名称
        tool_args: 工具参数
        observation: 执行结果
    """
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """
        检查三元组是否完整

        三元组完整定义：
            - 至少有 tool_name（表明发生了工具调用）
            - 有 observation（表明工具调用已完成）
        """
        return self.tool_name is not None and self.observation is not None

    @property
    def is_pending(self) -> bool:
        """
        检查三元组是否处于待完成状态

        即：有 tool_name 但还没有 observation
        """
        return self.tool_name is not None and self.observation is None

    @property
    def total_tokens(self) -> int:
        """估算三元组的 Token 数量"""
        tokens = 0
        if self.thought:
            tokens += estimate_tokens(self.thought)
        if self.tool_name:
            tokens += estimate_tokens(self.tool_name)
        if self.tool_args:
            tokens += estimate_tokens(str(self.tool_args))
        if self.observation:
            tokens += estimate_tokens(self.observation)
        return tokens

    def add_thought(self, thought: str) -> None:
        """添加思考内容"""
        self.thought = thought

    def add_tool_call(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None) -> None:
        """添加工具调用"""
        self.tool_name = tool_name
        self.tool_args = tool_args

    def add_observation(self, observation: str) -> None:
        """添加工具执行结果"""
        self.observation = observation


# ============ 逻辑原子块 ============

class LogicalBlock(BaseModel):
    """
    逻辑原子块 - 语义流感知层的最小处理单元

    结构：
        1. user_block: 用户意图（必须）
        2. execution_chain: 执行链（可选，三元组列表）
        3. response_block: 最终响应（必须）

    状态机逻辑：
        1. State: IDLE -> 收到 User Message -> 创建新 LogicalBlock
        2. State: PROCESSING -> 收到 Thought/Tool Call/Tool Output -> 暂存入 execution_chain
        3. State: PROCESSING -> 收到 Assistant Message -> 填入 response_block
        4. Block 闭合 (Sealed) -> is_complete = True

    Attributes:
        user_block: 用户消息块
        execution_chain: 执行链（三元组列表）
        response_block: 响应消息块
        created_at: 创建时间
        total_tokens: 总 Token 数
        block_id: 块唯一标识
    """
    user_block: Optional[StreamMessage] = None
    execution_chain: List[Triplet] = Field(default_factory=list)
    response_block: Optional[StreamMessage] = None

    # 辅助信息
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0
    block_id: str = Field(default_factory=lambda: str(uuid4()))

    # ========== v2.0 新增字段 (Gateway 集成) ==========

    #: Gateway 重写后的查询（指代消解后的完整查询）
    #: 这是 Gateway 的核心输出之一，用于替代 raw query 做语义锚点
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Gateway 重写后的查询（指代消解）"
    )

    #: 会话轮次 ID，用于去重与幂等
    #: 格式: conversation_id:turn_index 或单独的 turn_id
    turn_id: Optional[str] = Field(
        default=None,
        description="会话轮次 ID（用于去重与幂等）"
    )

    #: Gateway 意图分类结果
    gateway_intent: Optional[str] = Field(
        default=None,
        description="Gateway 意图分类 (RAG/CHAT/TOOL/SYSTEM)"
    )

    #: Gateway 记忆价值信号
    worth_saving: Optional[bool] = Field(
        default=None,
        description="Gateway 记忆价值判断"
    )

    @property
    def is_complete(self) -> bool:
        """
        Block 是否闭合（User 和 Response 都存在）

        闭合条件：
            - user_block 不为空
            - response_block 不为空
        """
        return (
            self.user_block is not None
            and self.response_block is not None
        )

    @property
    def anchor_text(self) -> str:
        """
        获取语义锚点文本

        锚点对齐策略 (v2.0 更新):
            1. 优先使用 rewritten_query（Gateway 指代消解后的查询）
            2. 回退到 user_block.content（原始查询）

        这样确保感知层获得最准确的语义表示。
        """
        if self.rewritten_query:
            return self.rewritten_query
        if self.user_block:
            return self.user_block.content
        return ""

    @property
    def has_pending_triplet(self) -> bool:
        """检查是否有未完成的三元组"""
        return any(t.is_pending for t in self.execution_chain)

    def add_stream_message(self, message: StreamMessage) -> None:
        """
        将流式消息添加到 Block

        状态转换逻辑：
            - User Query -> user_block
            - Thought -> 开始新的三元组或添加到当前三元组
            - Tool Call -> 附加到最新的三元组
            - Tool Output -> 闭合最新三元组
            - Assistant Message -> response_block

        Args:
            message: 流式消息
        """
        if message.message_type == StreamMessageType.USER_QUERY:
            # 新用户查询，重置 Block
            self.user_block = message
            self.execution_chain.clear()
            self.response_block = None

        elif message.message_type == StreamMessageType.THOUGHT:
            # 添加思考到执行链
            self._add_thought_to_chain(message.content)

        elif message.message_type == StreamMessageType.TOOL_CALL:
            # 添加工具调用
            self._add_tool_call_to_chain(message.tool_name, message.tool_args)

        elif message.message_type == StreamMessageType.TOOL_OUTPUT:
            # 添加工具输出，闭合三元组
            self._add_observation_to_chain(message.content)

        elif message.message_type == StreamMessageType.ASSISTANT_MESSAGE:
            # 最终响应
            self.response_block = message

        self._recalculate_tokens()

    def _add_thought_to_chain(self, thought: str) -> None:
        """添加思考到执行链"""
        # 如果最后一个三元组已完成，创建新的
        if not self.execution_chain or self.execution_chain[-1].is_complete:
            self.execution_chain.append(Triplet(thought=thought))
        else:
            # 否则更新当前三元组的思考
            self.execution_chain[-1].thought = thought

    def _add_tool_call_to_chain(self, tool_name: str, tool_args: Optional[Dict[str, Any]]) -> None:
        """添加工具调用"""
        # 如果最后一个三元组已完成，创建新的
        if not self.execution_chain or self.execution_chain[-1].is_complete:
            self.execution_chain.append(Triplet(tool_name=tool_name, tool_args=tool_args))
        else:
            # 否则更新当前三元组的工具调用
            self.execution_chain[-1].tool_name = tool_name
            self.execution_chain[-1].tool_args = tool_args

    def _add_observation_to_chain(self, observation: str) -> None:
        """添加工具输出，闭合三元组"""
        if self.execution_chain:
            self.execution_chain[-1].observation = observation

    def _recalculate_tokens(self) -> None:
        """重新计算 Token 总数"""
        tokens = 0
        if self.user_block:
            tokens += self.user_block.token_count
        for triplet in self.execution_chain:
            tokens += triplet.total_tokens
        if self.response_block:
            tokens += self.response_block.token_count
        self.total_tokens = tokens

    def to_conversation_messages(self, user_id: str, agent_id:str, session_id: str) -> List["ConversationMessage"]:
        """
        转换为 ConversationMessage 列表（向后兼容）

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            List[ConversationMessage]: 转换后的消息列表
        """
        from hivememory.generation.models import ConversationMessage

        messages = []

        if self.user_block:
            messages.append(self.user_block.to_conversation_message(user_id, agent_id, session_id))

        for triplet in self.execution_chain:
            if triplet.thought:
                messages.append(ConversationMessage(
                    role="assistant",
                    content=f"[思考] {triplet.thought}",
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    
                    timestamp=datetime.fromtimestamp(self.created_at),
                ))
            if triplet.tool_name:
                args_str = str(triplet.tool_args) if triplet.tool_args else ""
                messages.append(ConversationMessage(
                    role="assistant",
                    content=f"[调用工具] {triplet.tool_name}({args_str})",
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    timestamp=datetime.fromtimestamp(self.created_at),
                ))
            if triplet.observation:
                messages.append(ConversationMessage(
                    role="system",
                    content=f"[工具结果] {triplet.observation}",
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    timestamp=datetime.fromtimestamp(self.created_at),
                ))

        if self.response_block:
            messages.append(self.response_block.to_conversation_message(user_id, agent_id, session_id))

        return messages

    def get_summary(self) -> str:
        """获取 Block 的简要摘要"""
        parts = []
        if self.user_block:
            user_content = self.user_block.content[:50]
            parts.append(f"User: {user_content}...")
        if self.execution_chain:
            parts.append(f"Tools: {len(self.execution_chain)} calls")
        if self.response_block:
            response_content = self.response_block.content[:50]
            parts.append(f"Response: {response_content}...")
        return " | ".join(parts)

    class Config:
        use_enum_values = True


# ============ 语义缓冲区 ============

class SemanticBuffer(BaseModel):
    """
    基于语义的对话流缓冲区

    特性：
        - 存储 LogicalBlock 列表（非原始消息）
        - 维护话题核心向量（Topic Kernel）
        - 支持语义吸附判定
        - 支持接力摘要（处理 Token 溢出）

    Attributes:
        buffer_id: 缓冲区唯一标识
        user_id: 用户ID
        agent_id: Agent ID
        session_id: 会话ID
        blocks: 已闭合的 LogicalBlock 列表
        current_block: 当前正在构建的 LogicalBlock
        topic_kernel_vector: 话题核心向量（用于语义吸附判定）
        state: 缓冲区状态
        last_update: 最后更新时间
        total_tokens: 总 Token 数
        relay_summary: 接力摘要（处理 Token 溢出时生成）
        max_tokens: 最大 Token 限制
    """
    buffer_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    agent_id: str
    session_id: str

    blocks: List[LogicalBlock] = Field(default_factory=list)
    current_block: Optional[LogicalBlock] = None

    # 话题核心
    topic_kernel_vector: Optional[List[float]] = None

    # 状态
    state: BufferState = BufferState.IDLE
    last_update: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0

    # 接力摘要（处理 Token 溢出时生成）
    relay_summary: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

    def add_block(self, block: LogicalBlock) -> None:
        """
        添加已闭合的 Block 到缓冲区

        Args:
            block: 已闭合的 LogicalBlock
        """
        if not block.is_complete:
            raise ValueError("只能添加已闭合的 Block")

        self.blocks.append(block)
        self.total_tokens += block.total_tokens
        self.last_update = datetime.now().timestamp()

    def set_current_block(self, block: LogicalBlock) -> None:
        """
        设置当前正在构建的 Block

        Args:
            block: 正在构建的 LogicalBlock
        """
        self.current_block = block
        if block.user_block:
            self.state = BufferState.PROCESSING
        self.last_update = datetime.now().timestamp()

    def clear(self) -> None:
        """清空缓冲区"""
        self.blocks.clear()
        self.current_block = None
        self.total_tokens = 0
        self.state = BufferState.IDLE
        self.last_update = datetime.now().timestamp()

    def get_block_count(self) -> int:
        """获取 Block 数量"""
        return len(self.blocks)

    def get_topic_summary(self) -> str:
        """
        获取话题摘要

        Returns:
            str: 话题摘要字符串
        """
        if not self.blocks:
            return "空缓冲区"

        user_queries = [b.anchor_text for b in self.blocks if b.anchor_text]
        if user_queries:
            return f"包含 {len(user_queries)} 个用户查询"
        return f"{len(self.blocks)} 个 Block"

    def is_idle(self, timeout_seconds: int = 900) -> bool:
        """
        检查缓冲区是否空闲

        Args:
            timeout_seconds: 超时时间（秒）

        Returns:
            bool: 是否空闲
        """
        current_time = datetime.now().timestamp()
        return (current_time - self.last_update) > timeout_seconds


# ============ 简单缓冲区 ============

class SimpleBuffer(BaseModel):
    """
    简单缓冲区 - 纯数据容器

    特性：
        - 存储 ConversationMessage 列表
        - 无业务逻辑
        - 线程不安全（由 Perception Layer 管理）
        - 符合 SemanticBuffer 模式

    Attributes:
        buffer_id: 缓冲区唯一标识
        user_id: 用户ID
        agent_id: Agent ID
        session_id: 会话ID
        messages: 消息列表
        last_update: 最后更新时间
    """
    buffer_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    agent_id: str
    session_id: str

    messages: List[ConversationMessage] = Field(default_factory=list)
    last_update: float = Field(default_factory=lambda: datetime.now().timestamp())

    class Config:
        arbitrary_types_allowed = True  # 允许 ConversationMessage

    def add_message(self, message: ConversationMessage) -> None:
        """添加消息到缓冲区"""
        self.messages.append(message)
        self.last_update = datetime.now().timestamp()

    def clear(self) -> None:
        """清空缓冲区"""
        self.messages.clear()
        self.last_update = datetime.now().timestamp()

    @property
    def message_count(self) -> int:
        """获取消息数量"""
        return len(self.messages)


__all__ = [
    # 枚举
    "StreamMessageType",
    "BufferState",
    # FlushReason 从 core.models 导出
    # 模型
    "StreamMessage",
    "Triplet",
    "LogicalBlock",
    "SemanticBuffer",
    "SimpleBuffer",  # 新增
]
