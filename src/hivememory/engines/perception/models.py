"""
HiveMemory 感知层数据模型

定义统一语义流架构中的核心数据结构：
- Triplet: 执行链三元组 (Thought -> Tool Call -> Observation)
- LogicalBlock: 逻辑原子块（最小语义单元）
- SemanticBuffer: 语义缓冲区

参考: PROJECT.md 4.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.models import (
    Identity,
    StreamMessage,
    StreamMessageType,
    estimate_tokens,
)


# ============ 枚举定义 ============

class FlushReason(str, Enum):
    """缓冲区刷新原因枚举"""
    SEMANTIC_DRIFT = "semantic_drift"  # 语义漂移（话题切换）
    TOKEN_OVERFLOW = "token_overflow"  # Token 溢出
    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    MANUAL = "manual"  # 手动触发
    SHORT_TEXT_ADSORB = "short_text_adsorb"  # 短文本强吸附
    MESSAGE_COUNT = "message_count"  # 消息数量达到阈值（兼容旧版本）


class BufferState(str, Enum):
    """Buffer 状态枚举"""
    IDLE = "idle"  # 空闲，等待用户输入
    PROCESSING = "processing"  # 处理中，等待 Block 闭合
    FLUSHING = "flushing"  # 刷新中，触发记忆处理


# ============ Flush 事件 ============

class FlushEvent(BaseModel):
    """
    统一的 Flush 决策输出

    由 Adsorber 或 Relay 产生，表示需要触发 buffer flush。
    PerceptionLayer 根据此事件执行 flush 操作。

    Attributes:
        flush_reason: flush 原因
        blocks_to_flush: 要刷出的 blocks（不包含触发 flush 的新 block）
        relay_summary: 接力摘要（仅 TOKEN_OVERFLOW 时生成）
        triggered_by_block: 触发此 flush 的新 block（将在 flush 后添加到 buffer）
    """
    flush_reason: FlushReason
    blocks_to_flush: List["LogicalBlock"] = Field(
        default_factory=list,
        description="要刷出的 blocks（不包含触发 flush 的新 block）"
    )
    relay_summary: Optional[str] = Field(
        default=None,
        description="接力摘要（仅 TOKEN_OVERFLOW 时生成）"
    )
    triggered_by_block: Optional["LogicalBlock"] = Field(
        default=None,
        description="触发此 flush 的新 block"
    )

    @property
    def has_blocks(self) -> bool:
        """检查是否有 blocks 需要 flush"""
        return len(self.blocks_to_flush) > 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    def to_stream_messages(self, identity: Identity) -> List[StreamMessage]:
        """
        转换为 StreamMessage 列表

        Args:
            identity: 身份标识对象

        Returns:
            List[StreamMessage]: 转换后的消息列表
        """
        messages = []

        if self.user_block:
            msg = self.user_block.model_copy()
            msg.identity = identity
            messages.append(msg)

        for triplet in self.execution_chain:
            if triplet.thought:
                messages.append(StreamMessage(
                    message_type=StreamMessageType.THOUGHT,
                    content=triplet.thought,
                    timestamp=self.created_at,
                    identity=identity
                ))
            if triplet.tool_name:
                messages.append(StreamMessage(
                    message_type=StreamMessageType.TOOL_CALL,
                    content=f"Calling {triplet.tool_name}",  # Placeholder content
                    tool_name=triplet.tool_name,
                    tool_args=triplet.tool_args,
                    timestamp=self.created_at,
                    identity=identity
                ))
            if triplet.observation:
                messages.append(StreamMessage(
                    message_type=StreamMessageType.TOOL,
                    content=triplet.observation,
                    tool_name=triplet.tool_name,
                    timestamp=self.created_at,
                    identity=identity
                ))

        if self.response_block:
            msg = self.response_block.model_copy()
            msg.identity = identity
            messages.append(msg)

        return messages

    model_config = ConfigDict(use_enum_values=True)


# ============ 语义缓冲区 ============

class SemanticBuffer(BaseModel):
    """
    基于语义的对话流缓冲区 - 纯数据容器

    特性：
        - 存储 LogicalBlock 列表（非原始消息）
        - 维护话题核心向量（Topic Kernel）
        - 支持语义吸附判定
        - 支持接力摘要（处理 Token 溢出）

    Note:
        v2.0 重构：移除 current_block 字段和业务逻辑方法。
        Block 构建逻辑由 LogicalBlockBuilder 管理。
        Buffer 操作由 BufferManager 管理。

    Attributes:
        buffer_id: 缓冲区唯一标识
        user_id: 用户ID
        agent_id: Agent ID
        session_id: 会话ID
        blocks: 已闭合的 LogicalBlock 列表
        topic_kernel_vector: 话题核心向量（用于语义吸附判定）
        state: 缓冲区状态
        last_update: 最后更新时间
        total_tokens: 总 Token 数
        relay_summary: 接力摘要（处理 Token 溢出时生成）
    """
    buffer_id: str = Field(default_factory=lambda: str(uuid4()))
    identity: Identity

    blocks: List[LogicalBlock] = Field(default_factory=list)

    # 话题核心
    topic_kernel_vector: Optional[List[float]] = None

    # 状态
    state: BufferState = BufferState.IDLE
    last_update: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0

    # 接力摘要（处理 Token 溢出时生成）
    relay_summary: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)

    def clear(self) -> None:
        """清空缓冲区"""
        self.blocks.clear()
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
        - 存储 StreamMessage 列表
        - 无业务逻辑
        - 线程不安全（由 Perception Layer 管理）

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

    messages: List[StreamMessage] = Field(default_factory=list)
    last_update: float = Field(default_factory=lambda: datetime.now().timestamp())

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_message(self, message: StreamMessage) -> None:
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


__all__ = [
    "FlushReason",
    "BufferState",
    "FlushEvent",
    "Triplet",
    "LogicalBlock",
    "SemanticBuffer",
    "SimpleBuffer",
]
