"""
HiveMemory 感知层数据模型

定义统一语义流架构中的核心数据结构：
- TraceItem: MTP 操作语义轨迹项 (v3.0 新增，替代 Triplet 中的执行细节)
- InteractionPayload: Kernel → Perception 的原子传输包 (v3.0 新增)
- Triplet: 执行链三元组 (Thought -> Tool Call -> Observation) [向后兼容]
- LogicalBlock: 逻辑原子块（最小语义单元）
- SemanticBuffer: 语义缓冲区

参考: PROJECT.md 4.1 节, PerceptionLayerRefactoring.md

作者: HiveMemory Team
版本: 3.0.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.models import (
    Identity,
    StreamMessage,
    StreamMessageType,
)
from hivememory.utils.token_estimator import estimate_tokens

if TYPE_CHECKING:
    from hivememory.engines.generation.models import WriteFocus, UpdateFocus


# ============ 枚举定义 ============

class FlushReason(str, Enum):
    """缓冲区刷新原因枚举"""
    SEMANTIC_DRIFT = "semantic_drift"  # 语义漂移（话题切换）
    TOKEN_OVERFLOW = "token_overflow"  # Token 溢出
    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    MANUAL = "manual"  # 手动触发
    SHORT_TEXT_ADSORB = "short_text_adsorb"  # 短文本强吸附
    MESSAGE_COUNT = "message_count"  # 消息数量达到阈值（兼容旧版本）
    MTP_WRITE = "mtp_write"  # MTP WRITE 指令触发的强制刷新
    MTP_UPDATE = "mtp_update"  # MTP UPDATE 指令触发的强制刷新
    LRU_EVICTION = "lru_eviction"  # LRU 驱逐（活跃话题池满时换出最久未访问话题）


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
        write_focus: WRITE 指令控制信号（仅 MTP_WRITE flush 时携带）
        update_focus: UPDATE 指令控制信号（仅 MTP_UPDATE flush 时携带）
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

    #: WRITE 指令控制信号 (v3.0)
    write_focus: Optional[Any] = Field(
        default=None,
        description="WRITE 指令的核心素材 (仅 MTP_WRITE flush 时携带)"
    )
    #: UPDATE 指令控制信号 (v3.0)
    update_focus: Optional[Any] = Field(
        default=None,
        description="UPDATE 指令的修改意图 (仅 MTP_UPDATE flush 时携带)"
    )

    @property
    def has_blocks(self) -> bool:
        """检查是否有 blocks 需要 flush"""
        return len(self.blocks_to_flush) > 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============ 语义轨迹项 (v3.0 新增) ============

class TraceItem(BaseModel):
    """
    MTP 操作的语义轨迹项 (替代 Triplet 中的执行细节)

    清洗策略 (对齐 PerceptionLayerRefactoring.md §2.2):
        READ   -> 折叠:   仅记录查阅动作和目标
        SEARCH -> 保留:   记录 Agent 的探索意图
        RUN    -> 摘要:   记录副作用操作及状态
        WRITE/UPDATE -> 不生成 TraceItem (作为控制信号处理)
        XML 响应 -> 丢弃

    Attributes:
        action: 操作类型 (READ / SEARCH / RUN)
        target: READ 目标别名
        query: SEARCH 查询文本
        tool: RUN 工具名称
        status: RUN 执行状态 (success / error)
    """
    action: str = Field(..., description="操作类型: READ / SEARCH / RUN")
    target: Optional[str] = Field(default=None, description="READ 目标别名")
    query: Optional[str] = Field(default=None, description="SEARCH 查询文本")
    tool: Optional[str] = Field(default=None, description="RUN 工具名称")
    status: Optional[str] = Field(default=None, description="RUN 执行状态")

    model_config = ConfigDict(use_enum_values=True)


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
        description="Gateway 重写后的查询（指代消解与上下文补全）"
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

    # ========== v3.0 新增字段 (Kernel/MTP 模式) ==========
    # 参考: PerceptionLayerRefactoring.md §2.1

    #: 原始用户问题 (Kernel 模式下替代 user_block)
    user_query: str = Field(
        default="",
        description="原始用户问题"
    )

    #: 语义轨迹 (替代 execution_chain)
    semantic_traces: List[TraceItem] = Field(
        default_factory=list,
        description="经过清洗和降维的 MTP 操作摘要"
    )

    #: 包含 MTP 指令和 XML 的完整原始文本 (用于 Debug/Context)
    raw_response: str = Field(
        default="",
        description="包含 MTP 噪音的完整原始 assistant 文本"
    )

    #: 去除 MTP 噪音后的纯净回复 (用户可见版本)
    clean_response: str = Field(
        default="",
        description="去除 MTP 噪音后的纯净回复"
    )

    #: 优先级控制信号
    priority: str = Field(
        default="NORMAL",
        description="NORMAL | URGENT"
    )

    #: WRITE 指令的核心素材 (Kernel 模式)
    write_focus: Optional[Any] = Field(
        default=None,
        description="携带 WRITE 指令的核心素材 (WriteFocus)"
    )

    #: UPDATE 指令的修改意图 (Kernel 模式)
    update_focus: Optional[Any] = Field(
        default=None,
        description="携带 UPDATE 指令的修改意图 (UpdateFocus)"
    )

    @property
    def is_complete(self) -> bool:
        """
        Block 是否闭合

        闭合条件 (双模式):
            - Legacy 模式: user_block 和 response_block 都不为空
            - Kernel 模式: user_query 和 clean_response 都不为空
        """
        # Legacy mode
        if self.user_block is not None and self.response_block is not None:
            return True
        # Kernel mode (v3.0)
        if self.user_query and self.clean_response:
            return True
        return False

    @property
    def anchor_text(self) -> str:
        """
        获取语义锚点文本

        锚点对齐策略 (v3.0 更新):
            1. 优先使用 rewritten_query（Gateway 指代消解后的查询）
            2. 回退到 user_query（Kernel 模式原始查询）
            3. 回退到 user_block.content（Legacy 模式原始查询）
        """
        if self.rewritten_query:
            return self.rewritten_query
        if self.user_query:
            return self.user_query
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

        支持双模式:
            - Kernel 模式 (v3.0): 从 user_query + clean_response 构建
            - Legacy 模式: 从 user_block + execution_chain + response_block 构建

        Args:
            identity: 身份标识对象

        Returns:
            List[StreamMessage]: 转换后的消息列表
        """
        messages = []

        # Kernel 模式 (v3.0): 使用 user_query + clean_response
        if self.user_query and self.clean_response:
            messages.append(StreamMessage(
                message_type=StreamMessageType.USER,
                content=self.user_query,
                timestamp=self.created_at,
                identity=identity,
            ))
            messages.append(StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content=self.clean_response,
                timestamp=self.created_at,
                identity=identity,
            ))
            return messages

        # Legacy 模式: 使用 user_block + execution_chain + response_block
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
                    content=f"Calling {triplet.tool_name}",
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
    话题段 (TopicSegment) — 短期记忆的独立工作区

    MMU 架构中的核心数据容器，代表一个独立的讨论线程。
    每个 TopicSegment 拥有绝对纯净的上下文隔离。

    映射关系 (ShortTermMemory.md §2.2):
        TopicSegment = SemanticBuffer
        Pages = blocks (List[LogicalBlock])

    特性：
        - 存储 LogicalBlock 列表（页表 / Pages）
        - 维护话题标题与状态摘要（伪无限上下文基底）
        - 支持 LRU 驱逐判定（last_accessed_at）
        - 支持水位线监控（total_tokens）

    Attributes:
        topic_id: 话题唯一标识（主键），由 PerceptionLayer 创建时生成
        identity: 身份标识（归属元数据，用于权限控制）
        title: 话题标题，由 TheEye 或 Kernel 异步生成，用于菜单展示
        state_summary: 页折叠后的状态摘要，伪无限上下文基底
        blocks: 已闭合的 LogicalBlock 列表（页表）
        topic_kernel_vector: 话题核心向量（保留，用于向量相似度回退）
        state: 缓冲区状态
        last_update: 最后写入时间
        last_accessed_at: 最后访问时间（用于 LRU 驱逐）
        total_tokens: 总 Token 数（水位线监控）
    """
    topic_id: str = Field(default_factory=lambda: str(uuid4()), description="话题唯一标识")
    identity: Identity = Field(default_factory=Identity, description="归属元数据")

    @property
    def buffer_id(self) -> str:
        """向后兼容：返回 topic_id"""
        return self.topic_id

    @property
    def buffer_key(self) -> str:
        """向后兼容：返回 identity.buffer_key"""
        return self.identity.buffer_key

    # --- 话题元数据 (TopicSegment, Phase 4.5 新增) ---
    title: str = Field(default="新建话题", description="话题标题，由 TheEye 或 Kernel 异步生成")
    state_summary: str = Field(default="", description="页折叠后的状态摘要，伪无限上下文基底")

    # --- 页表 (Pages) ---
    blocks: List[LogicalBlock] = Field(default_factory=list)

    # 话题核心（保留，用于向量相似度回退）
    topic_kernel_vector: Optional[List[float]] = None

    # --- 生命周期元数据 ---
    state: BufferState = BufferState.IDLE
    last_update: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_accessed_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0

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


# ============ 交互载荷 (v3.0 新增) ============

class InteractionPayload(BaseModel):
    """
    Kernel -> Perception 的原子传输包

    在 Kernel 完成一轮递归生成循环后，将所有相关数据封装为一个原子包提交给感知层。
    这消除了 WRITE/UPDATE 旁路触发导致的上下文丢失问题。

    参考: PerceptionLayerRefactoring.md §3.1

    Attributes:
        user_message: 原始用户消息
        assistant_message: ���含 MTP 指令的完整 assistant 文本
        mtp_traces: 由 Koakuma 在执行过程中记录的 Trace 列表
        write_focus: WRITE 指令的核心素材 (挂载在 Payload 上，而非独立传输)
        update_focus: UPDATE 指令的修改意图
        identity: 身份标识（归属元数据）
        rewritten_query: Gateway 重写后的查询
        worth_saving: Gateway 价值判断
    """
    user_message: str = Field(..., description="原始用户消息")
    assistant_message: str = Field(..., description="包含 MTP 指令的完整 assistant 文本")
    mtp_traces: List[TraceItem] = Field(
        default_factory=list,
        description="由 Koakuma 在执行过程中记录的 Trace 列表"
    )

    # 控制信号 (挂载在 Payload 上，而非独立传输)
    write_focus: Optional[Any] = Field(
        default=None,
        description="WRITE 指令的核心素材 (WriteFocus)"
    )
    update_focus: Optional[Any] = Field(
        default=None,
        description="UPDATE 指令的修改意图 (UpdateFocus)"
    )

    # 上下文元数据
    identity: Identity = Field(default_factory=Identity, description="归属元数据")
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Gateway 重写后的查询"
    )
    worth_saving: Optional[bool] = Field(
        default=None,
        description="Gateway 价值判断"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = [
    "FlushReason",
    "BufferState",
    "FlushEvent",
    "TraceItem",
    "Triplet",
    "LogicalBlock",
    "InteractionPayload",
    "SemanticBuffer",
]
