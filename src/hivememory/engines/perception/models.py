"""
HiveMemory 感知层数据模型

定义统一语义流架构中的核心数据结构：
- TraceItem: MTP 操作语义轨迹项 (v3.0 新增，替代旧执行链的执行细节)
- LogicalBlock: 逻辑原子块（最小语义单元）
- SemanticBuffer: 语义缓冲区

参考: PROJECT.md 4.1 节, PerceptionLayerRefactoring.md

作者: HiveMemory Team
版本: 3.0.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.models import (
    AgentAction,
    Identity,
    TraceItem,
    TopicSnapshot,
    TurnEvent,
    TurnRecord,
)


# ============ 枚举定义 ============

class FlushReason(str, Enum):
    """缓冲区刷新原因枚举"""
    TOKEN_OVERFLOW = "token_overflow"  # Token 溢出
    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    MANUAL = "manual"  # 手动触发
    LRU_EVICTION = "lru_eviction"  # LRU 驱逐（活跃话题池满时换出最久未访问话题）
    SHUTDOWN = "shutdown"  # 进程关闭时的全局强制归档


class BufferState(str, Enum):
    """Buffer 状态枚举"""
    IDLE = "idle"  # 空闲，等待用户输入
    PROCESSING = "processing"  # 处理中，等待 Block 闭合
    FLUSHING = "flushing"  # 刷新中，触发记忆处理


# ============ Flush 事件 ============

class FlushEvent(BaseModel):
    """
    统一的 Flush 决策输出

    由 PerceptionLayer 或 Relay 产生，表示需要触发 buffer flush。
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

    @property
    def has_blocks(self) -> bool:
        """检查是否有 blocks 需要 flush"""
        return len(self.blocks_to_flush) > 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============ 逻辑原子块 ============

class LogicalBlock(BaseModel):
    """
    逻辑原子块 - 感知层的最小处理单元

    Attributes:
        turn: 单轮内容真相记录
        created_at: 创建时间
        total_tokens: 总 Token 数
        block_id: 块唯一标识
    """
    block_id: str = Field(default_factory=lambda: str(uuid4()))
    turn: TurnRecord = Field(
        default_factory=TurnRecord,
        description="本块承载的单轮内容真相"
    )

    # 辅助信息
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0

    # ========== 多智能体身份溯源 (Phase 1) ==========
    #: Gateway 意图分类结果
    gateway_intent: Optional[str] = Field(
        default=None,
        description="Gateway 意图分类 (RAG/CHAT/SYSTEM)"
    )

    #: Gateway 记忆价值信号
    worth_saving: Optional[bool] = Field(
        default=None,
        description="Gateway 记忆价值判断"
    )

    #: 优先级控制信号
    priority: str = Field(
        default="NORMAL",
        description="NORMAL | URGENT"
    )

    @property
    def is_complete(self) -> bool:
        """Block 是否闭合"""
        return bool(self.turn.user_query) and bool(
            self.turn.assistant_final_text
            or self.turn.turn_events
            or self.turn.actions
        )

    @property
    def anchor_text(self) -> str:
        """
        获取语义锚点文本

        锚点对齐策略:
            1. 优先使用 rewritten_query
            2. 回退到 user_query
        """
        return self.turn.anchor_text

    @property
    def identity(self) -> Identity:
        return self.turn.identity

    @property
    def rewritten_query(self) -> Optional[str]:
        return self.turn.rewritten_query

    @property
    def user_query(self) -> str:
        return self.turn.user_query

    @property
    def assistant_final_text(self) -> str:
        return self.turn.assistant_final_text

    @property
    def turn_events(self) -> List[TurnEvent]:
        return self.turn.turn_events

    @property
    def actions(self) -> List[AgentAction]:
        return self.turn.actions

    @property
    def semantic_traces(self) -> List[TraceItem]:
        return self.turn.semantic_traces

    model_config = ConfigDict(use_enum_values=True, extra="forbid")


# ============ 话题上下文缓冲区 ============

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
        user_id: 用户标识（归属元数据，用于权限控制）
        title: 话题标题，由 TheEye 或 Kernel 异步生成，用于菜单展示
        state_summary: 页折叠后的状态摘要，伪无限上下文基底
        blocks: 已闭合的 LogicalBlock 列表（页表）
        state: 缓冲区状态
        last_update: 最后写入时间
        last_accessed_at: 最后访问时间（用于 LRU 驱逐）
        total_tokens: 总 Token 数（水位线监控）
    """
    topic_id: str = Field(default_factory=lambda: str(uuid4()), description="话题唯一标识")
    user_id: str = Field(default="default", description="用户标识")

    # --- 多智能体调度 ---
    #: 当前话题挂载的活跃人偶别名，Kernel 切换 Agent 时更新此指针
    current_agent_id: str = Field(
        default="default",
        description="当前话题挂载的活跃 Agent 别名 (如 coder_doll)"
    )

    # --- 话题元数据 (TopicSegment ---
    topic_title: str = Field(default="新建话题", description="话题标题，由 TheEye 在话题创建时生成")
    topic_summary: str = Field(default="", description="话题展示摘要，由 TheEye 在话题创建时生成，面向前端展示")

    # --- 状态摘要 ---
    state_summary: str = Field(default="", description="页折叠后的状态摘要，伪无限上下文基底，随上下文压缩反复更新")

    # --- 页表 (Pages) ---
    blocks: List[LogicalBlock] = Field(default_factory=list)

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


# ============ 归档载荷 (Perception -> Librarian) ============

class ArchivePayload(BaseModel):
    """
    Perception -> GenerationEngine 的归档传输包

    当 TriggerManager 触发 Archive 操作时，将 buffer 中的 blocks 打包为此结构
    发送给 GenerationEngine 进行记忆生成。

    每个 LogicalBlock 自行携带 identity，无需在 payload 层面统一标识。

    Attributes:
        topic_id: 话题 ID
        user_id: 用户 ID（可选，用于兼容性）
        blocks: 从 buffer flush 出的 LogicalBlock 列表（每个 block 携带自己的 identity）
        state_summary: 话题状态摘要（如果有折叠）
        focus: write_focus 或 update_focus（仅 MTP_WRITE/UPDATE 时有值）
        reason: flush 触发原因
    """
    topic_id: str = Field(..., description="话题 ID")
    topic_title: str = Field(default="", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")

    user_id: Optional[str] = Field(default=None, description="用户 ID")

    blocks: List[LogicalBlock] = Field(default_factory=list, description="从 buffer flush 出的 blocks")
    state_summary: str = Field(default="", description="话题状态摘要")
    
    reason: FlushReason = Field(default=FlushReason.IDLE_TIMEOUT, description="flush 触发原因")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = [
    "FlushReason",
    "BufferState",
    "FlushEvent",
    "TurnEvent",
    "TraceItem",
    "LogicalBlock",
    "SemanticBuffer",
    "TopicSnapshot",
    "ArchivePayload",
]
