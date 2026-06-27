"""
HiveMemory 感知层数据模型

- FlushReason: 触发原因枚举
- FlushEvent: TriggerManager 统一输入协议
- LogicalBlock: 逻辑原子块（最小语义单元）
- TopicMaterializeTask: 感知层 → 生成层的话题结算传输包

Note: SemanticBuffer / BufferState 已迁移至
      hivememory.patchouli.memory_library.buffer
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
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
    IDLE = "idle"
    PROCESSING = "processing"
    FLUSHING = "flushing"


# ============ Flush 事件 ============

class FlushEvent(BaseModel):
    """
    话题结算触发指令，TriggerManager.resolve_topic 的统一输入协议。

    Attributes:
        topic_id: 目标话题 ID
        reason: 触发结算的原因
    """
    topic_id: str
    reason: FlushReason


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


# ============ 话题结算载荷 (Perception -> Generation) ============

class TopicMaterializeTask(BaseModel):
    """
    Perception -> Generation 的话题结算传输包

    当 TriggerManager 触发话题结算时，将 buffer 中的 blocks 打包为此结构
    发送给 Generation 模块进行记忆生成。
    """
    topic_id: str = Field(..., description="话题 ID")
    topic_title: str = Field(default="", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")

    user_id: Optional[str] = Field(default=None, description="用户 ID")

    blocks: List[LogicalBlock] = Field(default_factory=list, description="话题内容块列表")
    state_summary: str = Field(default="", description="话题状态摘要")

    reason: FlushReason = Field(default=FlushReason.IDLE_TIMEOUT, description="话题结算触发原因")

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = [
    "FlushReason",
    "FlushEvent",
    "TurnEvent",
    "TraceItem",
    "LogicalBlock",
    "TopicSnapshot",
    "TopicMaterializeTask",
]
