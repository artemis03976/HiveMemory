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

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.models import (
    AgentAction,
    LogicalBlock,
    Identity,
    TraceItem,
    MemoryCreationContext,
    TopicSnapshot,
    TurnEvent,
    TurnRecord,
    WorkspaceIdentity,
    WorkspaceTopicKey,
)


# ============ 枚举定义 ============

class FlushReason(str, Enum):
    """缓冲区刷新原因枚举"""
    TOKEN_OVERFLOW = "token_overflow"  # Token 溢出
    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    MANUAL = "manual"  # 手动触发
    LRU_EVICTION = "lru_eviction"  # LRU 驱逐（活跃话题池满时换出最久未访问话题）
    SHUTDOWN = "shutdown"  # 进程关闭时的全局强制归档


# ============ Flush 事件 ============

class FlushEvent(BaseModel):
    """
    话题结算触发指令，TriggerManager.resolve_topic 的统一输入协议。

    Attributes:
        topic_key: 已验证的目标话题复合键
        reason: 触发结算的原因
    """
    topic_key: WorkspaceTopicKey
    reason: FlushReason

    @property
    def topic_id(self) -> str:
        """返回展示用 topic ID；Store 寻址必须使用 topic_key。"""
        return self.topic_key.topic_id


# ============ 话题结算载荷 (Perception -> Generation) ============

class TopicMaterializeTask(BaseModel):
    """
    Perception -> Generation 的话题结算传输包

    当 TriggerManager 触发话题结算时，将 buffer 中的 blocks 打包为此结构
    发送给 Generation 模块进行记忆生成。
    """
    topic_id: str = Field(..., description="话题 ID")
    creation_context: MemoryCreationContext
    topic_title: str = Field(default="", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")

    blocks: List[LogicalBlock] = Field(default_factory=list, description="话题内容块列表")
    state_summary: str = Field(default="", description="话题状态摘要")

    reason: FlushReason = Field(default=FlushReason.IDLE_TIMEOUT, description="话题结算触发原因")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def user_id(self) -> str:
        """兼容生成层旧展示字段；归属以 workspace_identity 为准。"""
        return self.workspace_identity.owner_user_id

    @property
    def workspace_identity(self) -> WorkspaceIdentity:
        """返回生成输入中唯一的 Workspace ownership。"""
        return self.creation_context.workspace_identity


__all__ = [
    "FlushReason",
    "FlushEvent",
    "TurnEvent",
    "TraceItem",
    "TopicMaterializeTask",
]
