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

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.models import (
    AgentAction,
    LogicalBlock,
    Identity,
    IdentityScope,
    TopicAssetBinding,
    TraceItem,
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
    LRU_EVICTION = "lru_eviction"  # LRU 驱逐（活跃话题池满时换出最久未访问话题）
    SHUTDOWN = "shutdown"  # 进程关闭时的全局强制结算
    MANUAL_SETTLE = "manual_settle"  # 用户手动结算：结算为记忆资产并结束 Topic
    MANUAL_COMPACT = "manual_compact"  # 用户手动压缩：只压缩工作集，不结算、不驱逐
    MANUAL_DELETE = "manual_delete"  # 用户手动删除：丢弃 Topic，不写记忆


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
    identity_scope: IdentityScope
    topic_title: str = Field(default="", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")

    blocks: List[LogicalBlock] = Field(default_factory=list, description="话题内容块列表")
    state_summary: str = Field(default="", description="话题状态摘要")

    # 结束 Topic 生命周期的 settle 在清除 buffer 前冻结的资产关系事实。进入 queue
    # 后不再依赖 SemanticBuffer；codec/retry 必须原样保留 ref，不从最近资产重推导。
    asset_bindings: tuple[TopicAssetBinding, ...] = Field(
        default_factory=tuple,
        description="settle 前冻结的 Topic 真实使用资产关系",
    )

    reason: FlushReason = Field(default=FlushReason.IDLE_TIMEOUT, description="话题结算触发原因")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def user_id(self) -> str:
        """兼容生成层旧展示字段；归属以 workspace_identity 为准。"""
        return self.workspace_identity.owner_user_id

    @property
    def workspace_identity(self) -> WorkspaceIdentity:
        """返回生成输入中唯一的 Workspace ownership。"""
        return self.identity_scope.workspace_identity


@dataclass(frozen=True)
class AutomaticSettleResult:
    """automatic settle 的内部结果，区分实际驱逐与目标已缺失。

    busy 不属于正常返回值，由 ``TopicBusyError`` 显式表达；``settlement`` 为
    ``None`` 仅表示 Topic 已驱逐但没有可提交的生成材料，不能再被解释为 busy。
    """

    evicted: bool
    settlement: TopicMaterializeTask | None = None

    def __post_init__(self) -> None:
        if self.settlement is not None and not self.evicted:
            raise ValueError("未驱逐 Topic 不能携带 settlement payload")


__all__ = [
    "AutomaticSettleResult",
    "FlushReason",
    "FlushEvent",
    "TurnEvent",
    "TraceItem",
    "TopicMaterializeTask",
]
