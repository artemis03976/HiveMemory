"""
HiveMemory 感知层数据模型

- TriggerReason: 统一触发原因枚举（历史命名 ``FlushReason`` 已收敛）
- FlushEvent: 触发事件载体（统一输入协议）
- LogicalBlock: 逻辑原子块（最小语义单元）
- TopicMaterializeTask: 感知层 → 生成层的话题结算传输包

Note: 记录字段状态机（原 BufferState）已删除，跨 await 的占用权由
      TopicWorkingSet 的 lease 表表达。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import (
    IdentityScope,
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    TraceItem,
    TurnEvent,
    WorkspaceIdentity,
    require_identity_scope,
)

# ============ 枚举定义 ============

class TriggerReason(str, Enum):
    """统一触发原因枚举。

    七种原因对应 settle / compact / evict 三类具名用例，由
    ``PerceptionFamiliar`` 编排；``reason`` 仅作为 ``TopicMaterializeTask``
    的 provenance 标签，不驱动分支。历史命名 ``FlushReason`` 已删除；
    枚举字符串值保持不变，既有任务与事件载荷不受影响。
    """
    TOKEN_OVERFLOW = "token_overflow"  # Token 溢出
    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    LRU_EVICTION = "lru_eviction"  # LRU 驱逐（活跃话题池满时换出最久未访问话题）
    SHUTDOWN = "shutdown"  # 进程关闭时的全局强制结算
    MANUAL_SETTLE = "manual_settle"  # 用户手动结算：结算为记忆资产并结束 Topic
    MANUAL_COMPACT = "manual_compact"  # 用户手动压缩：只压缩工作集，不结算、不驱逐
    MANUAL_DELETE = "manual_delete"  # 用户手动删除：丢弃 Topic，不写记忆


# ============ 触发事件 ============

class FlushEvent(BaseModel):
    """
    话题触发事件载体。

    事件只携带触发目标与原因；``reason`` 仅作为 provenance 标签传递，
    settle / compact / evict 由 PerceptionFamiliar 的具名用例编排。
    """
    identity_scope: IdentityScope
    topic_id: str
    reason: TriggerReason


# ============ 话题结算载荷 (Perception -> Generation) ============

class TopicMaterializeTask(BaseModel):
    """
    Perception -> Generation 的话题结算传输包

    Topic 结算时将冻结的 buffer 内容打包为此结构发送给 Generation 模块
    进行记忆生成。字段转换统一由 :meth:`from_topic_data` 完成。
    """
    topic_id: str = Field(..., description="话题 ID")
    identity_scope: IdentityScope
    topic_title: str = Field(default="", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")

    blocks: tuple[LogicalBlock, ...] = Field(
        default_factory=tuple,
        description="话题内容块不可变快照",
    )
    state_summary: str = Field(default="", description="话题状态摘要")

    # 结束 Topic 生命周期的 settle 在清除 buffer 前冻结的资产关系事实。进入 queue
    # 后不再依赖短期 buffer 实体；codec/retry 必须原样保留 ref，不从最近资产重推导。
    asset_bindings: tuple[TopicAssetBinding, ...] = Field(
        default_factory=tuple,
        description="settle 前冻结的 Topic 真实使用资产关系",
    )

    reason: TriggerReason = Field(default=TriggerReason.IDLE_TIMEOUT, description="话题结算触发原因")

    # task 会跨越 journal 与 queue admission 边界；禁止字段重新赋值，并使用
    # tuple 承载 blocks，避免冻结模型内部仍可被原地 append。
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @classmethod
    def from_topic_data(
        cls,
        topic_data: TopicData,
        *,
        identity_scope: IdentityScope,
        reason: TriggerReason,
    ) -> "TopicMaterializeTask | None":
        """从冻结 TopicData 构造生成交接任务；无可保存 block 时返回 None。

        TopicData 只保存 Workspace 归属，不保存本次执行者身份，因此
        ``identity_scope`` 必须由调用方显式传入。字段映射、``worth_saving``
        过滤和 no-material 判断集中在此，服务与调用方不得重复拼装。
        """
        identity_scope = require_identity_scope(identity_scope)
        # 结算任务只携带值得保存的 block。
        blocks = tuple(
            block for block in topic_data.blocks if block.worth_saving is not False
        )
        if not blocks:
            return None

        return cls(
            topic_id=topic_data.topic_id,
            identity_scope=identity_scope,
            topic_title=topic_data.topic_title,
            topic_summary=topic_data.topic_summary,
            blocks=blocks,
            state_summary=topic_data.state_summary,
            asset_bindings=topic_data.bindings,
            reason=reason,
        )

    @property
    def user_id(self) -> str:
        """兼容生成层旧展示字段；归属以 workspace_identity 为准。"""
        return self.workspace_identity.owner_user_id

    @property
    def workspace_identity(self) -> WorkspaceIdentity:
        """返回生成输入中唯一的 Workspace ownership。"""
        return self.identity_scope.workspace_identity


__all__ = [
    "FlushEvent",
    "TraceItem",
    "TriggerReason",
    "TurnEvent",
    "TopicMaterializeTask",
]
