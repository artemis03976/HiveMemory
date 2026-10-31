"""
HiveMemory 感知层数据模型

- TriggerReason: 统一触发原因枚举（历史命名 ``FlushReason`` 已收敛）
- LogicalBlock: 逻辑原子块（最小语义单元）
- TopicMaterializeTask: 感知层 → 生成层的话题结算传输包

Note: 记录字段状态机（原 BufferState）已删除，跨 await 的占用权由
      TopicWorkingSet 的 lease 表表达。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import (
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    TopicFold,
    WorkspaceIdentity,
    topic_state_summary,
    topic_working_blocks,
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


# ============ 话题结算载荷 (Perception -> Generation) ============


class TopicMaterializeTask(BaseModel):
    """
    Perception -> Generation 的话题结算传输包

    Topic 结算时冻结话题的全部交互记录与折叠记录，发送给 Generation 模块：
    来源 InteractionArtifact 覆盖全部轮次（包括已折叠的轮次）并带上折叠记录；
    生成材料仍是工作视图（``state_summary`` 与未折叠的 blocks），与 ``TopicData``
    同一派生。字段转换统一由 :meth:`from_topic_data` 完成。
    """

    topic_id: str = Field(..., description="话题 ID")
    belong_to: WorkspaceIdentity
    topic_title: str = Field(default="", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")

    interactions: tuple[LogicalBlock, ...] = Field(
        default_factory=tuple,
        description="话题接收到的全部交互，按接收顺序",
    )
    folds: tuple[TopicFold, ...] = Field(
        default_factory=tuple,
        description="话题的全部折叠记录，按折叠顺序",
    )

    # 结束 Topic 生命周期的 settle 在清除 buffer 前冻结的资产关系事实。进入 queue
    # 后不再依赖短期 buffer 实体；codec/retry 必须原样保留 ref，不从最近资产重推导。
    asset_bindings: tuple[TopicAssetBinding, ...] = Field(
        default_factory=tuple,
        description="settle 前冻结的 Topic 真实使用资产关系",
    )

    reason: TriggerReason = Field(
        default=TriggerReason.IDLE_TIMEOUT, description="话题结算触发原因"
    )

    # task 会跨越 queue admission 边界；禁止字段重新赋值，并使用 tuple 承载
    # interactions 与 folds，避免冻结模型内部仍可被原地 append。工作视图是派生
    # 属性，以 blocks= 或 state_summary= 构造时显式失败。
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    @property
    def blocks(self) -> tuple[LogicalBlock, ...]:
        """生成材料的工作视图：最后一次折叠之后的交互记录。"""
        return topic_working_blocks(self.interactions, self.folds)

    @property
    def state_summary(self) -> str:
        """最后一次折叠的摘要，没有折叠时为空串。"""
        return topic_state_summary(self.folds)

    @classmethod
    def from_topic_data(
        cls,
        topic_data: TopicData,
        *,
        belong_to: WorkspaceIdentity,
        reason: TriggerReason,
    ) -> TopicMaterializeTask | None:
        """从冻结 TopicData 构造生成交接任务；话题没有接收过交互时返回 None。

        ``belong_to`` 必须由调用方显式传入。字段映射和 no-material 判断集中在此，
        服务与调用方不得重复拼装。
        """
        # 全部交互都进入结算，不在入口按用户消息筛选：只看用户消息会把"请提交"
        # "继续"这类消息背后的整轮实现丢掉。一轮有没有长期价值，由生成阶段在完整
        # 材料上判断。
        if not topic_data.interactions:
            return None

        return cls(
            topic_id=topic_data.topic_id,
            belong_to=belong_to,
            topic_title=topic_data.topic_title,
            topic_summary=topic_data.topic_summary,
            interactions=topic_data.interactions,
            folds=topic_data.folds,
            asset_bindings=topic_data.bindings,
            reason=reason,
        )


__all__ = [
    "TriggerReason",
    "TopicMaterializeTask",
]
