"""感知层数据模型。

- ``TriggerReason``：话题结算的触发原因，只作结算材料的 provenance 标签；
- ``TopicMaterializeTask``：感知层交给生成层的话题结算传输包。

话题的交互记录、折叠记录与工作视图的派生定义在 core 话题模型中，感知层与生成层
共用。跨 await 的占用权由 TopicWorkingSet 的 lease 表达，不建模为记录字段。
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
    topic_fold_summary,
    topic_working_blocks,
)

# ============ 枚举定义 ============


class TriggerReason(str, Enum):
    """统一触发原因枚举。

    具名用例由 ``PerceptionFamiliar`` 编排；``reason`` 仅作为 ``TopicMaterializeTask``
    的 provenance 标签，记录这次结算为什么发生，不驱动分支。枚举字符串值是任务与事件
    载荷的一部分。
    """

    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    LRU_EVICTION = "lru_eviction"  # LRU 驱逐（活跃话题池满时换出最久未访问话题）
    MANUAL_SETTLE = "manual_settle"  # 用户手动结算：结算为记忆资产并结束 Topic


# ============ 话题结算载荷 (Perception -> Generation) ============


class TopicMaterializeTask(BaseModel):
    """
    Perception -> Generation 的话题结算传输包

    Topic 结算时冻结话题的全部交互记录与折叠记录，发送给 Generation 模块：
    来源 InteractionArtifact 覆盖全部轮次（包括已折叠的轮次）并带上折叠记录；
    生成材料仍是工作视图（``fold_summary`` 与未折叠的 blocks），与 ``TopicData``
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

    # 冻结时本话题生成的记忆，执行时并入话题当时的记录；升级之前的版本保存的结算的话题不在
    # 话题池里，执行时只有这一份。
    topic_generated_memory_ids: frozenset[str] = Field(
        default_factory=frozenset,
        description="冻结材料时本话题生成的记忆，结算的生成请求带着它",
    )

    reason: TriggerReason = Field(
        default=TriggerReason.IDLE_TIMEOUT, description="话题结算触发原因"
    )
    retires_topic: bool = Field(
        description=(
            "这次沉淀之后话题是否随即退役（离开话题池）；空闲时的沉淀为假，话题之后"
            "还会接收交互、再次沉淀"
        )
    )

    # task 会跨越 queue admission 边界；禁止字段重新赋值，并使用 tuple 承载
    # interactions 与 folds，避免冻结模型内部仍可被原地 append。工作视图是派生
    # 属性，以 blocks= 或 fold_summary= 构造时显式失败。
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    @property
    def blocks(self) -> tuple[LogicalBlock, ...]:
        """生成材料的工作视图：最后一次折叠之后的交互记录。"""
        return topic_working_blocks(self.interactions, self.folds)

    @property
    def fold_summary(self) -> str:
        """最后一次折叠的摘要，没有折叠时为空串。"""
        return topic_fold_summary(self.folds)

    @classmethod
    def from_topic_data(
        cls,
        topic_data: TopicData,
        *,
        belong_to: WorkspaceIdentity,
        reason: TriggerReason,
        retires_topic: bool,
        settled_count: int,
    ) -> TopicMaterializeTask | None:
        """从冻结 TopicData 构造生成交接任务；``settled_count`` 之后没有新交互时返回 None。

        ``settled_count`` 是已经完成与仍在途的沉淀覆盖到的交互数，由调用方给出：在途的沉淀只记
        在进程内，话题的读取值只有已经完成的进度。``belong_to`` 必须由调用方显式传入。字段映射
        和 no-material 判断集中在此，服务与调用方不得重复拼装。
        """
        # 有新交互时冻结全部交互，不只是新增的部分：生成在完整材料上判断，此前沉淀
        # 出的记忆由生成侧记作本话题生成的记忆，复述它们的草稿在那里处理。
        # 全部交互都进入结算，不在入口按用户消息筛选：只看用户消息会把"请提交"
        # "继续"这类消息背后的整轮实现丢掉。一轮有没有长期价值，由生成阶段在完整
        # 材料上判断。
        if len(topic_data.interactions) <= settled_count:
            return None

        return cls(
            topic_id=topic_data.topic_id,
            belong_to=belong_to,
            topic_title=topic_data.topic_title,
            topic_summary=topic_data.topic_summary,
            interactions=topic_data.interactions,
            folds=topic_data.folds,
            asset_bindings=topic_data.bindings,
            topic_generated_memory_ids=topic_data.topic_generated_memory_ids,
            reason=reason,
            retires_topic=retires_topic,
        )


__all__ = [
    "TriggerReason",
    "TopicMaterializeTask",
]
