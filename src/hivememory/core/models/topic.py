"""依赖中立的话题只读领域模型。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from hivememory.core.models.interaction import TurnRecord, turn_reply_text, turn_user_message
from hivememory.core.models.workspace import WorkspaceIdentity
from hivememory.core.models.workspace_asset import TopicAssetBinding


class TopicLastTurn(BaseModel):
    """话题快照中的最后一轮展示值。"""

    user: str = ""
    assistant: str = ""

    model_config = ConfigDict(frozen=True)


class TopicSnapshot(BaseModel):
    """供 Gateway 路由和前端话题池使用的只读快照。"""

    topic_id: str
    workspace_identity: WorkspaceIdentity
    topic_title: str
    topic_summary: str = ""
    fold_summary: str = ""
    last_turn: TopicLastTurn | None = None
    total_tokens: int = 0
    block_count: int = 0
    model_used: str = ""
    last_update: float = Field(description="话题最近一次写入的 Unix 时间戳（秒）")
    has_unsettled_interactions: bool = Field(
        default=False, description="话题是否有还没有沉淀为长期记忆的交互"
    )

    model_config = ConfigDict(frozen=True, use_enum_values=True)


class LogicalBlock(BaseModel):
    """感知层写入短期话题的不可变逻辑块。

    ``total_tokens`` 是写入时对本轮渲染材料的估算；用户消息与回复正文从
    本轮事件集读取，不另存。
    """

    block_id: str = Field(default_factory=lambda: uuid4().hex)
    turn: TurnRecord
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0

    model_config = ConfigDict(frozen=True, use_enum_values=True, extra="forbid")

    @property
    def identity(self):
        """只读派生的内容来源；资源授权使用归属与本次操作的发起者。"""
        return self.turn.identity

    @property
    def turn_events(self):
        return self.turn.turn_events

    @property
    def user_message(self) -> str:
        return turn_user_message(self.turn.turn_events)

    @property
    def reply_text(self) -> str:
        return turn_reply_text(self.turn.turn_events)


class TopicFold(BaseModel):
    """话题的一次折叠记录。

    ``folded_block_ids`` 是这一次被折叠的块，按接收顺序排列，总是折叠当时工作
    视图的前缀；``summary`` 是折叠后的摘要，已经并入此前各次折叠的内容。
    """

    folded_block_ids: tuple[str, ...] = Field(min_length=1)
    summary: str
    folded_at: float = Field(description="折叠时间的 Unix 时间戳（秒）")

    model_config = ConfigDict(frozen=True, extra="forbid")


def topic_working_blocks(
    interactions: Sequence[LogicalBlock], folds: Sequence[TopicFold]
) -> tuple[LogicalBlock, ...]:
    """工作视图：最后一次折叠之后的交互记录。

    每次折叠的都是当时工作视图的前缀，全部折叠记录因此恰好覆盖交互记录开头
    同样数量的块。
    """
    folded_count = sum(len(fold.folded_block_ids) for fold in folds)
    return tuple(interactions[folded_count:])


def topic_fold_summary(folds: Sequence[TopicFold]) -> str:
    """折叠摘要：最后一次折叠的摘要，即被折叠的那部分交互的摘要；没有折叠时为空串。"""
    return folds[-1].summary if folds else ""


class TopicData(BaseModel):
    """短期话题的读取值（不可变），由话题的日志依次套用每条记录组装而成。

    原始数据只有两份：``interactions`` 按接收顺序保存全部交互，只追加，话题结束
    前不移除；``folds`` 每次折叠追加一条。工作视图（``blocks``、``fold_summary``
    与 ``total_tokens``）由二者派生、不单独存储：分开存储的副本需要在每次写入与
    折叠时维持“工作视图是交互记录的后缀”，派生不会失去一致。

    ``settled_interaction_count`` 是已经完成的沉淀覆盖到的交互数：话题空闲时只沉淀、不结束，
    之后还会接收交互，下一次沉淀据此判断有没有新内容。

    ``topic_generated_memory_ids`` 是本话题生成的正式记忆，随话题的日志保存，话题退役后保留到
    日志删除；同一话题之后的写入与沉淀据此不改写、不复述它们。

    话题经短期存储的写入方法写入：每条记录以完整构造产生新的读取值，构造时的校验不会被绕过。执行
    占用不建模为记录字段，跨 await 的占用权由 ``TopicWorkingSet`` 的 lease 表管理。不携带
    话题级 Agent 身份：来源 provenance 由每个 block 的 ``turn.identity`` 记录，资源授权使用
    归属与本次操作的发起者。
    """

    topic_id: str
    workspace_identity: WorkspaceIdentity

    topic_title: str
    topic_summary: str = ""

    interactions: tuple[LogicalBlock, ...] = Field(default_factory=tuple)

    folds: tuple[TopicFold, ...] = Field(default_factory=tuple)

    bindings: tuple[TopicAssetBinding, ...] = Field(default_factory=tuple)

    last_update: float
    model_used: str = ""

    settled_interaction_count: int = Field(default=0, ge=0)
    topic_generated_memory_ids: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "本话题生成的正式记忆 ID：主动写入（WRITE/UPDATE）已结算到的，以及话题此前几次"
            "沉淀新建的；合并吸收了其中一条时改为保留的那条"
        ),
    )

    model_config = ConfigDict(frozen=True, use_enum_values=False, extra="forbid")

    @field_serializer("topic_generated_memory_ids")
    def _serialize_topic_generated_memory_ids(self, ids: frozenset[str]) -> list[str]:
        # 排序输出：同一话题每次序列化的内容相同。
        return sorted(ids)

    @model_validator(mode="after")
    def _folds_cover_interaction_prefix(self) -> TopicData:
        """折叠记录依次拼接，必须等于交互记录开头同样数量的块标识。

        选择规则保证每次折叠的都是工作视图的前缀；校验挡住错误的折叠记录，例如读回的日志里
        被改动过的折叠。
        """
        folded_ids = [block_id for fold in self.folds for block_id in fold.folded_block_ids]
        prefix_ids = [block.block_id for block in self.interactions[: len(folded_ids)]]
        if folded_ids != prefix_ids:
            raise ValueError("折叠记录必须依次覆盖交互记录的前缀")
        return self

    @model_validator(mode="after")
    def _settled_within_interactions(self) -> TopicData:
        """交互记录只追加，沉淀覆盖的交互数不会超过话题接收过的交互数。"""
        if self.settled_interaction_count > len(self.interactions):
            raise ValueError("沉淀覆盖的交互数不能超过交互记录的长度")
        return self

    @property
    def blocks(self) -> tuple[LogicalBlock, ...]:
        """工作视图：最后一次折叠之后的交互记录。"""
        return topic_working_blocks(self.interactions, self.folds)

    @property
    def fold_summary(self) -> str:
        """最后一次折叠的摘要，没有折叠时为空串。"""
        return topic_fold_summary(self.folds)

    @property
    def total_tokens(self) -> int:
        """工作视图的 token 之和，折叠阈值据此判断。"""
        return sum(block.total_tokens for block in self.blocks)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def has_content(self) -> bool:
        """是否接收过交互，即是否有可参与路由与生命周期判断的内容。

        折叠不移除交互记录，有折叠摘要的话题也一定有交互记录。
        """
        return bool(self.interactions)

    @property
    def is_empty(self) -> bool:
        """Topic 是否真正为空：没有接收过任何交互。"""
        return not self.has_content

    @property
    def has_unsettled_interactions(self) -> bool:
        """最近一次沉淀之后是否接收过新的交互；没有接收过交互的话题为假。"""
        return len(self.interactions) > self.settled_interaction_count

    def recent_blocks(self, limit: int) -> tuple[LogicalBlock, ...]:
        if limit <= 0:
            return ()
        return self.blocks[-limit:]

    def is_idle(self, timeout_seconds: int) -> bool:
        return (datetime.now().timestamp() - self.last_update) > timeout_seconds

    def to_topic_snapshot(self) -> TopicSnapshot:
        last_turn = None
        if self.blocks:
            last_block = self.blocks[-1]
            last_turn = TopicLastTurn(
                user=last_block.user_message,
                assistant=last_block.reply_text,
            )
        return TopicSnapshot(
            topic_id=self.topic_id,
            workspace_identity=self.workspace_identity,
            topic_title=self.topic_title,
            topic_summary=self.topic_summary,
            fold_summary=self.fold_summary,
            last_turn=last_turn,
            total_tokens=self.total_tokens,
            block_count=self.block_count,
            model_used=self.model_used,
            last_update=self.last_update,
            has_unsettled_interactions=self.has_unsettled_interactions,
        )


__all__ = [
    "LogicalBlock",
    "TopicData",
    "TopicFold",
    "TopicLastTurn",
    "TopicSnapshot",
    "topic_fold_summary",
    "topic_working_blocks",
]
