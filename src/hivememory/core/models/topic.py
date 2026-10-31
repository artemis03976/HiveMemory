"""依赖中立的话题只读领域模型。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.models.interaction import TurnRecord, turn_reply_text, turn_user_message
from hivememory.core.models.workspace import WorkspaceIdentity
from hivememory.core.models.workspace_asset import TopicAssetBinding, WorkspaceAssetRef


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
    state_summary: str = ""
    last_turn: TopicLastTurn | None = None
    total_tokens: int = 0
    block_count: int = 0
    model_used: str = ""
    last_update: float = Field(description="话题最近一次写入的 Unix 时间戳（秒）")

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
    视图的前缀；``summary`` 是折叠后的状态摘要，已经并入此前各次折叠的内容。
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


def topic_state_summary(folds: Sequence[TopicFold]) -> str:
    """话题状态摘要：最后一次折叠的摘要，没有折叠时为空串。"""
    return folds[-1].summary if folds else ""


class TopicData(BaseModel):
    """短期话题的业务快照（不可变读取对象）。

    原始数据只有两份：``interactions`` 按接收顺序保存全部交互，只追加，话题结束
    前不移除；``folds`` 每次折叠追加一条。工作视图（``blocks``、``state_summary``
    与 ``total_tokens``）由二者派生、不单独存储：分开存储的副本需要在每次写入与
    折叠时维持“工作视图是交互记录的后缀”，派生不会失去一致。

    快照只经 ``merge_interaction_into_topic`` 与 ``fold_topic`` 写入，二者以完整
    构造产生新快照，构造时的校验不会被绕过。执行占用不建模为记录字段，跨 await
    的占用权由 ``TopicWorkingSet`` 的 lease 表管理。不携带话题级 Agent 身份：
    来源 provenance 由每个 block 的 ``turn.identity`` 记录，资源授权使用归属与
    本次操作的发起者。
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

    model_config = ConfigDict(frozen=True, use_enum_values=False, extra="forbid")

    @model_validator(mode="after")
    def _folds_cover_interaction_prefix(self) -> TopicData:
        """折叠记录依次拼接，必须等于交互记录开头同样数量的块标识。

        选择规则保证每次折叠的都是工作视图的前缀；校验挡住写函数之外的错误构造。
        """
        folded_ids = [block_id for fold in self.folds for block_id in fold.folded_block_ids]
        prefix_ids = [block.block_id for block in self.interactions[: len(folded_ids)]]
        if folded_ids != prefix_ids:
            raise ValueError("折叠记录必须依次覆盖交互记录的前缀")
        return self

    @property
    def blocks(self) -> tuple[LogicalBlock, ...]:
        """工作视图：最后一次折叠之后的交互记录。"""
        return topic_working_blocks(self.interactions, self.folds)

    @property
    def state_summary(self) -> str:
        """最后一次折叠的摘要，没有折叠时为空串。"""
        return topic_state_summary(self.folds)

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
            state_summary=self.state_summary,
            last_turn=last_turn,
            total_tokens=self.total_tokens,
            block_count=self.block_count,
            model_used=self.model_used,
            last_update=self.last_update,
        )


def merge_interaction_into_topic(
    topic: TopicData,
    block: LogicalBlock,
    asset_refs: tuple[WorkspaceAssetRef, ...],
    interaction_id: str | None,
    model_used: str | None,
) -> TopicData:
    """把一次 Interaction 的 block、首次绑定与元数据并成新快照。

    block 追加到交互记录末尾，也就进入工作视图。binding 以 ``asset_id`` 幂等
    去重；原子性由「frozen 快照 → 新快照 → ``Store.put`` 整条替换」承担，
    互斥由调用方持有的 lease 保证。
    """
    bindings = list(topic.bindings)
    now = datetime.now()
    if asset_refs:
        if not interaction_id:
            raise ValueError("建立 asset binding 必须携带 interaction_id")
        existing_ids = {binding.asset_id for binding in bindings}
        for asset_ref in asset_refs:
            if not isinstance(asset_ref, WorkspaceAssetRef):
                raise TypeError("asset_ref 必须是 WorkspaceAssetRef")
            asset_id = asset_ref.asset_id
            if asset_id in existing_ids:
                continue
            bindings.append(
                TopicAssetBinding(
                    asset_ref=asset_ref,
                    first_bound_interaction_id=interaction_id,
                    bound_at=now,
                )
            )
            existing_ids.add(asset_id)
    # 以完整构造产生新快照：model_copy(update=...) 不运行校验器。
    return TopicData.model_validate(
        {
            **dict(topic),
            "interactions": (*topic.interactions, block),
            "bindings": tuple(bindings),
            "last_update": now.timestamp(),
            "model_used": model_used or topic.model_used,
        }
    )


def fold_topic(topic: TopicData, folded: Sequence[LogicalBlock], summary: str) -> TopicData:
    """把工作视图开头的 ``folded`` 折叠为 ``summary``，追加一条折叠记录。

    不移除任何交互记录，只推进工作视图；``folded`` 不是当前工作视图的前缀时
    构造校验失败。互斥由调用方持有的 lease 保证。
    """
    now = datetime.now().timestamp()
    fold = TopicFold(
        folded_block_ids=tuple(block.block_id for block in folded),
        summary=summary,
        folded_at=now,
    )
    return TopicData.model_validate(
        {**dict(topic), "folds": (*topic.folds, fold), "last_update": now}
    )


__all__ = [
    "LogicalBlock",
    "TopicData",
    "TopicFold",
    "TopicLastTurn",
    "TopicSnapshot",
    "fold_topic",
    "merge_interaction_into_topic",
    "topic_state_summary",
    "topic_working_blocks",
]
