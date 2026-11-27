"""依赖中立的话题只读领域模型。"""

from __future__ import annotations

import re
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


class TopicProgressEntry(BaseModel):
    """进展记录的一条：话题自上一条以来有了什么进展，并指回它覆盖的原文。

    一条覆盖交互记录中相邻的若干轮：从上一条覆盖到的位置开始，到 ``covered_until`` 为止。
    ``from_turn_id`` 到 ``to_turn_id`` 的 ``to_event_seq`` 是指回原文的区间，与记忆引用是
    同一套坐标（轮次标识 + 事件序号）；首轮从序号 0 的用户消息开始。
    """

    text: str = Field(description="这一条的正文")
    from_turn_id: str = Field(description="覆盖的第一轮的轮次标识")
    to_turn_id: str = Field(description="覆盖的最后一轮的轮次标识")
    to_event_seq: int = Field(ge=0, description="覆盖的最后一轮中最后一个事件的序号")
    covered_until: int = Field(ge=1, description="这一条覆盖到第几个交互，即交互记录的前缀长度")
    written_at: float = Field(description="写下这一条的 Unix 时间戳（秒）")

    model_config = ConfigDict(frozen=True, extra="forbid")


# 快照只带最近这么多轮还没有整理的交互：话题池展开“还有 N 轮没有整理”时给人看的是节选，
# 不是原文；全部原文留在话题的交互记录里。
UNRECORDED_TURN_EXCERPTS = 3
# 每轮的用户消息与回复各截到这么多字符。
TURN_EXCERPT_MAX_CHARS = 200


class TopicTurnExcerpt(BaseModel):
    """一轮交互的节选：轮次号（话题中从 1 数起）、用户消息与回复，各截到固定长度。"""

    turn_number: int = Field(ge=1)
    user: str = ""
    assistant: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")


def topic_excerpt(text: str) -> str:
    """话题池节选的长度：超过 ``TURN_EXCERPT_MAX_CHARS`` 时截断并加省略号。"""
    if len(text) <= TURN_EXCERPT_MAX_CHARS:
        return text
    return text[:TURN_EXCERPT_MAX_CHARS] + "…"


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
    progress_record: tuple[TopicProgressEntry, ...] = Field(
        default_factory=tuple, description="话题的进展记录，按写下的顺序"
    )
    unrecorded_turn_count: int = Field(
        default=0, ge=0, description="进展记录还没有覆盖到的交互轮数"
    )
    unrecorded_turns: tuple[TopicTurnExcerpt, ...] = Field(
        default_factory=tuple,
        description="进展记录还没有覆盖到的交互中最近的几轮的节选，按轮次先后",
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


def topic_folded_until(folds: Sequence[TopicFold]) -> int:
    """折叠到第几轮：全部折叠记录覆盖的交互数。

    每次折叠的都是当时工作视图的前缀，全部折叠记录因此恰好覆盖交互记录开头
    同样数量的块。
    """
    return sum(len(fold.folded_block_ids) for fold in folds)


def topic_working_blocks(
    interactions: Sequence[LogicalBlock], folds: Sequence[TopicFold]
) -> tuple[LogicalBlock, ...]:
    """工作视图：最后一次折叠之后的交互记录。"""
    return tuple(interactions[topic_folded_until(folds) :])


def topic_fold_summary(folds: Sequence[TopicFold]) -> str:
    """折叠摘要：最后一次折叠的摘要，即被折叠的那部分交互的摘要；没有折叠时为空串。"""
    return folds[-1].summary if folds else ""


class TopicData(BaseModel):
    """短期话题的读取值（不可变），由话题的日志依次套用每条记录组装而成。

    原始数据只有两份：``interactions`` 按接收顺序保存全部交互，只追加，话题结束
    前不移除；``folds`` 每次折叠追加一条。工作视图（``blocks``、``fold_summary``
    与 ``total_tokens``）由二者派生、不单独存储：分开存储的副本需要在每次写入与
    折叠时维持“工作视图是交互记录的后缀”，派生不会失去一致。

    话题的日志中有四个“处理到第几轮”的位置，都是交互记录的前缀长度（覆盖到的交互数），
    并排在这里：

    - ``interaction_count``：话题接收了几轮交互；
    - ``folded_until``：折叠到第几轮，由折叠记录推出，工作视图从这里开始；
    - ``recorded_until``：进展记录写到第几轮，由最后一条进展记录推出；
    - ``settled_until``：已经完成的沉淀覆盖到第几轮。

    后三个都不超过 ``interaction_count``，彼此独立推进，谁也不等谁，也都不切分材料：沉淀
    总是冻结全部交互与折叠记录，生成材料始终是完整的工作视图；进展记录只决定下一条从哪一轮
    写起。进程内还有两个位置（在途沉淀与写进展记录失败的位置），在 ``TopicWorkingSet``。

    ``topic_generated_memory_ids`` 是本话题生成的正式记忆，随话题的日志保存，话题退役后保留到
    日志删除；同一话题之后的写入与沉淀据此不改写、不复述它们。

    ``progress_record`` 是话题的进展记录，只往后追加，各条依次覆盖交互记录中相邻的区间；最后
    一条覆盖到的位置之后的交互是还没有整理的原始交互记录。二者合起来回答话题进行到哪一步。

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

    settled_until: int = Field(default=0, ge=0)
    topic_generated_memory_ids: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "本话题生成的正式记忆 ID：主动写入（WRITE/UPDATE）已结算到的，以及话题此前几次"
            "沉淀新建的；合并吸收了其中一条时改为保留的那条"
        ),
    )
    progress_record: tuple[TopicProgressEntry, ...] = Field(default_factory=tuple)

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
        if self.settled_until > self.interaction_count:
            raise ValueError("沉淀覆盖的交互数不能超过交互记录的长度")
        return self

    @model_validator(mode="after")
    def _progress_record_covers_adjacent_turns(self) -> TopicData:
        """进展记录的每一条从上一条覆盖到的位置开始，往后覆盖至少一轮，不超过交互记录。

        指回原文的区间必须与覆盖的位置一致：首轮与末轮的标识、末轮最后一个事件的序号。校验挡住
        从错误位置写下的一条，例如读回的日志里被改动过的条目。
        """
        start = 0
        for entry in self.progress_record:
            if not start < entry.covered_until <= len(self.interactions):
                raise ValueError("进展记录的一条必须从上一条之后开始，且不超过交互记录的长度")
            first = self.interactions[start].turn
            last = self.interactions[entry.covered_until - 1].turn
            if (entry.from_turn_id, entry.to_turn_id, entry.to_event_seq) != (
                first.turn_id,
                last.turn_id,
                last.turn_events[-1].sequence,
            ):
                raise ValueError("进展记录的一条指回的区间与它覆盖的交互不一致")
            start = entry.covered_until
        return self

    @property
    def interaction_count(self) -> int:
        """话题接收了几轮交互。"""
        return len(self.interactions)

    @property
    def folded_until(self) -> int:
        """折叠到第几轮：工作视图从这一轮之后开始。"""
        return topic_folded_until(self.folds)

    @property
    def recorded_until(self) -> int:
        """进展记录写到第几轮：最后一条的覆盖位置，还没有写过一条时为 0。"""
        return self.progress_record[-1].covered_until if self.progress_record else 0

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
        return self.interaction_count > self.settled_until

    def recent_blocks(self, limit: int) -> tuple[LogicalBlock, ...]:
        if limit <= 0:
            return ()
        return self.blocks[-limit:]

    def to_topic_snapshot(self) -> TopicSnapshot:
        last_turn = None
        if self.blocks:
            last_block = self.blocks[-1]
            last_turn = TopicLastTurn(
                user=last_block.user_message,
                assistant=last_block.reply_text,
            )
        # 还没有整理的交互读交互记录，不读工作视图：被折叠的轮次也可能还没有写进进展记录。
        first_excerpt = max(self.recorded_until, self.interaction_count - UNRECORDED_TURN_EXCERPTS)
        unrecorded_turns = tuple(
            TopicTurnExcerpt(
                turn_number=index + 1,
                user=topic_excerpt(block.user_message),
                assistant=topic_excerpt(block.reply_text),
            )
            for index, block in enumerate(self.interactions[first_excerpt:], start=first_excerpt)
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
            progress_record=self.progress_record,
            unrecorded_turn_count=self.interaction_count - self.recorded_until,
            unrecorded_turns=unrecorded_turns,
        )


# 话题句柄是 agent 读话题的引用，与记忆 alias 共用读取入口：``topic:<话题 id>`` 读进展记录的
# 索引，``topic:<话题 id>#<n>`` 读进展记录第 n 条（从 1 数起）的全文与它覆盖的原文。句柄只由
# 系统签发（读话题的结果与压缩之后的注入），记忆 alias 不得以这个前缀开头。
TOPIC_HANDLE_PREFIX = "topic:"

# 这个会话这一轮所在的话题：读取时先为这一轮确定话题，再换成真实句柄读出索引。它由 agent
# 写出、不由系统签发，也不是话题 id 句柄：话题 id 是 uuid4 字符串，不会是 ``current``。
CURRENT_TOPIC_HANDLE = "topic:current"

_TOPIC_HANDLE = re.compile(r"topic:([^#\s]+)(?:#([1-9][0-9]*))?")
_CURRENT_TOPIC_ID = CURRENT_TOPIC_HANDLE.removeprefix(TOPIC_HANDLE_PREFIX)


def format_topic_handle(topic_id: str, entry_number: int | None = None) -> str:
    """话题的句柄；给出条目序号时是进展记录这一条的句柄。"""
    handle = f"{TOPIC_HANDLE_PREFIX}{topic_id}"
    return handle if entry_number is None else f"{handle}#{entry_number}"


def parse_topic_handle(handle: str) -> tuple[str, int | None] | None:
    """解析话题句柄，返回话题 id 与条目序号（读索引时为空）；不合语法时返回 ``None``。

    ``topic:current`` 不是话题 id 句柄，同样返回 ``None``；它没有带条目序号的写法。
    """
    match = _TOPIC_HANDLE.fullmatch(handle)
    if match is None or match.group(1) == _CURRENT_TOPIC_ID:
        return None
    topic_id, entry = match.groups()
    return topic_id, int(entry) if entry is not None else None


__all__ = [
    "CURRENT_TOPIC_HANDLE",
    "TOPIC_HANDLE_PREFIX",
    "LogicalBlock",
    "TopicData",
    "TopicFold",
    "TopicLastTurn",
    "TopicProgressEntry",
    "TopicSnapshot",
    "TopicTurnExcerpt",
    "format_topic_handle",
    "parse_topic_handle",
    "topic_excerpt",
    "topic_fold_summary",
    "topic_folded_until",
    "topic_working_blocks",
]
