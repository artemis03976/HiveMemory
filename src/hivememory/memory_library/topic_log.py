"""话题日志的记录与套用函数。

每个话题是一条只追加的日志，话题上发生的事各记一条：建立、接收一轮交互、一次折叠、一次沉淀
完成、本话题生成的记忆与退役。``TopicData`` 是由这些记录依次套用得到的读取值，不是存储的
单位。运行时的每次写入与启动时的重放用同一组套用函数，重放得到的读取值因此与写入时内存中的
相同。

套用函数是纯的：时间取记录的 ``at``，不取当前时间；以完整构造产生新值，``TopicData`` 的构造
校验（折叠覆盖交互记录的前缀、沉淀进度不超过交互数）照常生效。记录的内容沿用现有模型的 JSON
形式，不另设序列化。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_serializer,
    model_validator,
)

from hivememory.core.models import (
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    TopicFold,
    WorkspaceAssetRef,
    WorkspaceIdentity,
)
from hivememory.engines.perception.models import TriggerReason


class TopicCreatedRecord(BaseModel):
    """话题的第一条记录：格式版本、Workspace 归属、标题与展示摘要。

    话题 ID 是日志的文件名，不重复写在记录里。日志的格式改变时递增 ``format_version``；
    读回时不认识的版本使整个日志读不回。
    """

    kind: Literal["created"] = "created"
    at: float = Field(description="建立话题的 Unix 时间戳（秒）")
    format_version: Literal[1] = 1
    workspace_identity: WorkspaceIdentity
    topic_title: str
    topic_summary: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicInteractionRecord(BaseModel):
    """话题接收的一轮交互，以及这一轮实际使用的附件引用与模型名。"""

    kind: Literal["interaction"] = "interaction"
    at: float = Field(description="接收这一轮的 Unix 时间戳（秒）")
    block: LogicalBlock
    asset_refs: tuple[WorkspaceAssetRef, ...] = ()
    interaction_id: str | None = None
    model_used: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _asset_refs_need_interaction_id(self) -> TopicInteractionRecord:
        """附件绑定记下首次使用它的交互，没有交互标识就无从记起。"""
        if self.asset_refs and not self.interaction_id:
            raise ValueError("建立 asset binding 必须携带 interaction_id")
        return self


class TopicFoldRecord(BaseModel):
    """一次折叠：被折叠的块（当时工作视图的前缀）与折叠后的摘要。"""

    kind: Literal["fold"] = "fold"
    at: float = Field(description="折叠的 Unix 时间戳（秒）")
    folded_block_ids: tuple[str, ...] = Field(min_length=1)
    summary: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicSettledRecord(BaseModel):
    """一次沉淀完成：这次沉淀覆盖到第几个交互。

    在生成完成时记下，而不是在生成队列接纳时：接纳之后进程退出，生成随进程丢失，进度停在
    上一次完成的位置，重启后这段交互照样会沉淀。
    """

    kind: Literal["settled"] = "settled"
    at: float = Field(description="沉淀完成的 Unix 时间戳（秒）")
    interaction_count: int = Field(ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicGeneratedMemoriesRecord(BaseModel):
    """一次生成执行之后本话题新写下的记忆，以及合并的改指（被吸收的 → 保留的）。"""

    kind: Literal["generated_memories"] = "generated_memories"
    at: float = Field(description="记下生成的记忆的 Unix 时间戳（秒）")
    generated: frozenset[str] = frozenset()
    absorbed_into: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_serializer("generated")
    def _serialize_generated(self, ids: frozenset[str]) -> list[str]:
        # 排序输出：同样的写入每次写出同样的一行。
        return sorted(ids)


class TopicRetiredRecord(BaseModel):
    """话题退役，离开话题池。

    日志保留到话题最后一次沉淀完成后才删除，``reason`` 是退役的原因：重启时重新发起这次沉淀，
    材料带着原来的原因。
    """

    kind: Literal["retired"] = "retired"
    at: float = Field(description="退役的 Unix 时间戳（秒）")
    reason: TriggerReason

    model_config = ConfigDict(frozen=True, extra="forbid")


type TopicLogRecord = Annotated[
    TopicCreatedRecord
    | TopicInteractionRecord
    | TopicFoldRecord
    | TopicSettledRecord
    | TopicGeneratedMemoriesRecord
    | TopicRetiredRecord,
    Field(discriminator="kind"),
]

_RECORD_ADAPTER: TypeAdapter[TopicLogRecord] = TypeAdapter(TopicLogRecord)


def encode_topic_record(record: TopicLogRecord) -> bytes:
    """把一条记录编码为日志中的一行（不含换行）。

    文本中有不能编码为 UTF-8 的字符（例如半个代理对）时抛出 ``ValueError``，什么都不写。
    """
    return record.model_dump_json().encode("utf-8")


def decode_topic_record(line: bytes) -> TopicLogRecord:
    """解码日志中的一行；不是合法的 JSON、字段或取值不合格时抛出 ``ValueError``。"""
    return _RECORD_ADAPTER.validate_json(line)


def topic_from_created(topic_id: str, record: TopicCreatedRecord) -> TopicData:
    """由第一条记录建立话题的读取值：没有交互、折叠与写入。"""
    return TopicData(
        topic_id=topic_id,
        workspace_identity=record.workspace_identity,
        topic_title=record.topic_title,
        topic_summary=record.topic_summary,
        last_update=record.at,
    )


def apply_interaction(topic: TopicData, record: TopicInteractionRecord) -> TopicData:
    """把一轮交互追加到交互记录末尾，也就进入工作视图。

    附件以 ``asset_id`` 幂等绑定：已经绑定过的附件不再记新的绑定。
    """
    bindings = list(topic.bindings)
    bound_ids = {binding.asset_id for binding in bindings}
    for asset_ref in record.asset_refs:
        if asset_ref.asset_id in bound_ids:
            continue
        assert record.interaction_id is not None  # 带附件引用的记录由校验保证带交互标识
        bindings.append(
            TopicAssetBinding(
                asset_ref=asset_ref,
                first_bound_interaction_id=record.interaction_id,
                bound_at=datetime.fromtimestamp(record.at),
            )
        )
        bound_ids.add(asset_ref.asset_id)
    # 以完整构造产生新值：model_copy(update=...) 不运行校验器。
    return TopicData.model_validate(
        {
            **dict(topic),
            "interactions": (*topic.interactions, record.block),
            "bindings": tuple(bindings),
            "last_update": record.at,
            "model_used": record.model_used or topic.model_used,
        }
    )


def apply_fold(topic: TopicData, record: TopicFoldRecord) -> TopicData:
    """追加一条折叠记录，工作视图从被折叠的块之后开始。

    不移除任何交互记录；被折叠的块不是当前工作视图的前缀时构造校验失败。
    """
    fold = TopicFold(
        folded_block_ids=record.folded_block_ids, summary=record.summary, folded_at=record.at
    )
    return TopicData.model_validate(
        {**dict(topic), "folds": (*topic.folds, fold), "last_update": record.at}
    )


def apply_settled(topic: TopicData, record: TopicSettledRecord) -> TopicData:
    """推进沉淀进度，只往前：不改内容与最近写入时间。

    同一话题的沉淀按接纳顺序执行，用户取消排队中的沉淀时完成的顺序可能颠倒，较早的沉淀
    后完成也不会把进度拉回去。沉淀不是一次写入，话题的空闲计时与展示的最后写入时间都不因
    它变化。
    """
    settled = max(topic.settled_interaction_count, record.interaction_count)
    return TopicData.model_validate({**dict(topic), "settled_interaction_count": settled})


def apply_generated_memories(topic: TopicData, record: TopicGeneratedMemoriesRecord) -> TopicData:
    """记下本话题新写下的记忆，并按合并改指记录中的记忆。

    ``generated`` 无条件加入记录；被吸收的记忆在话题此刻的记录中时，记录删去它、改为保留的
    那条。只改这份记录，不改内容、沉淀进度与写入时间。
    """
    recorded = set(topic.topic_generated_memory_ids)
    for absorbed, retained in record.absorbed_into.items():
        if absorbed in recorded:
            recorded.discard(absorbed)
            recorded.add(retained)
    recorded.update(record.generated)
    return TopicData.model_validate(
        {**dict(topic), "topic_generated_memory_ids": frozenset(recorded)}
    )


def replay_topic_log(
    topic_id: str, records: Sequence[TopicLogRecord]
) -> tuple[TopicData, TopicRetiredRecord | None]:
    """依次套用一个话题的全部记录，返回读取值与退役记录（没有退役时为 ``None``）。

    日志必须以 ``created`` 开头、之后不再出现；不满足时，或套用时构造校验失败，抛出
    ``ValueError``。退役之后只会有沉淀完成与本话题生成的记忆：话题离开话题池后只有它们还会记下。
    """
    if not records or not isinstance(records[0], TopicCreatedRecord):
        raise ValueError("话题日志必须以 created 记录开头")
    topic = topic_from_created(topic_id, records[0])
    retired: TopicRetiredRecord | None = None
    for record in records[1:]:
        match record:
            case TopicCreatedRecord():
                raise ValueError("created 记录只能是话题日志的第一条")
            case TopicInteractionRecord():
                topic = apply_interaction(topic, record)
            case TopicFoldRecord():
                topic = apply_fold(topic, record)
            case TopicSettledRecord():
                topic = apply_settled(topic, record)
            case TopicGeneratedMemoriesRecord():
                topic = apply_generated_memories(topic, record)
            case TopicRetiredRecord():
                retired = record
    return topic, retired


__all__ = [
    "TopicCreatedRecord",
    "TopicFoldRecord",
    "TopicInteractionRecord",
    "TopicLogRecord",
    "TopicRetiredRecord",
    "TopicSettledRecord",
    "TopicGeneratedMemoriesRecord",
    "apply_fold",
    "apply_interaction",
    "apply_settled",
    "apply_generated_memories",
    "decode_topic_record",
    "encode_topic_record",
    "replay_topic_log",
    "topic_from_created",
]
