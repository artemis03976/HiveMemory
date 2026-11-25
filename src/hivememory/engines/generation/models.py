"""
HiveMemory Generation 模块数据模型

仅保留生成流水线内部 DTO（``ExtractedMemoryDraft`` / ``MemoryEditSet`` /
``GenerationRequest`` / ``GenerationContext`` / ``GenerationTurn`` / ``GenerationSegment`` /
``GenerationOutcome``）。

跨 alice/engines/compiler 共享的领域模型（``PendingAtom`` /
``PendingAtomSettlement`` / ``PendingAtomResolution`` / ``PendingAtomStatus`` /
``WriteFocus`` / ``UpdateFocus``）位于
``hivememory.core.models``，避免各子系统为共享模型互相导入。
"""

import re
from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    CitedSegment,
    UpdateFocus,
    WriteFocus,
)
from hivememory.core.models.interaction import InteractionOutcome
from hivememory.core.models.memory import MEMORY_SUMMARY_MAX_LENGTH, MEMORY_TITLE_MAX_LENGTH
from hivememory.core.models.provenance import (
    MemoryProvenance,
    normalize_contributing_agent_ids,
)


class DuplicateDecision(str, Enum):
    """查重器对草稿与最相似候选的判断，由生成引擎翻译为提交动作。"""

    DISTINCT = "distinct"  # 没有候选，或相似度不高于下限
    SIMILAR = "similar"  # 相似但内容不同：新建后提名合并
    IDENTICAL = "identical"  # 高相似且内容一致：只触碰已有记忆


class GenerationAction(str, Enum):
    """生成结果的提交动作；修订与合并写回都更新已有记忆。"""

    CREATE = "create"
    UPDATE = "update"
    TOUCH = "touch"


# ============ 提取结果模型 ============


# 段标签：轮次在材料中的位置与段在这一轮中的序号，例如 "2.3"。
_SEGMENT_LABEL = re.compile(r"\d+\.\d+")


def segment_label(turn_position: int, segment_number: int) -> str:
    """材料中第 ``turn_position`` 轮第 ``segment_number`` 段的标签，二者都从 1 起。"""
    return f"{turn_position}.{segment_number}"


def _keep_segment_labels(value: Any) -> list[str]:
    """模型给出的段引用只保留形如 ``2.3`` 的字符串，其余丢弃。

    引用格式不对只丢弃这一处引用，不让整条草稿或整份被动输出因此作废。
    """
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and _SEGMENT_LABEL.fullmatch(item.strip())
    ]


class ExtractedMemoryDraft(BaseModel):
    """
    提取的记忆草稿 - LLM 输出格式

    Attributes:
        title: 简洁明确的标题 (不超过100字)
        summary: 一句话摘要 (不超过200字)
        tags: 3-5个语义标签
        memory_type: 记忆类型 (FACT/RESOURCE/...)
        content: 清洗后的 Markdown 内容
        confidence_score: 置信度 (0.0-1.0)
        has_value: 是否有长期价值
    """

    title: str = Field(..., description="简洁明确的标题 (不超过100字)")
    summary: str = Field(..., description="一句话摘要 (不超过200字)")
    tags: list[str] = Field(..., description="3-5个语义标签")
    memory_type: str = Field(
        ...,
        description="记忆类型: FACT/REFLECTION/USER_PROFILE/RESOURCE",
    )
    content: str = Field(..., description="清洗后的Markdown内容")
    confidence_score: float = Field(..., description="置信度 (0.0-1.0)", ge=0.0, le=1.0)
    has_value: bool = Field(..., description="是否有长期价值 (true/false)")
    alias_suffix: str = Field(
        default="",
        description="别名后缀 (action/subject, snake_case, 不含类型前缀). 例如: 'quicksort_impl', 'project_env'",
    )
    source_segments: list[str] = Field(
        default_factory=list,
        description='支撑这条记忆的段标签，照抄材料中 [Segment x.y] 的 x.y，例如 ["2.3"]；没有可引用的段时为空列表',
    )

    @field_validator("source_segments", mode="before")
    @classmethod
    def _keep_source_segments(cls, value: Any) -> list[str]:
        return _keep_segment_labels(value)

    @model_validator(mode="after")
    def _fit_index_limits(self) -> "ExtractedMemoryDraft":
        """LLM 输出越界时截断到 IndexLayer 的长度上限；标题为空时从正文派生。

        LLM 不保证遵守提示中的长度要求。越界值在草稿入口收敛，避免一次越界
        输出让整个生成任务失败，或写入越界原子。
        """
        self.summary = self.summary.strip()[:MEMORY_SUMMARY_MAX_LENGTH].rstrip()
        title = self.title.strip() or _title_from_content(self.content)
        self.title = title[:MEMORY_TITLE_MAX_LENGTH].rstrip()
        return self


class ExtractionResult(BaseModel):
    """话题结算的提取输出：一段话题中彼此独立、各自值得保存的记忆草稿。"""

    memories: list[ExtractedMemoryDraft] = Field(
        ..., description="记忆草稿列表，没有值得保存的内容时为空"
    )


# 派生标题的截取长度，与 fallback 草稿从正文截取标题的长度一致。
_DERIVED_TITLE_LENGTH = 50


def _title_from_content(content: str) -> str:
    """取正文首个非空、非代码围栏的行作为标题，去掉 Markdown 行首标记。"""
    for line in content.splitlines():
        text = line.strip()
        if not text or text.startswith("```"):
            continue
        text = text.lstrip("#>*- ").strip()
        if text:
            return text[:_DERIVED_TITLE_LENGTH]
    return ""


# ============ 编辑集 ============


class ReplaceEdit(BaseModel):
    """把正文、标题或摘要中逐字出现且只出现一次的原文片段换成新文。"""

    op: Literal["replace"]
    field: Literal["content", "title", "summary"] = "content"
    old: str = Field(min_length=1)
    new: str

    # 多出的键说明模型把别的操作写成了这一种，拒绝而不是静默忽略。
    model_config = ConfigDict(extra="forbid")


class AppendEdit(BaseModel):
    """在正文末尾追加一段；只用于正文，标题与摘要只能片段替换。"""

    op: Literal["append"]
    text: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class RewriteEdit(BaseModel):
    """整篇改写正文；只用于正文。正文不进入索引，大段改写不会让检索漂移。"""

    op: Literal["rewrite"]
    text: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


type MemoryEdit = Annotated[ReplaceEdit | AppendEdit | RewriteEdit, Field(discriminator="op")]


class MemoryEditSet(BaseModel):
    """对一条已有记忆的修订：按顺序应用的若干处编辑，加上标签的逐个增删。

    馆长的修订、主动 UPDATE 与查重提名的合并共用这一种形状。类型、访问策略、alias
    与来源主体不在其中，沿用原记忆。编辑与标签增删都为空表示不需要改。
    """

    edits: list[MemoryEdit] = Field(default_factory=list)
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
    changelog: str
    source_segments: list[str] = Field(default_factory=list)

    @field_validator("source_segments", mode="before")
    @classmethod
    def _keep_source_segments(cls, value: Any) -> list[str]:
        return _keep_segment_labels(value)

    @property
    def is_empty(self) -> bool:
        return not (self.edits or self.add_tags or self.remove_tags)


class MergeFocus(BaseModel):
    """合并的保留与吸收对象；可选意图只用于成功后的改指，不拥有终态结算权。"""

    retained_memory_id: str
    absorbed_memory_id: str
    pending_alias: str | None = None
    intent_id: str | None = None

    @model_validator(mode="after")
    def _require_redirect_pair(self) -> "MergeFocus":
        """句柄改指必须携带完整的 alias 与意图标识。"""
        if (self.intent_id is None) != (self.pending_alias is None):
            raise ValueError("intent_id 与 pending_alias 必须同时提供或同时缺省")
        return self


# ============ 记忆生成视图数据模型 ============


class GenerationSegment(BaseModel):
    """材料中一轮的一段：事件序号区间（首尾都包含在内）与它渲染出的材料条目数。

    各段的条目按顺序拼起来就是这一轮的全部材料条目；第一段还包含用户消息。
    """

    first_sequence: int
    last_sequence: int
    entry_count: int


class GenerationTurn(BaseModel):
    """
    记忆生成视图中的单轮对话记录

    按顺序保留本轮的材料：助手正文与动作穿插排列，能力调用只有一行摘要，
    只有外部工具的结果按上限截断，供 GenerationEngine 提取记忆使用。

    Attributes:
        turn_id: 这一轮的标识，模型引用段时换算为它加事件序号区间
        outcome: 轮次结局，用于标注不完整材料
        user_message: 用户消息
        material_entries: 用户消息之后的材料条目，按出现顺序：助手正文、思考、
            能力调用的一行摘要与外部工具步骤（如 [Assistant]: ... / - THOUGHT: ... /
            - SEARCH: "..." / - TOOL: ...），条目可以多行
        segments: 这一轮的段，材料在每段之前印出段标签。升级前保存的生成任务没有
            段，也没有轮次标识（空串），材料不印段标签，模型也就无从引用
        identity: 产出此轮的身份标识
    """

    turn_id: str
    outcome: InteractionOutcome = "completed"
    user_message: str
    material_entries: list[str] = Field(default_factory=list)
    segments: list[GenerationSegment] = Field(default_factory=list)
    identity: ActorIdentity


class GenerationContext(BaseModel):
    """
    记忆生成视图的完整上下文

    以结构化方式承载一次 buffer flush 的语义内容，
    作为 GenerationEngine 的主输入。

    Attributes:
        fold_summary: 折叠摘要（被折叠的那部分交互的摘要，page folding 产生）
        turns: 本次 flush 包含的对话轮次列表
    """

    fold_summary: str = ""
    turns: list[GenerationTurn] = Field(default_factory=list)

    def cited_segments(self, labels: Sequence[str]) -> list[CitedSegment]:
        """把模型引用的段标签换算为轮次标识加事件序号区间，按引用顺序去重。

        材料中不存在的标签不对应任何段，丢弃。
        """
        known = {
            segment_label(position, number): CitedSegment(
                turn_id=turn.turn_id,
                first_sequence=segment.first_sequence,
                last_sequence=segment.last_sequence,
            )
            for position, turn in enumerate(self.turns, 1)
            for number, segment in enumerate(turn.segments, 1)
        }
        cited: list[CitedSegment] = []
        for label in labels:
            segment = known.get(label)
            if segment is not None and segment not in cited:
                cited.append(segment)
        return cited


def provenance_from_actor(
    from_actor: ActorIdentity,
    context: GenerationContext,
) -> MemoryProvenance:
    """主动模式（WRITE/UPDATE）的来源裁定：以执行 actor 为操作来源。

    内容贡献者先记录发起 Agent 本身——无上下文时主动写入的内容仍由
    发起 Agent 产出——再合并上下文轮次的贡献者。上下文相关的构造逻辑
    留在 generation 层；``MemoryProvenance`` 数据定义已上移 core.models。
    """
    return MemoryProvenance(
        source_agent_id=from_actor.agent_id,
        source_team_id=from_actor.team_id,
        contributing_agent_ids=normalize_contributing_agent_ids(
            [from_actor.agent_id, *_turn_agent_ids(context)]
        ),
    )


def system_settlement_provenance(context: GenerationContext) -> MemoryProvenance:
    """被动结算（SETTLE）的来源裁定：没有具体 Agent 作为操作来源主体。

    来源记录为保留 system；实际参与内容的 Agent 仍进入贡献者集合，
    不把 system 当作内容贡献者。
    """
    return MemoryProvenance(
        source_agent_id=SYSTEM_AGENT_ID,
        source_team_id=None,
        contributing_agent_ids=normalize_contributing_agent_ids(_turn_agent_ids(context)),
    )


def _turn_agent_ids(context: GenerationContext) -> list[str]:
    """收集上下文轮次的 Agent 身份（未去重，交给共享归一化处理）。"""
    return [turn.identity.agent_id for turn in context.turns]


class GenerationRequest(BaseModel):
    """
    Generation Engine 统一输入协议

    Mode A (被动观察): write_focus=None, update_focus=None
    Mode B (主动响应): write_focus=WriteFocus (WRITE 指令)
    Mode C (修订): update_focus=UpdateFocus (UPDATE 指令)
    两条正式记忆的合并: merge_focus=MergeFocus，由 Familiar 读取后调用合并计算。

    本协议不携带任何身份/ownership 字段；Memory ownership 由调用方通过
    ``MemoryGenerationTaskSpec.belong_to`` 传入。
    """

    context: GenerationContext = Field(
        default_factory=lambda: GenerationContext(), description="结构化生成上下文"
    )
    write_focus: WriteFocus | None = None
    update_focus: UpdateFocus | None = None
    merge_focus: MergeFocus | None = None
    existing_memory: Any | None = None
    topic_write_memory_ids: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "本话题写入过的正式记忆 ID：主动写入（WRITE/UPDATE）已结算到的，以及话题此前几次"
            "沉淀新建的。结算接纳时带上冻结材料时话题的记录，控制面在执行时再并入话题当前的"
            "记录。Mode A 草稿查重命中其中一条时丢弃；Mode B 草稿相似命中其中一条时照常新建、"
            "不提名合并"
        ),
    )

    @field_serializer("topic_write_memory_ids")
    def _serialize_topic_write_memory_ids(self, ids: frozenset[str]) -> list[str]:
        # 排序输出：生成任务的载荷要求规范编码，关闭时保存、下次启动读回的载荷
        # 在另一个进程中重新编码时必须得到同样的顺序。
        return sorted(ids)

    @property
    def is_write(self) -> bool:
        """是否为 Mode B (主动响应模式)"""
        return self.write_focus is not None

    @property
    def is_update(self) -> bool:
        """是否为 Mode C (修订模式)"""
        return self.update_focus is not None

    @property
    def has_context(self) -> bool:
        """是否携带结构化生成上下文"""
        return bool(self.context.turns)

    model_config = {"arbitrary_types_allowed": True}


class GenerationOutcome(BaseModel):
    """生成引擎产出的原子与提交动作；查重判为相似时为 CREATE 并携带合并目标。"""

    atom: Any | None = None
    action: GenerationAction
    merge_target_id: str | None = Field(
        default=None, description="新建记忆待合并的保留对象 ID；持久化成功后才提交合并任务"
    )
    memory_before_snapshot: Any | None = None
    changelog: str | None = None
    message: str | None = None
    cited_segments: list[CitedSegment] = Field(
        default_factory=list,
        description="这一版内容引用的交互段，随版本记录保存；只有新建与修订带",
    )
    edit_set: MemoryEditSet | None = Field(
        default=None,
        description=(
            "产生这一版的编辑集，以 memory_before_snapshot 为底；修订与合并的 UPDATE 都带。"
            "提交前基础的版本变了时，把它重放到当前版本上"
        ),
    )

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "DuplicateDecision",
    "GenerationAction",
    "ExtractedMemoryDraft",
    "ExtractionResult",
    "AppendEdit",
    "MemoryEdit",
    "MemoryEditSet",
    "MemoryProvenance",
    "ReplaceEdit",
    "RewriteEdit",
    "MergeFocus",
    "GenerationRequest",
    "GenerationSegment",
    "GenerationTurn",
    "GenerationContext",
    "GenerationOutcome",
    "provenance_from_actor",
    "segment_label",
    "system_settlement_provenance",
]
