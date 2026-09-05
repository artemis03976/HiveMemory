"""
HiveMemory Generation 模块数据模型

仅保留生成流水线内部 DTO（``ExtractedMemoryDraft`` / ``MergeResult`` /
``GenerationRequest`` / ``GenerationContext`` / ``GenerationTurn`` /
``GenerationOutcome``）。

跨 alice/engines/compiler 共享的领域模型（``PendingAtom`` /
``PendingAtomSettlement`` / ``PendingAtomResolution`` / ``PendingAtomStatus`` /
``WriteFocus`` / ``UpdateFocus`` / ``RuntimeScope``）已上移到
``hivememory.core.models``（见 docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md §6.2）。
"""
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from hivememory.core.models import (
    ActorIdentity,
    UpdateFocus,
    WriteFocus,
)


class DuplicateDecision(str, Enum):
    """Deduplication decision produced inside the generation pipeline."""

    CREATE = "create"
    UPDATE = "update"
    TOUCH = "touch"
    DISCARD = "discard"

# ============ 提取结果模型 ============

class ExtractedMemoryDraft(BaseModel):
    """
    提取的记忆草稿 - LLM 输出格式

    Attributes:
        title: 简洁明确的标题 (不超过100字)
        summary: 一句话摘要 (不超过200字)
        tags: 3-5个语义标签
        memory_type: 记忆类型 (CODE_SNIPPET/FACT/...)
        content: 清洗后的 Markdown 内容
        confidence_score: 置信度 (0.0-1.0)
        has_value: 是否有长期价值
    """
    title: str = Field(..., description="简洁明确的标题 (不超过100字)")
    summary: str = Field(..., description="一句话摘要 (不超过200字)")
    tags: List[str] = Field(..., description="3-5个语义标签")
    memory_type: str = Field(
        ...,
        description="记忆类型: CODE_SNIPPET/FACT/URL_RESOURCE/REFLECTION/USER_PROFILE/WORK_IN_PROGRESS"
    )
    content: str = Field(..., description="清洗后的Markdown内容")
    confidence_score: float = Field(..., description="置信度 (0.0-1.0)", ge=0.0, le=1.0)
    has_value: bool = Field(..., description="是否有长期价值 (true/false)")
    alias_suffix: str = Field(
        default="",
        description="别名后缀 (action/subject, snake_case, 不含类型前缀). 例如: 'quicksort_impl', 'project_env'"
    )


class MergeResult(BaseModel):
    """
    LLM Merge Prompt 的输出结果

    Mode C (UPDATE) 中 LLM 执行合并后的结构化输出。

    Attributes:
        new_content: 合并后的新内容
        changelog: 变更日志 (一句话总结此次修改)
    """
    new_content: str
    changelog: str


# ============ Phase 3: 记忆生成视图数据模型 ============

class GenerationTurn(BaseModel):
    """
    记忆生成视图中的单轮对话记录

    保留语义摘要，丢弃 MTP 执行细节，供 GenerationEngine 提取记忆使用。

    Attributes:
        user_query: 用户的原始问题
        assistant_final_text: Agent 最终自然语言回复（已去除 MTP 噪音）
        trace_summaries: 本轮动作摘要（如 SEARCH: "..." / READ: alias_x）
        identity: 产出此轮的身份标识
    """
    user_query: str
    assistant_final_text: str = ""
    trace_summaries: List[str] = Field(default_factory=list)
    identity: ActorIdentity = Field(default_factory=ActorIdentity)


class GenerationContext(BaseModel):
    """
    记忆生成视图的完整上下文

    以结构化方式承载一次 buffer flush 的语义内容，
    作为 GenerationEngine 的主输入。

    Attributes:
        state_summary: 话题状态摘要（page folding 后的语义快照）
        turns: 本次 flush 包含的对话轮次列表
    """
    state_summary: str = ""
    turns: List[GenerationTurn] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    """
    Generation Engine 统一输入协议

    Mode A (被动观察): write_focus=None, update_focus=None
    Mode B (主动响应): write_focus=WriteFocus (WRITE 指令)
    Mode C (合并更新): update_focus=UpdateFocus (UPDATE 指令)

    本协议不携带任何身份/ownership 字段；Memory ownership 由调用方通过
    ``MemoryGenerationTaskSpec.identity_scope`` 传入。
    """
    context: GenerationContext = Field(
        default_factory=lambda: GenerationContext(),
        description="结构化生成上下文"
    )
    write_focus: Optional[WriteFocus] = None
    update_focus: Optional[UpdateFocus] = None
    existing_memory: Optional[Any] = None

    @property
    def is_write(self) -> bool:
        """是否为 Mode B (主动响应模式)"""
        return self.write_focus is not None

    @property
    def is_update(self) -> bool:
        """是否为 Mode C (合并更新模式)"""
        return self.update_focus is not None

    @property
    def has_context(self) -> bool:
        """是否携带结构化生成上下文"""
        return bool(self.context.turns)

    model_config = {"arbitrary_types_allowed": True}


class GenerationOutcome(BaseModel):
    """Pure compute result produced by the generation engine."""

    atom: Optional[Any] = None
    duplicate_decision: DuplicateDecision
    memory_before_snapshot: Optional[Any] = None
    changelog: Optional[str] = None
    message: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "DuplicateDecision",
    "ExtractedMemoryDraft",
    "MergeResult",
    "GenerationRequest",
    "GenerationTurn",
    "GenerationContext",
    "GenerationOutcome",
]
