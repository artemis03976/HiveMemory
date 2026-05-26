"""
HiveMemory Generation 模块数据模型
"""
from enum import Enum
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field

from hivememory.core.models import Identity


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


# ============ 枚举定义 ============

class DuplicateDecision(str, Enum):
    """
    查重决策类型

    Attributes:
        CREATE: 创建新记忆
        UPDATE: 更新现有记忆（知识演化）
        TOUCH: 仅更新访问时间（完全重复）
        DISCARD: 丢弃（低质量重复）
    """
    CREATE = "create"
    UPDATE = "update"
    TOUCH = "touch"
    DISCARD = "discard"


# ============ WRITE 指令数据模型 ============

class WriteFocus(BaseModel):
    """
    WRITE 指令的聚焦内容

    当 Agent 通过 MTP WRITE 指令提交记忆草稿时，
    Koakuma 将指令参数打包为 WriteFocus 对象，
    传递给 LibrarianCore → GenerationEngine 处理。

    Attributes:
        content: WRITE 指令的 content 参数 (必需)
        reason: WRITE 指令的 reason 参数 (可选)
        title: WRITE 指令的 title 参数 (可选)
        identity: 当前身份标识
        pending_alias: 运行时 pending alias (Phase 2)
        intent_id: 系统内部写入意图 ID (Phase 2)
    """
    content: str
    reason: Optional[str] = None
    title: Optional[str] = None
    identity: Identity = Field(default_factory=Identity)
    pending_alias: Optional[str] = None
    intent_id: Optional[str] = None


# ============ UPDATE 指令数据模型 ============

class UpdateFocus(BaseModel):
    """
    UPDATE 指令的聚焦内容

    当 Agent 通过 MTP UPDATE 指令提交修改请求时，
    Koakuma 将指令参数打包为 UpdateFocus 对象，
    传递给 LibrarianCore → GenerationEngine 处理。

    Attributes:
        instruction: 修改指令 (必填，自然语言描述)
        content: 新素材 (选填，代码替换或文本追加)
        base_uuid: 本次 revision 基于的正式记忆 UUID
        base_alias: 本次 revision 基于的正式记忆 alias
        identity: 当前身份标识
        pending_alias: 运行时 pending alias (Phase 2)
        intent_id: 系统内部写入意图 ID (Phase 2)
    """
    instruction: str
    content: Optional[str] = None
    base_uuid: str
    base_alias: str
    identity: Identity = Field(default_factory=Identity)
    pending_alias: Optional[str] = None
    intent_id: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


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
    identity: Identity = Field(default_factory=Identity)


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

    封装感知层 flush 产生的结构化生成上下文和可选的指令聚焦内容。
    - Mode A (被动观察): write_focus=None, update_focus=None
    - Mode B (主动响应): write_focus=WriteFocus (WRITE 指令)
    - Mode C (合并更新): update_focus=UpdateFocus (UPDATE 指令)

    Attributes:
        context: 结构化生成上下文
        write_focus: WRITE 指令的聚焦内容 (None 表示非 Mode B)
        update_focus: UPDATE 指令的聚焦内容 (None 表示非 Mode C)
        existing_memory: UPDATE 目标正式记忆，由 LibrarianCore 在构建请求时注入
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

    @property
    def identity(self) -> Identity:
        """
        获取本次 generation 请求的标准身份标识。

        优先级：
            1. write_focus.identity
            2. update_focus.identity
            3. context.turns[0].identity
            4. 默认 Identity()
        """
        if self.write_focus is not None:
            return self.write_focus.identity
        if self.update_focus is not None:
            return self.update_focus.identity
        if self.context.turns:
            return self.context.turns[0].identity
        return Identity()

    model_config = {"arbitrary_types_allowed": True}


# ============ Phase 2: Settlement 数据模型 ============

class PendingAtomSettlement(BaseModel):
    """
    Pending intent 的结算视图。

    由 GenerationEngine 在生成完成后产出，通过 GlobalSystemBus 回填到 AliceRuntime。
    只有 MTP WRITE/UPDATE 触发的主动写入链路（携带 intent_id）才会生成 settlement。
    """
    pending_alias: str
    intent_id: str
    status: Literal["COMMITTED", "MERGED", "UPDATED", "TOUCHED", "DISCARDED", "FAILED"]
    duplicate_decision: Optional[Literal["CREATE", "UPDATE", "TOUCH", "DISCARD"]] = None
    canonical_alias: Optional[str] = None
    canonical_uuid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reason: Optional[str] = None


class MemoryGenerationResult(BaseModel):
    """
    单次记忆生成操作的结构化结果。

    替代原有 List[MemoryAtom] 返回值，携带 intent 追踪和 settlement 信息。
    被动生成（Mode A）中 intent_id/pending_alias/settlement 均为 None。
    """
    intent_id: Optional[str] = None
    pending_alias: Optional[str] = None

    atom: Optional[Any] = None
    canonical_alias: Optional[str] = None
    canonical_uuid: Optional[str] = None

    duplicate_decision: Optional[Literal["CREATE", "UPDATE", "TOUCH", "DISCARD"]] = None
    operation: Literal["created", "merged", "touched", "discarded", "updated", "failed"]

    settlement: Optional[PendingAtomSettlement] = None
    message: Optional[str] = None
    error: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "ExtractedMemoryDraft",
    "DuplicateDecision",
    "WriteFocus",
    "UpdateFocus",
    "MergeResult",
    "GenerationRequest",
    "GenerationTurn",
    "GenerationContext",
    "PendingAtomSettlement",
    "MemoryGenerationResult",
]
