"""
HiveMemory Generation 模块数据模型
"""
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from hivememory.core.models import Identity, StreamMessage


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
    """
    content: str
    reason: Optional[str] = None
    title: Optional[str] = None
    identity: Identity = Field(default_factory=Identity)


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
        target_uuid: 目标记忆 UUID (由 Koakuma 解析 alias 得到)
        target_alias: 目标记忆 alias
        existing_memory: 由 LibrarianCore 从 storage 加载后注入
        identity: 当前身份标识
    """
    instruction: str
    content: Optional[str] = None
    target_uuid: str
    target_alias: str
    existing_memory: Optional[Any] = None
    identity: Identity = Field(default_factory=Identity)

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


class GenerationRequest(BaseModel):
    """
    Generation Engine 统一输入协议

    封装感知层 flush 产生的上下文消息和可选的指令聚焦内容。
    - Mode A (被动观察): write_focus=None, update_focus=None
    - Mode B (主动响应): write_focus=WriteFocus (WRITE 指令)
    - Mode C (合并更新): update_focus=UpdateFocus (UPDATE 指令)

    Attributes:
        context_messages: 感知层 buffer flush 产生的上下文消息
        write_focus: WRITE 指令的聚焦内容 (None 表示非 Mode B)
        update_focus: UPDATE 指令的聚焦内容 (None 表示非 Mode C)
    """
    context_messages: List[StreamMessage] = Field(default_factory=list)
    write_focus: Optional[WriteFocus] = None
    update_focus: Optional[UpdateFocus] = None

    @property
    def is_focused(self) -> bool:
        """是否为 Mode B (主动响应模式)"""
        return self.write_focus is not None

    @property
    def is_update(self) -> bool:
        """是否为 Mode C (合并更新模式)"""
        return self.update_focus is not None

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "ExtractedMemoryDraft",
    "DuplicateDecision",
    "WriteFocus",
    "UpdateFocus",
    "MergeResult",
    "GenerationRequest",
]
