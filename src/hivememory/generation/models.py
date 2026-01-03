"""
HiveMemory Generation 模块数据模型
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ============ 基础模型: 对话消息 ============

class ConversationMessage(BaseModel):
    """
    对话消息 - 用于缓冲区暂存原始对话
    """
    role: str = Field(..., description="角色: user/assistant/system")
    content: str = Field(..., description="消息内容")
    
    user_id: str = Field(default="unknown", description="用户ID")
    agent_id: str = Field(default="unknown", description="Agent ID")
    session_id: str = Field(default="unknown", description="会话ID")

    timestamp: datetime = Field(default_factory=datetime.now)

    def to_langchain_message(self) -> Dict[str, str]:
        """转换为 LangChain 消息格式"""
        return {
            "role": self.role,
            "content": self.content
        }

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
