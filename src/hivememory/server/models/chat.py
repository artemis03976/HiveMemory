"""Chat 相关的 Request/Response 模型"""

from typing import List, Optional

from pydantic import BaseModel, Field

from hivememory.core.constants import DEFAULT_USER_ID, DEFAULT_AGENT_ID


class GenerationOptions(BaseModel):
    model: Optional[str] = Field(default=None, min_length=1, description="模型名称")
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description="采样温度")
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description="Top-p 采样率")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32768, description="最大生成长度")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default=DEFAULT_USER_ID, description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    enable_memory_retrieval: bool = Field(default=True, description="是否启用记忆检索")
    generation_options: Optional[GenerationOptions] = Field(default=None, description="本次请求的生成参数覆盖")


# ========== SSE 事件数据模型 ==========

class ChatTokenEvent(BaseModel):
    """token 事件: LLM 生成的文本增量"""
    content: str


class MTPStartEvent(BaseModel):
    """mtp_start 事件: MTP 指令被拦截"""
    verb: str
    target: str = ""
    args: dict = Field(default_factory=dict)
    raw_text: str = ""
    iteration: int


class MTPResultEvent(BaseModel):
    """mtp_result 事件: MTP 执行完成"""
    verb: str
    target: str = ""
    args: dict = Field(default_factory=dict)
    raw_text: str = ""
    status: str
    iteration: int


class TopicInfoEvent(BaseModel):
    """topic_info 事件: 话题路由结果"""
    topic_id: str
    is_new: bool


class ChatDoneEvent(BaseModel):
    """done 事件: 生成完成"""
    final_text: str
    mtp_iterations: int
    total_iterations: int
    mtp_commands_executed: List[str]


class ChatErrorEvent(BaseModel):
    """error 事件: 错误发生"""
    message: str
    detail: Optional[str] = None
