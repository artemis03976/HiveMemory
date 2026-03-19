"""Chat 相关的 Request/Response 模型"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default="default", description="用户 ID")
    agent_id: str = Field(default="default", description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    enable_memory_retrieval: bool = Field(default=True, description="是否启用记忆检索")


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
