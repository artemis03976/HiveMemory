"""Ingest 相关的 Request/Response 模型"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    role: str = Field(..., description="消息角色 (user/assistant)")
    content: str = Field(..., description="消息内容")
    user_id: str = Field(..., description="用户 ID")
    agent_id: str = Field(default="default", description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    context: Optional[List[Dict[str, Any]]] = Field(default=None, description="对话上下文")


class IngestResponse(BaseModel):
    intent: str
    rewritten: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    worth_saving: bool = False
    memory: Optional[str] = None
