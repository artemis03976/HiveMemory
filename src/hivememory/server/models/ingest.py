"""Passive ingest HTTP API 的 Request/Response 模型"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hivememory.core.constants import DEFAULT_AGENT_ID


class PassiveIngressRequest(BaseModel):
    role: str = Field(..., description="消息角色 (user/assistant/tool_call/tool_result)")
    content: str = Field(..., description="消息内容")
    user_id: str = Field(..., description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    action_id: Optional[str] = Field(default=None, description="工具调用 ID (tool_call/tool_result)")
    tool_name: Optional[str] = Field(default=None, description="工具名称 (tool_call)")
    tool_kind: Optional[str] = Field(default=None, description="工具类型 (tool_call)")
    tool_args: Optional[Dict[str, Any]] = Field(default=None, description="工具参数 (tool_call)")
    target: Optional[str] = Field(default=None, description="目标 (tool_call)")
    status: Optional[str] = Field(default=None, description="执行状态 (tool_result)")
    render_as: str = Field(default="plain", description="渲染方式 (tool_result)")


class PassiveIngressResponse(BaseModel):
    intent: str
    rewritten: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    worth_saving: bool = False
    memory: Optional[str] = None
