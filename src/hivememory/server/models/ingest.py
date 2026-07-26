"""Passive ingest HTTP API 的 Request/Response 模型"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from hivememory.core.constants import DEFAULT_AGENT_ID


class PassiveIngressRequest(BaseModel):
    # ---------- 外部来源与关联标识 ----------
    source: str = Field(
        ...,
        description="外部来源标识，如 claude_code / codex / telegram_bot",
    )
    external_conversation_id: str = Field(
        ...,
        description="外部会话 ID，与 source 一起构成外部会话命名空间",
    )
    external_event_id: Optional[str] = Field(
        default=None,
        description="外部事件 ID，与 source 一起构成幂等键；缺省时由服务端生成",
    )
    turn_id: Optional[str] = Field(default=None, description="外部 turn 关联 ID")
    occurred_at: Optional[datetime] = Field(
        default=None, description="事件在外部系统发生的时间"
    )
    sequence: Optional[int] = Field(default=None, description="外部事件序号")
    is_final: bool = Field(
        default=False, description="该事件是否完成当前 turn（与 role 无关）"
    )

    # ---------- 事件内容 ----------
    role: Literal["user", "assistant", "tool_call", "tool_result"] = Field(
        ..., description="消息角色"
    )
    content: str = Field(..., description="消息内容")
    user_id: str = Field(..., description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")
    action_id: Optional[str] = Field(
        default=None, description="工具调用 ID (tool_call/tool_result)"
    )
    tool_name: Optional[str] = Field(default=None, description="工具名称 (tool_call)")
    tool_kind: Optional[str] = Field(default=None, description="工具类型 (tool_call)")
    tool_args: Optional[Dict[str, Any]] = Field(
        default=None, description="工具参数 (tool_call)"
    )
    target: Optional[str] = Field(default=None, description="目标 (tool_call)")
    status: Optional[str] = Field(default=None, description="执行状态 (tool_result)")
    render_as: str = Field(default="plain", description="渲染方式 (tool_result)")


class PassiveIngressResponse(BaseModel):
    """Passive ingress 公共响应。

    只包含外部调用方实际需要的接收状态与 memory context；
    不暴露 Gateway execution state、runtime event 或 fallback 细节。
    """

    status: Literal["accepted", "buffered", "duplicate", "ignored"]
    external_event_id: str
    memory: Optional[str] = None


class PassiveFlushRequest(BaseModel):
    source: str = Field(..., description="外部来源标识")
    external_conversation_id: str = Field(..., description="外部会话 ID")
    user_id: str = Field(..., description="用户 ID")
    agent_id: str = Field(default=DEFAULT_AGENT_ID, description="Agent ID")


class PassiveFlushResponse(BaseModel):
    submitted: bool = Field(
        ..., description="是否至少有一个 sealed turn 被成功提交"
    )
