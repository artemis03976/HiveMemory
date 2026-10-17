"""单条 MTP 指令的执行上下文与回填结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hivememory.agent_runtime.frame import FrameExecutionPolicy, RuntimeScope
from hivememory.core.mtp.models import MTPCallRequest
from hivememory.i18n.resolver import resolve_language

if TYPE_CHECKING:
    from hivememory.workspace.contracts import OperationSubmitter


@dataclass(frozen=True)
class MTPExecutionContext:
    """单条 MTP 指令的观测坐标、显式动词策略与已绑定线程的操作提交函数。"""

    runtime_scope: RuntimeScope
    submit_operation: OperationSubmitter
    execution_policy: FrameExecutionPolicy
    agent_profile: Any = None
    language: str | None = None  # 显式语言覆盖；None 时由 runtime 从 agent_profile 派生


class MTPExecutionResult(BaseModel):
    """MTP 指令执行结果"""

    command: Any | None = Field(default=None)
    response_status: str = Field(default="error")
    response_content: str = Field(default="")
    formatted_response: str = Field(default="")
    success: bool = Field(default=False)
    execution_time_ms: float = Field(default=0.0)
    pending_alias: str | None = Field(default=None)
    call_request: MTPCallRequest | None = Field(default=None)


def resolve_context_language(context: MTPExecutionContext) -> str:
    """优先使用指令语言，其次使用 Profile 语言，最后回落到系统语言。"""
    return resolve_language(
        explicit=context.language,
        profile_language=getattr(context.agent_profile, "language", None),
    ).value
