"""单条 MTP 指令的执行上下文与回填结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hivememory.agent_runtime.frame import FrameExecutionPolicy, RuntimeScope
from hivememory.core.mtp.models import MTPCallRequest

if TYPE_CHECKING:
    from hivememory.workspace.contracts import OperationSubmitter


@dataclass(frozen=True)
class MTPExecutionContext:
    """单条 MTP 指令的观测坐标、显式动词策略、已绑定线程的操作提交函数与回填语言。"""

    runtime_scope: RuntimeScope
    submit_operation: OperationSubmitter
    execution_policy: FrameExecutionPolicy
    language: str


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
