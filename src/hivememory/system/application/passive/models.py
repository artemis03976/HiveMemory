"""
被动接入事件模型

定义顶层 passive ingress 使用的统一事件输入模型与路由结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel

from hivememory.core.models import Identity
from hivememory.core.protocol.models import (
    AnalyzeAndRetrieveResult,
    EyeGazeResult,
    InteractionPayload,
    KernelHotResult,
)


@dataclass(frozen=True)
class PassiveSessionKey:
    """被动 ingress 专用的会话分桶键。"""

    user_id: str
    agent_id: str
    session_id: Optional[str] = None

    @classmethod
    def from_identity(cls, identity: Identity) -> "PassiveSessionKey":
        return cls(
            user_id=identity.user_id,
            agent_id=identity.agent_id,
            session_id=identity.session_id,
        )

    @property
    def label(self) -> str:
        session = self.session_id or "<default>"
        return f"{self.user_id}:{self.agent_id}:{session}"


class PassiveIngressEvent(BaseModel):
    """被动模式统一事件输入模型。"""

    role: Literal["user", "assistant", "tool_call", "tool_result"]
    content: str
    action_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_kind: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    target: Optional[str] = None
    status: Optional[str] = None
    render_as: str = "plain"


@dataclass(frozen=True)
class PassiveIngressOutcome:
    """被动事件路由结果。"""

    kind: Literal["user", "buffered", "ignored"]
    analysis_result: Optional[AnalyzeAndRetrieveResult] = None
    gaze_result: Optional[EyeGazeResult] = None
    flushed: Optional[tuple[InteractionPayload, Optional[str]]] = None

    @property
    def hot_result(self) -> Optional[KernelHotResult]:
        if self.analysis_result is None:
            return None
        return self.analysis_result.hot_result


__all__ = [
    "PassiveSessionKey",
    "PassiveIngressEvent",
    "PassiveIngressOutcome",
]
