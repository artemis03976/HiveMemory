"""
被动接入事件模型 (Passive Ingest Event Models)

定义被动模式的统一事件输入模型 PassiveIngressEvent，
用于替代旧的散装被动 ingest 入参。

作者: HiveMemory Team
版本: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel

from hivememory.core.models import Identity
from hivememory.patchouli.protocol.models import (
    AnalyzeAndRetrieveResult,
    EyeGazeResult,
    InteractionPayload,
    KernelHotResult,
)


@dataclass(frozen=True)
class PassiveSessionKey:
    """
    被动 ingest 专用的会话分桶键。

    该键只用于 observer turn buffer 的事件归并，不进入感知层主链，
    以避免将外部 session 窗口概念重新污染到 Identity / topic 语义。
    """

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
    """
    被动模式统一事件输入模型

    将 user / assistant / tool_call / tool_result 四种事件
    统一为一个结构化输入，替代旧的散装被动 ingest 入参。

    使用示例:
        # 普通用户消息
        PassiveIngressEvent(role="user", content="你好")

        # 助手回复
        PassiveIngressEvent(role="assistant", content="你好！有什么可以帮你的？")

        # 工具调用
        PassiveIngressEvent(
            role="tool_call",
            content="get_weather(city='北京')",
            action_id="act_1",
            tool_name="weather_api",
            tool_kind="function_call",
            tool_args={"city": "北京"},
        )

        # 工具结果
        PassiveIngressEvent(
            role="tool_result",
            content="北京 25°C 晴",
            action_id="act_1",
            status="success",
        )
    """

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
    """
    被动事件路由结果。

    ingressor 负责将离散事件分派到具体缓冲逻辑，并返回统一结果；
    system 层再决定是否提交 flush 结果、是否继续执行 hot 路由以及如何组织外部 API 返回值。
    """

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
