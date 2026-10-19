"""Alice run 输出契约。

该契约连接父子 frame 调度与当前请求的交互输出，不承担 RuntimeEvent 可观测性发布。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from hivememory.agent_runtime import ExecutionFrame, FrameOutputSink


@dataclass(frozen=True, slots=True)
class CallOutputStarted:
    agent_id: str
    task: str
    iteration: int
    action_id: str
    frame_id: str


@dataclass(frozen=True, slots=True)
class CallOutputFinished:
    status: str
    final_text: str
    iteration: int
    action_id: str
    frame_id: str | None
    agent_id: str
    terminal_status: str | None = None
    error_code: str | None = None


class AgentRunOutput(Protocol):
    """父子 frame 共享的 run-local 输出端口。"""

    def for_frame(
        self,
        frame: ExecutionFrame,
        *,
        action_id: str | None,
        scope: Literal["main", "sub"],
        depth: int,
    ) -> FrameOutputSink: ...

    async def call_started(self, output: CallOutputStarted) -> None: ...

    async def call_finished(self, output: CallOutputFinished) -> None: ...


__all__ = [
    "AgentRunOutput",
    "CallOutputFinished",
    "CallOutputStarted",
]
