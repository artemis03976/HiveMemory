"""丢弃一切交互输出的空输出替身：被测行为不关心 frame、CALL 与压缩进度输出时使用。"""

from __future__ import annotations

from typing import Literal

from hivememory.agent_runtime import ExecutionFrame, FrameOutput
from hivememory.alice.orchestration.output import CallOutputFinished, CallOutputStarted


class NullFrameOutputSink:
    """接收并丢弃 frame 输出。"""

    async def send(self, output: FrameOutput) -> None:
        del output


class NullAgentRunOutput:
    """不产出 frame 与 CALL 交互事件的调度输出。"""

    def __init__(self) -> None:
        self._frame_output = NullFrameOutputSink()

    def for_frame(
        self,
        frame: ExecutionFrame,
        *,
        action_id: str | None,
        scope: Literal["main", "sub"],
        depth: int,
    ) -> NullFrameOutputSink:
        del frame, action_id, scope, depth
        return self._frame_output

    async def call_started(self, output: CallOutputStarted) -> None:
        del output

    async def call_finished(self, output: CallOutputFinished) -> None:
        del output


class NullRunOutput(NullAgentRunOutput):
    """运行服务的空输出：不交付交互事件，取消观测不带关闭原因。"""

    cancellation_message = "Agent run cancelled before terminal event."
    close_reason = None

    def bind_run(self, agent_run_id: str) -> None:
        del agent_run_id

    async def context_compaction(self, *, status: str, data: dict[str, object]) -> None:
        del status, data


__all__ = ["NullAgentRunOutput", "NullFrameOutputSink", "NullRunOutput"]
