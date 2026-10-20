"""Alice run 的交互输出端口：把帧输出、CALL 边界与压缩进度投影为交互事件，推送给 CPU 事件 sink。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from hivememory.agent_runtime import (
    ExecutionFrame,
    FrameOutput,
    FrameOutputSink,
    MTPFinished,
    MTPStarted,
    TokenDelta,
)
from hivememory.alice.orchestration.output import CallOutputFinished, CallOutputStarted
from hivememory.alice.run.output import RunOutput
from hivememory.workspace.contracts import CPUEventSink


class _BoundFrameOutputSink:
    """为单个 frame 绑定固定元数据的 FrameOutputSink 适配。"""

    def __init__(
        self,
        output: SinkRunOutput,
        metadata: Mapping[str, Any],
    ) -> None:
        self._output = output
        self._metadata = dict(metadata)

    async def send(self, output: FrameOutput) -> None:
        event, data = _project_frame_output(output)
        await self._output.send_event(event, data, metadata=self._metadata)


class SinkRunOutput(RunOutput):
    """单个 run 独占的交互输出端口，直接推送给任务进程交来的事件 sink。

    端口只负责投影：补全 ``agent_run_id`` 与 frame 元数据，按 run 递增
    ``stream_sequence``。端口不缓存事件，背压由 sink 施加（进程的输出通道
    写满时等待）；停止时取消在等待的 sink 处抛出，沿 Alice 的调用栈原样
    传播。压缩进度与其他交互事件一样推送，不等待消费方确认。
    """

    # 取消观测沿用流式交付的既有取值，``agent.run.cancelled`` 的载荷保持不变。
    cancellation_message = "Agent stream closed before terminal event."
    close_reason = "stream_closed"

    def __init__(self, events: CPUEventSink) -> None:
        self._events = events
        self._agent_run_id = ""
        self._sequence = 0

    def bind_run(self, agent_run_id: str) -> None:
        self._agent_run_id = agent_run_id

    async def context_compaction(self, *, status: str, data: dict[str, object]) -> None:
        await self.send_event("context_compaction", {"status": status, **data})

    def for_frame(
        self,
        frame: ExecutionFrame,
        *,
        action_id: str | None,
        scope: Literal["main", "sub"],
        depth: int,
    ) -> FrameOutputSink:
        # 展示取 Profile 的源 alias（内置图纸为 omni_doll）；只有未携带身份的 Profile 回退注册标签。
        agent_id = frame.agent_profile.agent_id or frame.runtime_scope.labels.agent_id
        return _BoundFrameOutputSink(
            self,
            metadata={
                "agent_run_id": frame.runtime_scope.run_id,
                "action_id": action_id,
                "scope": scope,
                "depth": depth,
                "agent_id": agent_id,
                "frame_id": frame.runtime_scope.frame_id,
            },
        )

    async def call_started(self, output: CallOutputStarted) -> None:
        await self.send_event(
            "sub_agent_start",
            {
                "agent_id": output.agent_id,
                "task": output.task,
                "iteration": output.iteration,
                "action_id": output.action_id,
                "scope": "sub",
                "depth": 1,
                "frame_id": output.frame_id,
            },
        )

    async def call_finished(self, output: CallOutputFinished) -> None:
        data: dict[str, Any] = {
            "status": output.status,
            "final_text": output.final_text,
            "iteration": output.iteration,
            "action_id": output.action_id,
            "scope": "sub",
            "depth": 1,
            "frame_id": output.frame_id,
            "agent_id": output.agent_id,
        }
        if output.terminal_status is not None:
            data["terminal_status"] = output.terminal_status
        if output.error_code is not None:
            data["error_code"] = output.error_code
        await self.send_event("sub_agent_end", data)

    async def send_event(
        self,
        event: str,
        data: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """补全 run 关联与序号后推送一项交互事件。"""
        payload = {
            "agent_run_id": self._agent_run_id,
            **dict(metadata or {}),
            **dict(data),
            "stream_sequence": self._sequence,
        }
        self._sequence += 1
        await self._events({"event": event, "data": payload})


def _project_frame_output(output: FrameOutput) -> tuple[str, dict[str, Any]]:
    """把强类型 frame 输出投影为兼容的交互事件名与数据。"""
    if isinstance(output, TokenDelta):
        return "token", {"content": output.content}
    if isinstance(output, MTPStarted):
        return "mtp_start", {
            "verb": output.verb,
            "target": output.target,
            "args": output.args,
            "raw_text": output.raw_text,
            "iteration": output.iteration,
            "action_id": output.action_id,
        }
    if isinstance(output, MTPFinished):
        return "mtp_result", {
            "verb": output.verb,
            "target": output.target,
            "args": output.args,
            "raw_text": output.raw_text,
            "status": output.status,
            "iteration": output.iteration,
            "action_id": output.action_id,
        }
    raise TypeError(f"Unsupported frame output: {type(output)!r}")


__all__ = ["SinkRunOutput"]
