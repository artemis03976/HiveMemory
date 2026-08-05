"""Alice Agent run 的 queue-backed 流式输出适配。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Mapping
from contextlib import suppress
from typing import Any, Literal

from hivememory.agent_runtime.models import ExecutionFrame, FrameExecutionResult
from hivememory.agent_runtime.output import (
    FrameOutput,
    FrameOutputSink,
    MTPFinished,
    MTPStarted,
    TokenDelta,
)
from hivememory.alice.orchestration.run_output import (
    AgentRunOutput,
    CallOutputFinished,
    CallOutputStarted,
)
from hivememory.alice.orchestration.run_session import RunSession


class _BoundFrameOutputSink:
    """为单个 frame 绑定固定元数据的 FrameOutputSink 适配。"""

    def __init__(
        self,
        output: QueueAgentRunOutput,
        metadata: Mapping[str, Any],
    ) -> None:
        self._output = output
        self._metadata = dict(metadata)

    @property
    def streams_tokens(self) -> bool:
        return True

    async def send(self, output: FrameOutput) -> None:
        event, data = _project_frame_output(output)
        await self._output.send_event(event, data, metadata=self._metadata)


class QueueAgentRunOutput(AgentRunOutput):
    """单个 run 独占的有界交互输出队列。"""

    def __init__(self, agent_run_id: str, *, maxsize: int = 256) -> None:
        self._agent_run_id = agent_run_id
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=maxsize)
        self._sequence = 0

    @property
    def next_sequence(self) -> int:
        return self._sequence

    def for_frame(
        self,
        frame: ExecutionFrame,
        *,
        action_id: str | None,
        scope: Literal["main", "sub"],
        depth: int,
    ) -> FrameOutputSink:
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
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
        payload = {
            "agent_run_id": self._agent_run_id,
            **dict(metadata or {}),
            **dict(data),
            "stream_sequence": self._sequence,
        }
        self._sequence += 1
        await self._queue.put({"event": event, "data": payload})

    async def receive(self) -> dict[str, Any] | None:
        return await self._queue.get()

    async def finish(self) -> None:
        await self._queue.put(None)


class AgentRunStream:
    """管理一次流式 run 的 runner task、队列消费与断流取消。"""

    def __init__(self, session: RunSession, *, queue_size: int = 256) -> None:
        self._session = session
        self.output = QueueAgentRunOutput(
            session.agent_run_id,
            maxsize=queue_size,
        )

    @property
    def next_sequence(self) -> int:
        return self.output.next_sequence

    def events(
        self,
        runner: Awaitable[FrameExecutionResult],
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def _events() -> AsyncGenerator[dict[str, Any], None]:
            consumer_closed = asyncio.Event()

            async def _run() -> FrameExecutionResult:
                try:
                    return await runner
                except asyncio.CancelledError:
                    raise
                finally:
                    # runner 自身被取消时，仍要唤醒正在消费的客户端；只有消费端已经
                    # 主动关闭时才跳过 sentinel，避免在满队列上等待一个已离开的消费者。
                    if not consumer_closed.is_set():
                        await self.output.finish()

            task = asyncio.create_task(_run())
            try:
                while True:
                    event = await self.output.receive()
                    if event is None:
                        break
                    yield event
                await task
            finally:
                consumer_closed.set()
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                elif not task.cancelled():
                    # 若消费端在 runner 异常后、读取 sentinel 前关闭，主动取出异常，
                    # 避免产生 "Task exception was never retrieved"；正常消费路径仍会
                    # 由上面的 await task 把异常传播给调用方。
                    task.exception()

        return _events()


class AgentRunStreamAdapter:
    """由 AliceSystem 注入 application API 的流式运行时工厂。"""

    def __init__(self, *, queue_size: int = 256) -> None:
        self._queue_size = queue_size

    def create(self, session: RunSession) -> AgentRunStream:
        return AgentRunStream(session, queue_size=self._queue_size)


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


__all__ = [
    "AgentRunStream",
    "AgentRunStreamAdapter",
    "QueueAgentRunOutput",
]
