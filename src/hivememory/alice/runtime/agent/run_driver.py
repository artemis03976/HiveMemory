from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.events import QueueFrameEventSink
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.runtime import AgentRuntime


SuspendHandler = Callable[[FrameExecutionResult], Awaitable[None]]
StreamSuspendHandler = Callable[
    [FrameExecutionResult, Callable[[dict[str, Any]], Awaitable[None]]],
    Awaitable[None],
]


class RunDriver:
    """Run-local state machine shared by non-streaming and streaming entrypoints."""

    def __init__(self, agent_runtime: AgentRuntime) -> None:
        self._agent_runtime = agent_runtime
        self.terminal_result: FrameExecutionResult | None = None
        self.next_stream_sequence = 0

    async def run(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        on_suspend: SuspendHandler | None = None,
    ) -> FrameExecutionResult:
        while True:
            result = await self._agent_runtime.run_frame(
                frame,
                generation_options=generation_options,
                cancel_event=cancel_event,
            )
            if result.status != FrameExecutionStatus.SUSPENDED:
                self.terminal_result = result
                return result
            if on_suspend is None:
                result = FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED,
                    error=RuntimeError("Frame suspended without an orchestration callback."),
                )
                self.terminal_result = result
                return result
            await on_suspend(result)

    def run_stream(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        event_metadata: dict[str, Any] | None = None,
        on_suspend: StreamSuspendHandler | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def _stream() -> AsyncGenerator[dict[str, Any], None]:
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
            sink = QueueFrameEventSink(queue, metadata=event_metadata)

            async def _runner() -> None:
                try:
                    while True:
                        result = await self._agent_runtime.run_frame(
                            frame,
                            generation_options=generation_options,
                            event_sink=sink,
                            cancel_event=cancel_event,
                        )
                        if result.status != FrameExecutionStatus.SUSPENDED:
                            self.terminal_result = result
                            break
                        if on_suspend is None:
                            self.terminal_result = FrameExecutionResult(
                                status=FrameExecutionStatus.FAILED,
                                error=RuntimeError(
                                    "Frame suspended without an orchestration callback."
                                ),
                            )
                            break
                        await on_suspend(result, sink.emit)
                finally:
                    self.next_stream_sequence = sink.next_sequence
                    await queue.put(None)

            task = asyncio.create_task(_runner())
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield event
            finally:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        return _stream()


__all__ = ["RunDriver"]
