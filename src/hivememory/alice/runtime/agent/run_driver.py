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
from hivememory.alice.runtime.agent.call_record import CallRecord

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.call_coordinator import CallCoordinator
    from hivememory.alice.runtime.agent.runtime import AgentRuntime


SuspendHandler = Callable[[FrameExecutionResult], Awaitable[None]]
StreamSuspendHandler = Callable[
    [FrameExecutionResult, Callable[[dict[str, Any]], Awaitable[None]]],
    Awaitable[None],
]


class RunDriver:
    """Run-local state machine shared by non-streaming and streaming entrypoints."""

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        call_coordinator: CallCoordinator | None = None,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_coordinator = call_coordinator
        self.call_records: dict[tuple[str, str], CallRecord] = {}
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
            if self._call_coordinator is not None:
                record = self._register_call(frame, result)
                record.begin_resolution()
                response = await self._call_coordinator.resolve_call(
                    frame,
                    result,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                )
                record.mark_resolved()
                if cancel_event is not None and cancel_event.is_set():
                    record.cancel()
                    continue
                self._agent_runtime.apply_call_response(frame, result, response)
                record.mark_applied()
                continue
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
                        if self._call_coordinator is not None:
                            record = self._register_call(frame, result)
                            record.begin_resolution()
                            response = await self._call_coordinator.resolve_call(
                                frame,
                                result,
                                generation_options=generation_options,
                                cancel_event=cancel_event,
                                emit=sink.emit,
                            )
                            record.mark_resolved()
                            if cancel_event is not None and cancel_event.is_set():
                                record.cancel()
                                continue
                            self._agent_runtime.apply_call_response(frame, result, response)
                            record.mark_applied()
                            continue
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

    def _register_call(
        self,
        frame: ExecutionFrame,
        suspension: FrameExecutionResult,
    ) -> CallRecord:
        action_id = suspension.suspend_action_id
        if not action_id:
            raise ValueError("Suspended frame is missing a CALL action id.")
        key = (frame.runtime_scope.frame_id, action_id)
        if key in self.call_records:
            raise RuntimeError(f"CALL record already exists: {key!r}")
        record = CallRecord(caller_frame_id=key[0], action_id=key[1])
        self.call_records[key] = record
        return record


__all__ = ["RunDriver"]
