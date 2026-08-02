from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.events import (
    FrameEventSink,
    NullFrameEventSink,
    QueueFrameEventSink,
)
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.alice.runtime.agent.call_record import CallRecord
from hivememory.alice.runtime.agent.run_session import RunSession

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.call_coordinator import CallCoordinator
    from hivememory.alice.runtime.agent.runtime import AgentRuntime


class RunDriver:
    """Run-local state machine shared by non-streaming and streaming entrypoints."""

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        *,
        session: RunSession,
        call_coordinator: CallCoordinator | None = None,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._session = session
        self._call_coordinator = call_coordinator
        self.terminal_result: FrameExecutionResult | None = None
        self.runtime_products: RuntimeProducts | None = None

    @property
    def call_records(self) -> dict[tuple[str, str], CallRecord]:
        return self._session.call_records

    @property
    def next_stream_sequence(self) -> int:
        return self._session.stream_sequence

    async def run(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
    ) -> FrameExecutionResult:
        self._session.require_frame(frame)
        return await self._run_until_terminal(
            frame,
            generation_options=generation_options,
            event_sink=NullFrameEventSink(),
        )

    def run_stream(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        event_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def _stream() -> AsyncGenerator[dict[str, Any], None]:
            self._session.require_frame(frame)
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
            sink = QueueFrameEventSink(
                queue,
                metadata=event_metadata,
                sequence_start=self._session.stream_sequence,
            )

            async def _runner() -> None:
                cancelled = False
                try:
                    await self._run_until_terminal(
                        frame,
                        generation_options=generation_options,
                        event_sink=sink,
                    )
                except asyncio.CancelledError:
                    cancelled = True
                    self._session.cancel_event.set()
                    if self.terminal_result is None:
                        self._finish(FrameExecutionResult(status=FrameExecutionStatus.CANCELLED))
                    raise
                finally:
                    self._session.stream_sequence = sink.next_sequence
                    if not cancelled:
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

    async def _run_until_terminal(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None,
        event_sink: FrameEventSink,
    ) -> FrameExecutionResult:
        """Drive one root frame until it reaches a terminal outcome."""
        while True:
            result = await self._agent_runtime.run_frame(
                frame,
                generation_options=generation_options,
                event_sink=event_sink,
                cancel_event=self._session.cancel_event,
            )

            match result.status:
                case FrameExecutionStatus.SUSPENDED:
                    terminal_result = await self._resolve_suspension(
                        frame,
                        result,
                        generation_options=generation_options,
                        event_sink=event_sink,
                    )
                    if terminal_result is not None:
                        return self._finish(terminal_result)
                    continue

                case (
                    FrameExecutionStatus.COMPLETED
                    | FrameExecutionStatus.CANCELLED
                    | FrameExecutionStatus.FAILED
                    | FrameExecutionStatus.BUDGET_EXHAUSTED
                ):
                    result = self._normalize_terminal_result(
                        result,
                        self._session.cancel_event,
                    )
                    return self._finish(result)

                case unexpected_status:
                    raise RuntimeError(
                        f"RunDriver received an unsupported frame status: {unexpected_status!r}"
                    )

    async def _resolve_suspension(
        self,
        frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        generation_options: dict[str, Any] | None,
        event_sink: FrameEventSink,
    ) -> FrameExecutionResult | None:
        """Resolve one suspended CALL and return a terminal failure when it cannot run."""
        if self._call_coordinator is None:
            return FrameExecutionResult(
                status=FrameExecutionStatus.FAILED,
                error=RuntimeError("Frame suspended without an orchestration callback."),
            )

        record = self._register_call(frame, suspension)
        record.begin_resolution()
        response = await self._call_coordinator.resolve_call(
            frame,
            suspension,
            session=self._session,
            generation_options=generation_options,
            event_sink=event_sink,
        )
        record.mark_resolved()
        if self._session.cancel_event.is_set():
            record.cancel()
            return None

        self._agent_runtime.apply_call_response(frame, suspension, response)
        record.mark_applied()
        return None

    def _finish(self, result: FrameExecutionResult) -> FrameExecutionResult:
        if self.terminal_result is not None:
            raise RuntimeError("RunDriver attempted to finalize a run more than once.")
        self.terminal_result = result
        finalize_run = getattr(self._agent_runtime, "finalize_run", None)
        if callable(finalize_run):
            self.runtime_products = finalize_run(
                self._session.agent_run_id,
                result,
            )
        else:
            self.runtime_products = RuntimeProducts()
        return result

    @staticmethod
    def _normalize_terminal_result(
        result: FrameExecutionResult,
        cancel_event: asyncio.Event,
    ) -> FrameExecutionResult:
        if cancel_event.is_set() and result.status != FrameExecutionStatus.CANCELLED:
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        return result

    def _register_call(
        self,
        frame: ExecutionFrame,
        suspension: FrameExecutionResult,
    ) -> CallRecord:
        action_id = suspension.suspend_action_id
        if not action_id:
            raise ValueError("Suspended frame is missing a CALL action id.")
        return self._session.register_call(frame, action_id)


__all__ = ["RunDriver"]
