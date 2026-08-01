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
        call_coordinator: CallCoordinator | None = None,
        session: RunSession | None = None,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_coordinator = call_coordinator
        self._session = session
        self.terminal_result: FrameExecutionResult | None = None
        self.runtime_products: RuntimeProducts | None = None

    @property
    def call_records(self) -> dict[tuple[str, str], CallRecord]:
        return self._ensure_session(None).call_records

    @property
    def next_stream_sequence(self) -> int:
        return self._ensure_session(None).stream_sequence

    async def run(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> FrameExecutionResult:
        session = self._ensure_session(frame, cancel_event=cancel_event)
        self._register_frame_if_supported(session, frame)
        cancel_event = session.cancel_event
        return await self._run_until_terminal(
            frame,
            generation_options=generation_options,
            cancel_event=cancel_event,
            event_sink=NullFrameEventSink(),
            session=session,
        )

    async def _run_until_terminal(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None,
        cancel_event: asyncio.Event,
        event_sink: FrameEventSink,
        session: RunSession,
    ) -> FrameExecutionResult:
        """Drive one root frame until it reaches a terminal outcome."""
        while True:
            result = await self._agent_runtime.run_frame(
                frame,
                generation_options=generation_options,
                event_sink=event_sink,
                cancel_event=cancel_event,
            )
            if result.status != FrameExecutionStatus.SUSPENDED:
                result = self._normalize_terminal_result(result, session.cancel_event)
                return self._finish(result)
            if self._call_coordinator is None:
                result = FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED,
                    error=RuntimeError("Frame suspended without an orchestration callback."),
                )
                return self._finish(result)
            record = self._register_call(frame, result)
            record.begin_resolution()
            response = await self._call_coordinator.resolve_call(
                frame,
                result,
                generation_options=generation_options,
                cancel_event=cancel_event,
                event_sink=event_sink,
                session=session,
            )
            record.mark_resolved()
            if cancel_event.is_set():
                record.cancel()
                continue
            self._agent_runtime.apply_call_response(frame, result, response)
            record.mark_applied()

    def run_stream(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        event_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def _stream() -> AsyncGenerator[dict[str, Any], None]:
            session = self._ensure_session(frame, cancel_event=cancel_event)
            self._register_frame_if_supported(session, frame)
            active_cancel_event = session.cancel_event
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
            sink = QueueFrameEventSink(
                queue,
                metadata=event_metadata,
                sequence_start=session.stream_sequence,
            )

            async def _runner() -> None:
                cancelled = False
                try:
                    await self._run_until_terminal(
                        frame,
                        generation_options=generation_options,
                        cancel_event=active_cancel_event,
                        event_sink=sink,
                        session=session,
                    )
                except asyncio.CancelledError:
                    cancelled = True
                    session.cancel_event.set()
                    if self.terminal_result is None:
                        self._finish(FrameExecutionResult(status=FrameExecutionStatus.CANCELLED))
                    raise
                finally:
                    session.stream_sequence = sink.next_sequence
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

    def _finish(self, result: FrameExecutionResult) -> FrameExecutionResult:
        if self.terminal_result is not None:
            raise RuntimeError("RunDriver attempted to finalize a run more than once.")
        self.terminal_result = result
        finalize_run = getattr(self._agent_runtime, "finalize_run", None)
        if callable(finalize_run):
            self.runtime_products = finalize_run(
                self._ensure_session(None).agent_run_id,
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
        return self._ensure_session(frame).register_call(frame, action_id)

    def _ensure_session(
        self,
        frame: ExecutionFrame | None,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> RunSession:
        if self._session is None:
            scope = getattr(frame, "runtime_scope", None)
            run_id = getattr(scope, "run_id", "")
            self._session = RunSession(
                agent_run_id=run_id, cancel_event=cancel_event or asyncio.Event()
            )
        elif cancel_event is not None and self._session.cancel_event is not cancel_event:
            raise ValueError("RunDriver received a cancel event different from its RunSession.")
        return self._session

    @staticmethod
    def _register_frame_if_supported(session: RunSession, frame: object) -> None:
        scope = getattr(frame, "runtime_scope", None)
        if getattr(scope, "run_id", None) is not None:
            session.register_frame(frame)


__all__ = ["RunDriver"]
