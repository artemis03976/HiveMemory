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
from hivememory.alice.runtime.agent.call_coordinator import CallNextAction
from hivememory.alice.runtime.agent.call_record import CallRecord
from hivememory.alice.runtime.agent.run_session import (
    FrameSchedulingStatus,
    RunSession,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.call_coordinator import CallCoordinator
    from hivememory.alice.runtime.agent.runtime import AgentRuntime


class RunScheduler:
    """每次 Alice run 独立创建的单活动 frame 调度状态机。"""

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
        self._require_root(frame)
        return await self._drive(
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
            self._require_root(frame)
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
            sink = QueueFrameEventSink(
                queue,
                metadata=event_metadata,
                sequence_start=self._session.stream_sequence,
            )

            async def _runner() -> None:
                cancelled = False
                try:
                    await self._drive(
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

    async def _drive(
        self,
        root_frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None,
        event_sink: FrameEventSink,
    ) -> FrameExecutionResult:
        """用同一循环推进 root 与当前唯一的 callee。"""
        root_frame_id = root_frame.runtime_scope.frame_id
        self._session.transition_frame(root_frame_id, FrameSchedulingStatus.RUNNABLE)
        current_frame = root_frame
        pending_suspension: FrameExecutionResult | None = None

        while True:
            current_frame_id = current_frame.runtime_scope.frame_id
            self._session.transition_frame(current_frame_id, FrameSchedulingStatus.RUNNING)
            current_sink = event_sink
            if current_frame is not root_frame:
                if pending_suspension is None or self._call_coordinator is None:
                    raise RuntimeError("Callee frame is missing its pending CALL suspension.")
                current_sink = self._call_coordinator.event_sink_for_callee(
                    current_frame,
                    pending_suspension.suspend_action_id,
                    event_sink,
                )

            result = await self._agent_runtime.run_frame(
                current_frame,
                generation_options=generation_options,
                event_sink=current_sink,
                cancel_event=self._session.cancel_event,
            )

            if current_frame is root_frame:
                if result.status == FrameExecutionStatus.SUSPENDED:
                    if self._call_coordinator is None:
                        self._session.transition_frame(
                            root_frame_id,
                            FrameSchedulingStatus.TERMINATED,
                        )
                        return self._finish(
                            FrameExecutionResult(
                                status=FrameExecutionStatus.FAILED,
                                error=RuntimeError("Frame suspended without a CALL coordinator."),
                            )
                        )
                    transition = await self._begin_call(
                        root_frame,
                        result,
                        generation_options=generation_options,
                        event_sink=event_sink,
                    )
                    if transition.action == CallNextAction.DISPATCH_CALLEE:
                        next_frame = self._require_transition_frame(transition.next_frame)
                        self._session.transition_frame(
                            root_frame_id,
                            FrameSchedulingStatus.WAITING,
                        )
                        self._session.transition_frame(
                            next_frame.runtime_scope.frame_id,
                            FrameSchedulingStatus.RUNNABLE,
                        )
                        pending_suspension = result
                        current_frame = next_frame
                        continue
                    if transition.action == CallNextAction.RESUME_CALLER:
                        self._session.transition_frame(
                            root_frame_id,
                            FrameSchedulingStatus.RUNNABLE,
                        )
                        continue
                    return self._cancel_root(root_frame_id)

                if result.status in {
                    FrameExecutionStatus.COMPLETED,
                    FrameExecutionStatus.CANCELLED,
                    FrameExecutionStatus.FAILED,
                    FrameExecutionStatus.BUDGET_EXHAUSTED,
                }:
                    terminal_result = self._normalize_terminal_result(
                        result,
                        self._session.cancel_event,
                    )
                    self._session.transition_frame(
                        root_frame_id,
                        FrameSchedulingStatus.TERMINATED,
                    )
                    return self._finish(terminal_result)
                raise RuntimeError(
                    f"RunScheduler received an unsupported root status: {result.status!r}"
                )

            transition = await self._complete_call(
                root_frame,
                pending_suspension,
                current_frame,
                result,
                generation_options=generation_options,
                event_sink=event_sink,
            )
            self._session.transition_frame(
                current_frame_id,
                FrameSchedulingStatus.TERMINATED,
            )
            if transition.action == CallNextAction.RESUME_CALLER:
                self._session.transition_frame(
                    root_frame_id,
                    FrameSchedulingStatus.RUNNABLE,
                )
                pending_suspension = None
                current_frame = root_frame
                continue
            if transition.action == CallNextAction.CANCEL_RUN:
                return self._cancel_root(root_frame_id)
            raise RuntimeError(
                "Call completion attempted to dispatch another callee in a single-layer run."
            )

    async def _begin_call(
        self,
        frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        generation_options: dict[str, Any] | None,
        event_sink: FrameEventSink,
    ):
        if self._call_coordinator is None:
            raise RuntimeError("CALL coordinator is unavailable.")
        return await self._call_coordinator.begin_call(
            frame,
            suspension,
            session=self._session,
            generation_options=generation_options,
            event_sink=event_sink,
        )

    async def _complete_call(
        self,
        root_frame: ExecutionFrame,
        suspension: FrameExecutionResult | None,
        callee_frame: ExecutionFrame,
        result: FrameExecutionResult,
        *,
        generation_options: dict[str, Any] | None,
        event_sink: FrameEventSink,
    ):
        if suspension is None or self._call_coordinator is None:
            raise RuntimeError("Callee completion is missing its CALL coordination context.")
        return await self._call_coordinator.complete_call(
            root_frame,
            suspension,
            callee_frame,
            result,
            session=self._session,
            generation_options=generation_options,
            event_sink=event_sink,
        )

    def _cancel_root(self, root_frame_id: str) -> FrameExecutionResult:
        status = self._session.frame_statuses[root_frame_id]
        if status != FrameSchedulingStatus.TERMINATED:
            self._session.transition_frame(
                root_frame_id,
                FrameSchedulingStatus.TERMINATED,
            )
        return self._finish(FrameExecutionResult(status=FrameExecutionStatus.CANCELLED))

    def _finish(self, result: FrameExecutionResult) -> FrameExecutionResult:
        if self.terminal_result is not None:
            raise RuntimeError("RunScheduler attempted to finalize a run more than once.")
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

    def _require_root(self, frame: ExecutionFrame) -> None:
        self._session.require_frame(frame)
        if self._session.root_frame_id != frame.runtime_scope.frame_id:
            raise ValueError(
                f"Frame {frame.runtime_scope.frame_id!r} is not the root of "
                f"RunSession {self._session.agent_run_id!r}."
            )
        self._session.require_frame_status(
            frame.runtime_scope.frame_id,
            FrameSchedulingStatus.PENDING,
        )

    @staticmethod
    def _require_transition_frame(frame: ExecutionFrame | None) -> ExecutionFrame:
        if frame is None:
            raise RuntimeError("CALL transition is missing its next frame.")
        return frame

    @staticmethod
    def _normalize_terminal_result(
        result: FrameExecutionResult,
        cancel_event: asyncio.Event,
    ) -> FrameExecutionResult:
        if cancel_event.is_set() and result.status != FrameExecutionStatus.CANCELLED:
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        return result


__all__ = ["RunScheduler"]
