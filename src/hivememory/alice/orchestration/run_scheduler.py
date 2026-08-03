from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.alice.orchestration.call_coordinator import CallNextAction
from hivememory.alice.orchestration.call_record import CallRecord
from hivememory.alice.orchestration.run_output import AgentRunOutput, NullAgentRunOutput
from hivememory.alice.orchestration.run_session import (
    FrameSchedulingStatus,
    RunSession,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.runtime import AgentRuntime
    from hivememory.alice.orchestration.call_coordinator import CallCoordinator


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

    async def run(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        run_output: AgentRunOutput | None = None,
    ) -> FrameExecutionResult:
        self._require_root(frame)
        return await self._drive(
            frame,
            generation_options=generation_options,
            run_output=run_output or NullAgentRunOutput(),
        )

    async def _drive(
        self,
        root_frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None,
        run_output: AgentRunOutput,
    ) -> FrameExecutionResult:
        """
        用同一循环推进 root 与当前唯一的 callee。
        """

        root_frame_id = root_frame.runtime_scope.frame_id
        self._session.transition_frame(root_frame_id, FrameSchedulingStatus.RUNNABLE)
        current_frame = root_frame
        pending_suspension: FrameExecutionResult | None = None

        while True:
            current_frame_id = current_frame.runtime_scope.frame_id
            self._session.transition_frame(current_frame_id, FrameSchedulingStatus.RUNNING)

            # 为当前 frame 注册可观测事件 scope
            if current_frame is root_frame:
                frame_output = run_output.for_frame(
                    current_frame,
                    action_id=None,
                    scope="main",
                    depth=0,
                )
            else:
                if pending_suspension is None or self._call_coordinator is None:
                    raise RuntimeError("Callee frame is missing its pending CALL suspension.")
                frame_output = run_output.for_frame(
                    current_frame,
                    action_id=pending_suspension.suspend_action_id,
                    scope="sub",
                    depth=1,
                )

            # 运行当前 frame
            try:
                result = await self._agent_runtime.run_frame(
                    current_frame,
                    generation_options=generation_options,
                    output_sink=frame_output,
                    cancel_event=self._session.cancel_event,
                )
            except asyncio.CancelledError:
                self._abort_cancelled_run(root_frame, pending_suspension)
                raise

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
                    pending_suspension = result
                    try:
                        transition = await self._begin_call(
                            root_frame,
                            result,
                            generation_options=generation_options,
                            run_output=run_output,
                        )
                    except asyncio.CancelledError:
                        self._abort_cancelled_run(root_frame, pending_suspension)
                        raise
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
                        current_frame = next_frame
                        continue
                    if transition.action == CallNextAction.RESUME_CALLER:
                        self._session.transition_frame(
                            root_frame_id,
                            FrameSchedulingStatus.RUNNABLE,
                        )
                        pending_suspension = None
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

            try:
                transition = await self._complete_call(
                    root_frame,
                    pending_suspension,
                    current_frame,
                    result,
                    generation_options=generation_options,
                    run_output=run_output,
                )
            except asyncio.CancelledError:
                self._abort_cancelled_run(root_frame, pending_suspension)
                raise

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
        run_output: AgentRunOutput,
    ):
        if self._call_coordinator is None:
            raise RuntimeError("CALL coordinator is unavailable.")
        return await self._call_coordinator.begin_call(
            frame,
            suspension,
            session=self._session,
            generation_options=generation_options,
            run_output=run_output,
        )

    async def _complete_call(
        self,
        root_frame: ExecutionFrame,
        suspension: FrameExecutionResult | None,
        callee_frame: ExecutionFrame,
        result: FrameExecutionResult,
        *,
        generation_options: dict[str, Any] | None,
        run_output: AgentRunOutput,
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
            run_output=run_output,
        )

    def _cancel_root(self, root_frame_id: str) -> FrameExecutionResult:
        status = self._session.frame_statuses[root_frame_id]
        if status != FrameSchedulingStatus.TERMINATED:
            self._session.transition_frame(
                root_frame_id,
                FrameSchedulingStatus.TERMINATED,
            )
        return self._finish(FrameExecutionResult(status=FrameExecutionStatus.CANCELLED))

    def _abort_cancelled_run(
        self,
        root_frame: ExecutionFrame,
        suspension: FrameExecutionResult | None,
    ) -> None:
        """协程取消时只清理当前 Session，并保持 frame/run exactly-once 收尾。"""
        self._session.cancel_event.set()
        if suspension is not None and self._call_coordinator is not None:
            action_id = suspension.suspend_action_id
            key = (root_frame.runtime_scope.frame_id, action_id)
            record = self._session.call_records.get(key) if action_id else None
            if record is not None:
                self._call_coordinator.cancel_call(
                    root_frame,
                    suspension,
                    session=self._session,
                )
                if record.callee_frame_id is not None:
                    self._terminate_frame_if_needed(record.callee_frame_id)

        self._session.cancel_unapplied_calls()
        self._terminate_frame_if_needed(root_frame.runtime_scope.frame_id)
        if self.terminal_result is None:
            self._finish(FrameExecutionResult(status=FrameExecutionStatus.CANCELLED))

    def _terminate_frame_if_needed(self, frame_id: str) -> None:
        status = self._session.frame_statuses.get(frame_id)
        if status is not None and status != FrameSchedulingStatus.TERMINATED:
            self._session.transition_frame(frame_id, FrameSchedulingStatus.TERMINATED)

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
