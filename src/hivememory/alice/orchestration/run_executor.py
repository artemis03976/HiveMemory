from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.alice.orchestration.run_output import AgentRunOutput, NullAgentRunOutput
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent.call_coordinator import (
    CancelRun,
    DispatchCallee,
    ResumeCaller,
)
from hivememory.alice.orchestration.sub_agent.call_record import CallRecord

if TYPE_CHECKING:
    from hivememory.agent_runtime.runtime import AgentRuntime
    from hivememory.alice.orchestration.sub_agent.call_coordinator import CallCoordinator


@dataclass(frozen=True, slots=True)
class _FrameOutputContext:
    """递归 frame 的观测坐标，不参与权限或执行拓扑判断。"""

    action_id: str | None = None
    depth: int = 0

    @property
    def scope(self) -> Literal["main", "sub"]:
        return "main" if self.depth == 0 else "sub"

    def child(self, *, action_id: str) -> _FrameOutputContext:
        return _FrameOutputContext(action_id=action_id, depth=self.depth + 1)


class RunExecutor:
    """递归解释一次 Alice run 中的 frame 与 CALL suspension。"""

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
        self._started = False
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
        if self._started:
            raise RuntimeError("RunExecutor cannot execute the same run more than once.")
        self._started = True

        try:
            result = await self._execute_frame(
                frame,
                generation_options=generation_options,
                run_output=run_output or NullAgentRunOutput(),
                output_context=_FrameOutputContext(),
            )
        except asyncio.CancelledError:
            self._abort_cancelled_run()
            raise

        return self._finish(self._normalize_terminal_result(result, self._session.cancel_event))

    async def _execute_frame(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None,
        run_output: AgentRunOutput,
        output_context: _FrameOutputContext,
    ) -> FrameExecutionResult:
        """运行并重入同一 frame；CALL child 由本方法递归执行。"""
        self._session.require_frame(frame)
        frame_output = run_output.for_frame(
            frame,
            action_id=output_context.action_id,
            scope=output_context.scope,
            depth=output_context.depth,
        )

        while True:
            result = await self._agent_runtime.run_frame(
                frame,
                generation_options=generation_options,
                output_sink=frame_output,
                cancel_event=self._session.cancel_event,
            )

            match result.status:
                # 进入 Agent Run Subagent CALL 执行
                case FrameExecutionStatus.SUSPENDED:
                    if self._call_coordinator is None:
                        return FrameExecutionResult(
                            status=FrameExecutionStatus.FAILED,
                            error=RuntimeError("Frame suspended without a CALL coordinator."),
                        )

                    outcome = await self._execute_call(
                        frame,
                        result,
                        generation_options=generation_options,
                        run_output=run_output,
                        output_context=output_context,
                    )

                    match outcome:
                        case ResumeCaller():
                            continue
                        case CancelRun():
                            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
                        case _:
                            raise RuntimeError(
                                f"CALL execution returned an unsupported outcome: {outcome!r}"
                            )

                # 进入 Agent Run 执行终态
                case (
                    FrameExecutionStatus.COMPLETED
                    | FrameExecutionStatus.CANCELLED
                    | FrameExecutionStatus.FAILED
                    | FrameExecutionStatus.BUDGET_EXHAUSTED
                ):
                    return result

                case unexpected_status:
                    raise RuntimeError(
                        f"RunExecutor received an unsupported frame status: {unexpected_status!r}"
                    )

    async def _execute_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        generation_options: dict[str, Any] | None,
        run_output: AgentRunOutput,
        output_context: _FrameOutputContext,
    ) -> ResumeCaller | CancelRun:
        coordinator = self._require_call_coordinator()
        try:
            outcome = await coordinator.begin_call(
                caller_frame,
                suspension,
                session=self._session,
                run_output=run_output,
            )
            match outcome:
                case DispatchCallee(frame=callee_frame):
                    action_id = self._require_suspension_action_id(suspension)

                    callee_result = await self._execute_frame(
                        callee_frame,
                        generation_options=generation_options,
                        run_output=run_output,
                        output_context=output_context.child(action_id=action_id),
                    )

                    completion = await coordinator.complete_call(
                        caller_frame,
                        suspension,
                        callee_frame,
                        callee_result,
                        session=self._session,
                        generation_options=generation_options,
                        run_output=run_output,
                    )
                    match completion:
                        case ResumeCaller() | CancelRun():
                            return completion
                        case _:
                            raise RuntimeError(
                                f"CALL completion returned an unsupported outcome: {completion!r}"
                            )
                case ResumeCaller() | CancelRun():
                    return outcome
                case _:
                    raise RuntimeError(
                        f"CALL preparation returned an unsupported outcome: {outcome!r}"
                    )
        except asyncio.CancelledError:
            self._cancel_call_if_registered(caller_frame, suspension)
            raise

    def _cancel_call_if_registered(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
    ) -> None:
        action_id = suspension.suspend_action_id
        key = (caller_frame.runtime_scope.frame_id, action_id) if action_id else None
        if key is None or key not in self._session.call_records:
            return
        self._require_call_coordinator().cancel_call(
            caller_frame,
            suspension,
            session=self._session,
        )

    def _abort_cancelled_run(self) -> None:
        """协程取消沿递归栈清理 CALL 后，在最外层收尾整个 run。"""
        self._session.cancel_event.set()
        self._session.cancel_unapplied_calls()
        if self.terminal_result is None:
            self._finish(FrameExecutionResult(status=FrameExecutionStatus.CANCELLED))

    def _finish(self, result: FrameExecutionResult) -> FrameExecutionResult:
        if self.terminal_result is not None:
            raise RuntimeError("RunExecutor attempted to finalize a run more than once.")
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

    def _require_call_coordinator(self) -> CallCoordinator:
        if self._call_coordinator is None:
            raise RuntimeError("CALL coordinator is unavailable.")
        return self._call_coordinator

    @staticmethod
    def _require_suspension_action_id(suspension: FrameExecutionResult) -> str:
        action_id = suspension.suspend_action_id
        if not action_id:
            raise RuntimeError("CALL suspension is missing its action id.")
        return action_id

    @staticmethod
    def _normalize_terminal_result(
        result: FrameExecutionResult,
        cancel_event: asyncio.Event,
    ) -> FrameExecutionResult:
        if cancel_event.is_set() and result.status != FrameExecutionStatus.CANCELLED:
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        return result


__all__ = ["RunExecutor"]
