"""Gateway 固定拓扑 workflow。"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from hivememory.core.models import WorkspaceAccessContext, require_workspace_access_context
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewayProcessResult,
    GatewayTimeoutError,
)
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.gateway.workflow.state import GatewayExecutionState
from hivememory.gateway.workflow.steps import (
    GatewayStepResult,
    GatewayWorkflowStep,
    RecoverableGatewayError,
)
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink


class GatewayWorkflow:
    """按固定顺序执行 Gateway 原子状态转换。"""

    def __init__(
        self,
        *,
        entry_step: GatewayWorkflowStep[Any, Any],
        command_dispatch_step: GatewayWorkflowStep[Any, Any],
        decision_prefix: tuple[GatewayWorkflowStep[Any, Any], ...],
        simple_chat_defaults_step: GatewayWorkflowStep[Any, Any],
        user_query_analysis_step: GatewayWorkflowStep[Any, Any],
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._entry_step = entry_step
        self._command_dispatch_step = command_dispatch_step
        self._decision_prefix = decision_prefix
        self._simple_chat_defaults_step = simple_chat_defaults_step
        self._user_query_analysis_step = user_query_analysis_step
        self._runtime_events = runtime_events or NullRuntimeEventSink()

    async def run(
        self,
        message: str,
        *,
        access_context: WorkspaceAccessContext,
        ingress_mode: GatewayIngressMode,
        request_timeout_ms: int | None = None,
    ) -> GatewayProcessResult:
        """执行 Entry、固定 topic 前缀和唯一 analysis 分支。"""

        started_at = perf_counter()
        state = GatewayExecutionState(
            raw_message=message,
            access_context=require_workspace_access_context(access_context),
            ingress_mode=ingress_mode,
        )
        current_step_id: str | None = None
        completed_steps = 0
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + request_timeout_ms / 1000
            if request_timeout_ms is not None
            else None
        )
        self._emit(
            RuntimeEventType.GATEWAY_WORKFLOW_STARTED,
            data={"ingress_mode": ingress_mode.value},
        )

        try:
            current_step_id = self._entry_step.step_id
            deadline_reached = await self._run_step(
                state,
                self._entry_step,
                completed_steps,
                deadline=deadline,
            )
            if deadline_reached:
                deadline = 0.0
            completed_steps += 1

            if state.flow_end_reason is not None:
                if state.flow_end_reason != "system_command":
                    raise RuntimeError(
                        f"不支持的 Gateway terminal branch: {state.flow_end_reason}"
                    )
                current_step_id = self._command_dispatch_step.step_id
                deadline_reached = await self._run_step(
                    state,
                    self._command_dispatch_step,
                    completed_steps,
                    deadline=deadline,
                )
                if deadline_reached:
                    deadline = 0.0
                completed_steps += 1
                outcome = state.finalize()
                self._emit_completed(outcome, completed_steps, started_at)
                return outcome

            for step in self._decision_prefix:
                current_step_id = step.step_id
                deadline_reached = await self._run_step(
                    state,
                    step,
                    completed_steps,
                    deadline=deadline,
                )
                if deadline_reached:
                    deadline = 0.0
                completed_steps += 1

            analysis_step = (
                self._simple_chat_defaults_step
                if state.l1_result is not None
                and state.l1_result.intent == GatewayIntent.CHAT
                else self._user_query_analysis_step
            )
            current_step_id = analysis_step.step_id
            deadline_reached = await self._run_step(
                state,
                analysis_step,
                completed_steps,
                deadline=deadline,
            )
            if deadline_reached:
                deadline = 0.0
            completed_steps += 1

            outcome = state.finalize()
            self._emit_completed(outcome, completed_steps, started_at)
            return outcome
        except Exception as exc:
            self._emit(
                RuntimeEventType.GATEWAY_WORKFLOW_FAILED,
                severity="error",
                reason=type(exc).__name__,
                message=str(exc),
                data={
                    "step_id": current_step_id,
                    "completed_steps": completed_steps,
                    "duration_ms": _elapsed_ms(started_at),
                },
            )
            raise

    async def _run_step(
        self,
        state: GatewayExecutionState,
        step: GatewayWorkflowStep[Any, Any],
        step_index: int,
        *,
        deadline: float | None = None,
    ) -> bool:
        """统一执行一次 Step，并保证只有一次 state apply。"""

        step_started_at = perf_counter()
        selected_input = step.select_input(state.snapshot())
        is_fallback = False
        fallback_reason: str | None = None

        remaining_seconds = self._remaining_seconds(deadline)
        if remaining_seconds is not None and remaining_seconds <= 0:
            if step.fallback is None:
                raise GatewayTimeoutError(
                    f"Gateway deadline 已耗尽，且 Step 无本地 fallback: {step.step_id}"
                )
            timeout_error = TimeoutError("Gateway request deadline exceeded")
            updates = step.fallback(selected_input, timeout_error)
            flow_end_reason = None
            is_fallback = True
            fallback_reason = "GatewayTimeoutError"
            state._apply_step_result(
                GatewayStepResult(
                    updates=updates,
                    flow_end_reason=flow_end_reason,
                    is_fallback=is_fallback,
                    fallback_reason=fallback_reason,
                )
            )
            self._emit_step_completed(
                step,
                step_index,
                step_started_at,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
                flow_end_reason=flow_end_reason,
            )
            return True

        step_timeout_seconds = (
            step.timeout_ms / 1000 if step.timeout_ms is not None else None
        )
        effective_timeout = _minimum_timeout(
            step_timeout_seconds,
            remaining_seconds,
        )
        deadline_limited = (
            remaining_seconds is not None
            and (
                step_timeout_seconds is None
                or remaining_seconds <= step_timeout_seconds
            )
        )

        try:
            invocation = step.invoke(selected_input)
            output = (
                await asyncio.wait_for(invocation, timeout=effective_timeout)
                if effective_timeout is not None
                else await invocation
            )
        except (TimeoutError, RecoverableGatewayError) as exc:
            if step.fallback is None:
                if isinstance(exc, TimeoutError) and deadline_limited:
                    raise GatewayTimeoutError(
                        f"Gateway deadline 内无法完成 Step: {step.step_id}"
                    ) from exc
                raise
            updates = step.fallback(selected_input, exc)
            flow_end_reason = None
            is_fallback = True
            fallback_reason = (
                "GatewayTimeoutError"
                if isinstance(exc, TimeoutError) and deadline_limited
                else type(exc).__name__
            )
        else:
            updates = step.project(output)
            flow_end_reason = (
                step.resolve_flow_end(output)
                if step.resolve_flow_end is not None
                else None
            )

        state._apply_step_result(
            GatewayStepResult(
                updates=updates,
                flow_end_reason=flow_end_reason,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
            )
        )
        self._emit_step_completed(
            step,
            step_index,
            step_started_at,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            flow_end_reason=flow_end_reason,
        )
        return is_fallback and fallback_reason == "GatewayTimeoutError"

    @staticmethod
    def _remaining_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - asyncio.get_running_loop().time()

    def _emit_step_completed(
        self,
        step: GatewayWorkflowStep[Any, Any],
        step_index: int,
        step_started_at: float,
        *,
        is_fallback: bool,
        fallback_reason: str | None,
        flow_end_reason: str | None,
    ) -> None:
        self._emit(
            RuntimeEventType.GATEWAY_STEP_COMPLETED,
            data={
                "step_id": step.step_id,
                "step_index": step_index,
                "duration_ms": _elapsed_ms(step_started_at),
                "is_fallback": is_fallback,
                "fallback_reason": fallback_reason,
                "flow_ended": flow_end_reason is not None,
            },
        )

    def _emit_completed(
        self,
        outcome: GatewayProcessResult,
        completed_steps: int,
        started_at: float,
    ) -> None:
        data: dict[str, Any] = {
            "duration_ms": _elapsed_ms(started_at),
            "completed_steps": completed_steps,
            "outcome_kind": outcome.kind,
        }
        if outcome.kind == "decision":
            data.update(
                {
                    "topic_id": outcome.decision.target_topic_id,
                    "intent_type": outcome.decision.intent_type.value,
                    "retrieval_mode": outcome.decision.retrieval_plan.mode.value,
                }
            )
        else:
            data["command_id"] = outcome.command_execution_result.command_id
        self._emit(RuntimeEventType.GATEWAY_WORKFLOW_COMPLETED, data=data)

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        severity: str = "info",
        reason: str | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """事件仅用于观测，sink 异常不得改变业务结果。"""

        try:
            self._runtime_events.emit(
                RuntimeEvent(
                    event_type=event_type,
                    subsystem="gateway",
                    component="workflow",
                    severity=severity,
                    reason=reason,
                    message=message,
                    data=data or {},
                )
            )
        except Exception:
            return


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000)


def _minimum_timeout(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


__all__ = ["GatewayWorkflow"]
