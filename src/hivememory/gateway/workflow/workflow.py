"""Gateway 固定拓扑 workflow。"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.contracts.runtime_events import (
    RuntimeEventSeverity,
    RuntimeEventType,
)
from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewayProcessResult,
    GatewaySessionTurn,
    GatewayTimeoutError,
)
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.gateway.workflow.state import GatewayExecutionState
from hivememory.gateway.workflow.steps import (
    GatewayStepResult,
    GatewayWorkflowStep,
    RecoverableGatewayError,
)

logger = logging.getLogger(__name__)


class GatewayWorkflow:
    """按固定顺序执行 Gateway 原子状态转换。"""

    def __init__(
        self,
        *,
        entry_step: GatewayWorkflowStep[Any, Any],
        decision_prefix: tuple[GatewayWorkflowStep[Any, Any], ...],
        simple_chat_defaults_step: GatewayWorkflowStep[Any, Any],
        user_query_analysis_step: GatewayWorkflowStep[Any, Any],
        runtime_events: RuntimeEventPublisher | None = None,
    ) -> None:
        self._entry_step = entry_step
        self._decision_prefix = decision_prefix
        self._simple_chat_defaults_step = simple_chat_defaults_step
        self._user_query_analysis_step = user_query_analysis_step
        self._runtime_events = (
            runtime_events or RuntimeEventPublisher(NullRuntimeEventSink())
        ).scoped(subsystem="gateway", component="workflow")

    async def run(
        self,
        message: str,
        *,
        identity_scope: IdentityScope,
        ingress_mode: GatewayIngressMode,
        request_timeout_ms: int | None = None,
        session_turns: tuple[GatewaySessionTurn, ...] = (),
    ) -> GatewayProcessResult:
        """执行 Entry、固定 topic 前缀和唯一 analysis 分支。

        ``identity_scope`` 是调用方冻结的执行坐标，随状态贯穿各步骤并原
        样传给话题读取；Gateway 不接收工作 actor，也不做授权判断。
        """

        started_at = perf_counter()
        state = GatewayExecutionState(
            raw_message=message,
            identity_scope=require_identity_scope(identity_scope),
            ingress_mode=ingress_mode,
            session_turns=session_turns,
        )
        workspace_id = state.identity_scope.workspace_identity.workspace_id
        current_step_id: str | None = None
        completed_steps = 0
        # 每一步的耗时与降级原因（没有降级为 None），只汇总进终态事件，不逐步发事件。
        step_records: list[tuple[str, float, str | None]] = []
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + request_timeout_ms / 1000 if request_timeout_ms is not None else None
        )
        self._emit(
            RuntimeEventType.GATEWAY_WORKFLOW_STARTED,
            workspace_id=workspace_id,
            data={"ingress_mode": ingress_mode.value},
        )

        try:
            current_step_id = self._entry_step.step_id
            deadline_reached = await self._run_step(
                state,
                self._entry_step,
                step_records,
                deadline=deadline,
            )
            if deadline_reached:
                deadline = 0.0
            completed_steps += 1

            if state.flow_end_reason is not None:
                if state.flow_end_reason != "system_command":
                    raise RuntimeError(f"不支持的 Gateway terminal branch: {state.flow_end_reason}")
                # 命令只解析不执行：入口拦截识别出命令后直接以命令结果收口，
                # 不再运行任何执行步骤。
                outcome = state.finalize()
                self._emit_completed(
                    outcome,
                    step_records,
                    started_at,
                    workspace_id=workspace_id,
                )
                return outcome

            for step in self._decision_prefix:
                current_step_id = step.step_id
                deadline_reached = await self._run_step(
                    state,
                    step,
                    step_records,
                    deadline=deadline,
                )
                if deadline_reached:
                    deadline = 0.0
                completed_steps += 1

            # 被动记忆入口不检索记忆，查询分析的产物（重写、关键词与检索计划）没有
            # 使用方：与简单寒暄一样以不检索的默认值收口，省去一次模型调用。
            use_defaults = ingress_mode == GatewayIngressMode.PASSIVE_MEMORY or (
                state.l1_result is not None and state.l1_result.intent == GatewayIntent.CHAT
            )
            analysis_step = (
                self._simple_chat_defaults_step if use_defaults else self._user_query_analysis_step
            )
            current_step_id = analysis_step.step_id
            deadline_reached = await self._run_step(
                state,
                analysis_step,
                step_records,
                deadline=deadline,
            )
            if deadline_reached:
                deadline = 0.0
            completed_steps += 1

            outcome = state.finalize()
            self._emit_completed(
                outcome,
                step_records,
                started_at,
                workspace_id=workspace_id,
            )
            return outcome
        except Exception as exc:
            self._emit(
                RuntimeEventType.GATEWAY_WORKFLOW_FAILED,
                workspace_id=workspace_id,
                severity="error",
                reason=(
                    "gateway_timeout"
                    if isinstance(exc, (TimeoutError, GatewayTimeoutError))
                    else (
                        "dependency_unavailable"
                        if isinstance(exc, RecoverableGatewayError)
                        else "workflow_failed"
                    )
                ),
                message="Gateway 处理失败",
                data={
                    "error_class": type(exc).__name__,
                    "step_id": current_step_id,
                    "completed_steps": completed_steps,
                    "duration_ms": _elapsed_ms(started_at),
                    **_step_summary(step_records),
                },
            )
            raise

    async def _run_step(
        self,
        state: GatewayExecutionState,
        step: GatewayWorkflowStep[Any, Any],
        step_records: list[tuple[str, float, str | None]],
        *,
        deadline: float | None = None,
    ) -> bool:
        """统一执行一次 Step，并保证只有一次 state apply；耗时与降级原因记入 ``step_records``。"""

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
            fallback_reason = "request_deadline_exceeded"
            state._apply_step_result(
                GatewayStepResult(
                    updates=updates,
                    flow_end_reason=flow_end_reason,
                    is_fallback=is_fallback,
                    fallback_reason=fallback_reason,
                )
            )
            step_records.append((step.step_id, _elapsed_ms(step_started_at), fallback_reason))
            return True

        step_timeout_seconds = step.timeout_ms / 1000 if step.timeout_ms is not None else None
        effective_timeout = _minimum_timeout(
            step_timeout_seconds,
            remaining_seconds,
        )
        deadline_limited = remaining_seconds is not None and (
            step_timeout_seconds is None or remaining_seconds <= step_timeout_seconds
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
                "request_deadline_exceeded"
                if isinstance(exc, TimeoutError) and deadline_limited
                else "step_timeout" if isinstance(exc, TimeoutError) else "dependency_unavailable"
            )
            # 降级的原因码在终态事件的步骤汇总里；底层异常只在这里记录。
            logger.warning(
                "Gateway 步骤降级: step_id=%s, reason=%s, cause=%r",
                step.step_id,
                fallback_reason,
                exc.__cause__ or exc,
            )
        else:
            updates = step.project(output)
            flow_end_reason = (
                step.resolve_flow_end(output) if step.resolve_flow_end is not None else None
            )

        state._apply_step_result(
            GatewayStepResult(
                updates=updates,
                flow_end_reason=flow_end_reason,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
            )
        )
        step_records.append((step.step_id, _elapsed_ms(step_started_at), fallback_reason))
        return is_fallback and fallback_reason == "request_deadline_exceeded"

    @staticmethod
    def _remaining_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - asyncio.get_running_loop().time()

    def _emit_completed(
        self,
        outcome: GatewayProcessResult,
        step_records: list[tuple[str, float, str | None]],
        started_at: float,
        *,
        workspace_id: str,
    ) -> None:
        data: dict[str, Any] = {
            "duration_ms": _elapsed_ms(started_at),
            "completed_steps": len(step_records),
            "outcome_kind": outcome.kind,
            **_step_summary(step_records),
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
            data["command_id"] = outcome.command_parse_result.effective_command_id
        self._emit(
            RuntimeEventType.GATEWAY_WORKFLOW_COMPLETED,
            workspace_id=workspace_id,
            data=data,
        )

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        severity: RuntimeEventSeverity = "info",
        reason: str | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """事件仅用于观测，发布失败由 publisher 吞下，不改变业务结果。"""
        self._runtime_events.bind(workspace_id=workspace_id).emit(
            event_type, severity=severity, reason=reason, message=message, data=data
        )


def _step_summary(step_records: list[tuple[str, float, str | None]]) -> dict[str, Any]:
    """终态事件中的各步耗时与发生过的降级；降级很少出现，平时 ``fallbacks`` 为空列表。"""
    return {
        "step_durations_ms": {step_id: duration for step_id, duration, _ in step_records},
        "fallbacks": [
            {"step_id": step_id, "reason": reason}
            for step_id, _, reason in step_records
            if reason is not None
        ],
    }


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000)


def _minimum_timeout(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


__all__ = ["GatewayWorkflow"]
