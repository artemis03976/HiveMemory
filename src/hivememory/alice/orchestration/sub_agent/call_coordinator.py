from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import FrameProducts
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_output import (
    AgentRunOutput,
    CallOutputFinished,
    CallOutputStarted,
    NullAgentRunOutput,
)
from hivememory.alice.orchestration.sub_agent.call_context_provider import CallContextProvider
from hivememory.alice.orchestration.sub_agent.call_record import CallRecordStatus
from hivememory.core.mtp import MTPCallResponse, MTPResponseStatus
from hivememory.core.mtp.exceptions import (
    AgentModelUnavailableError,
    MTPError,
    SubAgentBudgetExhaustedError,
    SubAgentExecutionError,
    SubAgentUnexpectedSuspendError,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.runtime import AgentRuntime
    from hivememory.alice.orchestration.run_session import RunSession
    from hivememory.core.models import AgentProfile
    from hivememory.prompts.assembler import AgentPromptAssembler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchCallee:
    """CALL 已准备好 callee，由执行器递归运行该 frame。"""

    frame: ExecutionFrame


@dataclass(frozen=True)
class ResumeCaller:
    """CALL 已完成回填，执行器可以重入原 caller。"""


@dataclass(frozen=True)
class CancelRun:
    """全局取消在 CALL 提交前胜出，终止当前 run。"""


type CallStartResult = DispatchCallee | ResumeCaller | CancelRun
type CallCompletionResult = ResumeCaller | CancelRun


class CallCoordinator:
    """把 CALL suspension 转换为 callee dispatch 或 caller resume。"""

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        call_context_provider: CallContextProvider,
        *,
        frame_factory: FrameFactory,
        prompt_assembler: AgentPromptAssembler,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_context_provider = call_context_provider
        self._frame_factory = frame_factory
        self._prompt_assembler = prompt_assembler

    async def begin_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        session: RunSession,
        generation_options: dict[str, Any] | None = None,
        run_output: AgentRunOutput | None = None,
    ) -> CallStartResult:
        """同步登记 CALL 后准备 callee；准备失败时直接恢复 caller。"""
        output = run_output or NullAgentRunOutput()
        call_request, action_id = self._require_suspension(suspension)
        session.require_frame(caller_frame)
        record = session.register_call(caller_frame, action_id)
        record.begin_resolution()

        if session.cancel_event.is_set():
            return await self._complete_preparation(
                caller_frame,
                suspension,
                self._cancelled_response(call_request.target_alias),
                session=session,
                run_output=output,
            )

        logger.info(
            "CALL suspend: target=%s, task=%r",
            call_request.target_alias,
            call_request.task[:80],
        )

        sub_profile: AgentProfile | None = None
        try:
            call_context = await self._call_context_provider.provide(
                caller_frame,
                call_request,
            )
            if session.cancel_event.is_set():
                return await self._complete_preparation(
                    caller_frame,
                    suspension,
                    self._cancelled_response(call_request.target_alias),
                    session=session,
                    run_output=output,
                )
            sub_profile = call_context.agent_profile
            policy = FrameExecutionPolicy.from_profile(
                sub_profile,
                max_iterations=getattr(self._agent_runtime, "max_iterations", None),
                denied_verbs={"CALL"},
            )
            messages = self._prompt_assembler.build_sub_agent_messages(
                profile=sub_profile,
                task=call_request.task,
                shared_context=call_context.shared_context,
            )
            scope = self._frame_factory.scope(run_id=caller_frame.runtime_scope.run_id)
            sub_frame = self._frame_factory.create(
                FrameSpec(
                    runtime_scope=scope,
                    profile=sub_profile,
                    identity=caller_frame.identity,
                    messages=messages,
                    topic_id=None,
                    execution_policy=policy,
                )
            )
        except MTPError as error:
            logger.warning("CALL rejected for %r: %s", call_request.target_alias, error.code)
            response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=error.to_error_info(),
            )
            return await self._complete_preparation(
                caller_frame,
                suspension,
                response,
                session=session,
                run_output=output,
            )
        except Exception as error:
            logger.error("Sub-agent preparation failed: %s", error, exc_info=True)
            response = self._response_for_exception(
                call_request.target_alias,
                error,
                profile=sub_profile,
                generation_options=generation_options,
            )
            return await self._complete_preparation(
                caller_frame,
                suspension,
                response,
                session=session,
                run_output=output,
            )

        # Session 绑定失败属于编排不变量，不能伪装成 Agent 可消费的 CALL error。
        session.register_callee_frame(sub_frame, record)
        await output.call_started(
            CallOutputStarted(
                agent_id=call_request.target_alias,
                task=call_request.task,
                iteration=caller_frame.progress.iteration,
                action_id=action_id,
                frame_id=sub_frame.runtime_scope.frame_id,
            )
        )
        return DispatchCallee(sub_frame)

    async def complete_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        callee_frame: ExecutionFrame,
        callee_result: FrameExecutionResult,
        *,
        session: RunSession,
        generation_options: dict[str, Any] | None = None,
        run_output: AgentRunOutput | None = None,
    ) -> CallCompletionResult:
        """把 callee outcome 收口为 caller 可消费的 CALL response。"""
        output = run_output or NullAgentRunOutput()
        call_request, action_id = self._require_suspension(suspension)
        session.require_frame(caller_frame)
        session.require_frame(callee_frame)
        record = session.call_for_callee(callee_frame.runtime_scope.frame_id)
        if record.caller_frame_id != caller_frame.runtime_scope.frame_id:
            raise ValueError("Callee CALL record does not belong to the supplied caller frame.")
        if record.action_id != action_id:
            raise ValueError("Callee CALL record action does not match the suspension.")

        response = self._call_response_for_frame(
            call_request,
            callee_result,
            cancelled=session.cancel_event.is_set(),
        )
        if callee_result.status == FrameExecutionStatus.FAILED and callee_result.error is not None:
            from hivememory.system.model_registry import ModelNotFoundError

            if isinstance(callee_result.error, ModelNotFoundError):
                response = self._response_for_exception(
                    call_request.target_alias,
                    callee_result.error,
                    profile=callee_frame.agent_profile,
                    generation_options=generation_options,
                )

        response, effective_result, frame_products = self._finalize_callee(
            callee_frame,
            callee_result,
            response,
            agent_alias=call_request.target_alias,
        )
        if (
            effective_result.status == FrameExecutionStatus.COMPLETED
            and response.status == MTPResponseStatus.SUCCESS
        ):
            response = response.model_copy(
                update={
                    "reply": "".join(callee_frame.progress.text_segments),
                    "artifact_aliases": list(frame_products.artifact_aliases),
                }
            )

        record.mark_resolved()
        await self._emit_call_end(
            caller_frame,
            action_id=action_id,
            agent_alias=call_request.target_alias,
            response=response,
            callee_frame=callee_frame,
            callee_result=effective_result,
            run_output=output,
        )
        return self._apply_or_cancel(
            caller_frame,
            suspension,
            response,
            record=record,
            session=session,
        )

    def cancel_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        session: RunSession,
    ) -> None:
        """协程取消时收尾未 apply 的 CALL，不发布事件也不回填 caller。"""
        _, action_id = self._require_suspension(suspension)
        record = session.require_call(caller_frame, action_id)
        if record.status in {CallRecordStatus.APPLIED, CallRecordStatus.CANCELLED}:
            return

        # RESOLVED 表示 callee 已完成逻辑收尾；只在仍为 RESOLVING 时清理 frame。
        if record.status == CallRecordStatus.RESOLVING and record.callee_frame_id is not None:
            callee_frame = session.frames[record.callee_frame_id]
            try:
                self._agent_runtime.finalize_frame(
                    callee_frame,
                    FrameExecutionResult(status=FrameExecutionStatus.CANCELLED),
                )
            except Exception:
                logger.exception("Failed to finalize cancelled sub-agent frame")
        record.cancel()

    async def _complete_preparation(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        response: MTPCallResponse,
        *,
        session: RunSession,
        run_output: AgentRunOutput,
    ) -> CallCompletionResult:
        """CALL 准备失败/取消的收尾路径：结算 record 并回填错误或取消响应。"""
        call_request, action_id = self._require_suspension(suspension)
        record = session.require_call(caller_frame, action_id)
        record.mark_resolved()
        await self._emit_call_end(
            caller_frame,
            action_id=action_id,
            agent_alias=call_request.target_alias,
            response=response,
            callee_frame=None,
            callee_result=None,
            run_output=run_output,
        )
        return self._apply_or_cancel(
            caller_frame,
            suspension,
            response,
            record=record,
            session=session,
        )

    def _apply_or_cancel(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        response: MTPCallResponse,
        *,
        record,
        session: RunSession,
    ) -> CallCompletionResult:
        """根据取消状态决定：取消整个 run，或 exactly-once 回填 caller。"""
        if session.cancel_event.is_set():
            record.cancel()
            return CancelRun()
        self._agent_runtime.apply_call_response(caller_frame, suspension, response)
        record.mark_applied()
        return ResumeCaller()

    def _finalize_callee(
        self,
        callee_frame: ExecutionFrame,
        callee_result: FrameExecutionResult,
        response: MTPCallResponse,
        *,
        agent_alias: str,
    ) -> tuple[MTPCallResponse, FrameExecutionResult, FrameProducts]:
        frame_products = FrameProducts()
        finalization_result = self._frame_result_for_finalization(callee_result, response)
        try:
            frame_products = self._agent_runtime.finalize_frame(
                callee_frame,
                finalization_result,
            )
        except Exception as error:
            logger.error("Failed to finalize sub-agent frame: %s", error, exc_info=True)
            response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=agent_alias,
                error=SubAgentExecutionError(
                    params={"agent_alias": agent_alias},
                    cause=error,
                ).to_error_info(),
            )
            callee_result = FrameExecutionResult(
                status=FrameExecutionStatus.FAILED,
                error=error,
            )
            if finalization_result.status == FrameExecutionStatus.COMPLETED:
                try:
                    self._agent_runtime.finalize_frame(callee_frame, callee_result)
                except Exception:
                    logger.exception("Failed to clean up sub-agent frame after harvest error")
        return response, callee_result, frame_products

    async def _emit_call_end(
        self,
        caller_frame: ExecutionFrame,
        *,
        action_id: str,
        agent_alias: str,
        response: MTPCallResponse,
        callee_frame: ExecutionFrame | None,
        callee_result: FrameExecutionResult | None,
        run_output: AgentRunOutput,
    ) -> None:
        await run_output.call_finished(
            CallOutputFinished(
                status=response.status.value,
                final_text=(response.reply if response.status == MTPResponseStatus.SUCCESS else ""),
                iteration=caller_frame.progress.iteration,
                action_id=action_id,
                frame_id=(
                    callee_frame.runtime_scope.frame_id if callee_frame is not None else None
                ),
                agent_id=agent_alias,
                terminal_status=(callee_result.status.value if callee_result is not None else None),
                error_code=(response.error.code if response.error is not None else None),
            )
        )

    @staticmethod
    def _require_suspension(
        suspension: FrameExecutionResult,
    ) -> tuple[Any, str]:
        call_request = suspension.call_request
        action_id = suspension.suspend_action_id
        if suspension.status != FrameExecutionStatus.SUSPENDED or call_request is None:
            raise ValueError("CALL coordinator requires a suspended frame result.")
        if not action_id:
            raise ValueError("CALL suspension is missing its action id.")
        return call_request, action_id

    @staticmethod
    def _response_for_exception(
        agent_alias: str,
        error: Exception,
        *,
        profile: AgentProfile | None,
        generation_options: dict[str, Any] | None,
    ) -> MTPCallResponse:
        from hivememory.system.model_registry import ModelNotFoundError

        if isinstance(error, ModelNotFoundError):
            error_info = AgentModelUnavailableError(
                params={
                    "agent_alias": agent_alias,
                    "model_name": (
                        (generation_options or {}).get("model")
                        or getattr(profile, "model_name", "unknown")
                    ),
                },
                cause=error,
            ).to_error_info()
        else:
            error_info = SubAgentExecutionError(
                params={"agent_alias": agent_alias},
                cause=error,
            ).to_error_info()
        return MTPCallResponse(
            status=MTPResponseStatus.ERROR,
            agent_alias=agent_alias,
            error=error_info,
        )

    @staticmethod
    def _cancelled_response(agent_alias: str) -> MTPCallResponse:
        return MTPCallResponse(
            status=MTPResponseStatus.CANCELLED,
            agent_alias=agent_alias,
        )

    @staticmethod
    def _call_response_for_frame(
        call_request,
        result: FrameExecutionResult,
        *,
        cancelled: bool = False,
    ) -> MTPCallResponse:
        if cancelled or result.status == FrameExecutionStatus.CANCELLED:
            return MTPCallResponse(
                status=MTPResponseStatus.CANCELLED,
                agent_alias=call_request.target_alias,
            )
        if result.status == FrameExecutionStatus.COMPLETED:
            return MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias=call_request.target_alias,
            )
        if result.status == FrameExecutionStatus.BUDGET_EXHAUSTED:
            return MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=SubAgentBudgetExhaustedError(
                    params={"agent_alias": call_request.target_alias},
                ).to_error_info(),
            )
        if result.status == FrameExecutionStatus.SUSPENDED:
            return MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=SubAgentUnexpectedSuspendError(
                    params={"agent_alias": call_request.target_alias},
                    cause=result.error,
                ).to_error_info(),
            )
        return MTPCallResponse(
            status=MTPResponseStatus.ERROR,
            agent_alias=call_request.target_alias,
            error=SubAgentExecutionError(
                params={"agent_alias": call_request.target_alias},
                cause=result.error,
            ).to_error_info(),
        )

    @staticmethod
    def _frame_result_for_finalization(
        result: FrameExecutionResult | None,
        response: MTPCallResponse,
    ) -> FrameExecutionResult:
        if response.status == MTPResponseStatus.SUCCESS and result is not None:
            return result
        if response.status == MTPResponseStatus.CANCELLED:
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        if result is not None and result.status != FrameExecutionStatus.COMPLETED:
            return result
        return FrameExecutionResult(status=FrameExecutionStatus.FAILED)


__all__ = [
    "CallCompletionResult",
    "CallCoordinator",
    "CallStartResult",
    "CancelRun",
    "DispatchCallee",
    "ResumeCaller",
]
