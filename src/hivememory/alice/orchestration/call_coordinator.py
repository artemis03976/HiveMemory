from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    MTPExecutionContext,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import FrameProducts
from hivememory.alice.orchestration.call_record import CallRecordStatus
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_output import (
    AgentRunOutput,
    CallOutputFinished,
    CallOutputStarted,
    NullAgentRunOutput,
)
from hivememory.core.mtp import MTPCallResponse, MTPResponseStatus
from hivememory.core.mtp.exceptions import (
    AgentModelUnavailableError,
    MTPError,
    SubAgentBudgetExhaustedError,
    SubAgentExecutionError,
    SubAgentUnexpectedSuspendError,
)
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.aliases import RuntimeAliasResolver
    from hivememory.agent_runtime.runtime import AgentRuntime
    from hivememory.alice.orchestration.profile_resolver import AgentProfileResolver
    from hivememory.alice.orchestration.run_session import RunSession
    from hivememory.core.models import AgentProfile, Identity
    from hivememory.prompts.assembler import AgentPromptAssembler

logger = logging.getLogger(__name__)


class CallNextAction(str, Enum):
    """一次 CALL 边界处理完成后交给调度器的下一步动作。"""

    DISPATCH_CALLEE = "dispatch_callee"
    RESUME_CALLER = "resume_caller"
    CANCEL_RUN = "cancel_run"


@dataclass(frozen=True)
class CallTransition:
    """CALL 协调阶段返回的窄调度结果。"""

    action: CallNextAction
    next_frame: ExecutionFrame | None = None

    def __post_init__(self) -> None:
        requires_frame = self.action in {
            CallNextAction.DISPATCH_CALLEE,
            CallNextAction.RESUME_CALLER,
        }
        if requires_frame and self.next_frame is None:
            raise ValueError(f"CALL transition {self.action.value!r} requires a next frame.")
        if self.action == CallNextAction.CANCEL_RUN and self.next_frame is not None:
            raise ValueError("A cancelled run cannot schedule another frame.")


class CallCoordinator:
    """把 CALL suspension 转换为 callee dispatch 或 caller resume。"""

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        agent_profile_resolver: AgentProfileResolver,
        alias_resolver: RuntimeAliasResolver,
        *,
        frame_factory: FrameFactory,
        prompt_assembler: AgentPromptAssembler,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._agent_profile_resolver = agent_profile_resolver
        self._alias_resolver = alias_resolver
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
    ) -> CallTransition:
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
            sub_profile = await self._agent_profile_resolver.resolve(
                call_request.target_alias,
                identity=caller_frame.identity,
            )
            if session.cancel_event.is_set():
                return await self._complete_preparation(
                    caller_frame,
                    suspension,
                    self._cancelled_response(call_request.target_alias),
                    session=session,
                    run_output=output,
                )
            shared_context = await self._fetch_context_refs_content(
                aliases=call_request.context_refs,
                identity=caller_frame.identity,
                language=getattr(caller_frame.agent_profile, "language", None),
            )
            if session.cancel_event.is_set():
                return await self._complete_preparation(
                    caller_frame,
                    suspension,
                    self._cancelled_response(call_request.target_alias),
                    session=session,
                    run_output=output,
                )
            policy = FrameExecutionPolicy.from_profile(
                sub_profile,
                max_iterations=getattr(self._agent_runtime, "max_iterations", None),
                denied_verbs={"CALL"},
            )
            messages = self._prompt_assembler.build_sub_agent_messages(
                profile=sub_profile,
                task=call_request.task,
                shared_context=shared_context,
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
        return CallTransition(CallNextAction.DISPATCH_CALLEE, sub_frame)

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
    ) -> CallTransition:
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
    ) -> CallTransition:
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
    ) -> CallTransition:
        if session.cancel_event.is_set():
            record.cancel()
            return CallTransition(CallNextAction.CANCEL_RUN)
        self._agent_runtime.apply_call_response(caller_frame, suspension, response)
        record.mark_applied()
        return CallTransition(CallNextAction.RESUME_CALLER, caller_frame)

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

    async def _fetch_context_refs_content(
        self,
        aliases: list[str],
        identity: Identity,
        language: str | None = None,
    ) -> str:
        if not aliases:
            return ""
        compiler = MemoryCompiler()
        sources = []
        context = MTPExecutionContext(identity=identity)
        for alias in aliases:
            try:
                resolved = await self._alias_resolver.resolve(alias, context=context)
            except Exception as error:
                logger.warning("Failed to resolve context_ref %s: %s", alias, error)
                continue
            if resolved.kind in {"pending", "redirect", "atom"} and (
                resolved.pending is not None or resolved.atom is not None
            ):
                sources.append(resolved)
            else:
                logger.warning("Context ref alias not found: %s", alias)
        if not sources:
            logger.warning("No rendered context returned for context_refs: %s", aliases)
            return ""
        return compiler.compile(
            sources,
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
            MemoryCompileOptions(language=language),
        ).text


__all__ = ["CallCoordinator", "CallNextAction", "CallTransition"]
