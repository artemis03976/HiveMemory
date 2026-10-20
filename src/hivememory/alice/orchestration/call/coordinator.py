from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime import (
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
    FrameProducts,
)
from hivememory.alice.orchestration.call.context import CallContextProvider
from hivememory.alice.orchestration.call.record import CallRecord, CallRecordStatus
from hivememory.alice.orchestration.call.response import (
    preparation_error_response,
    response_for_frame_result,
)
from hivememory.alice.orchestration.frames import FrameFactory, FrameSpec
from hivememory.alice.orchestration.output import (
    AgentRunOutput,
    CallOutputFinished,
    CallOutputStarted,
)
from hivememory.alice.orchestration.verbs import mtp_verbs_for_operations
from hivememory.config.alice import KoakumaConfig
from hivememory.core.mtp import MTPCallRequest, MTPCallResponse, MTPResponseStatus
from hivememory.core.mtp.models import MTPErrorSeverity
from hivememory.i18n import resolve_language
from hivememory.prompts.compilation import render_persona
from hivememory.prompts.mtp import render_mtp_teaching

if TYPE_CHECKING:
    from hivememory.agent_runtime import AgentRuntime
    from hivememory.alice.orchestration.session import RunSession
    from hivememory.core.models import AgentProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchCallee:
    """CALL 已准备好 callee，由执行器递归运行该 frame。"""

    frame: ExecutionFrame


@dataclass(frozen=True)
class ResumeCaller:
    """CALL 已完成回填，执行器可以重入原 caller。"""


type CallStartResult = DispatchCallee | ResumeCaller
type CallCompletionResult = ResumeCaller
type _PreparationResult = ExecutionFrame | MTPCallResponse


class CallCoordinator:
    """把 CALL suspension 转换为 callee dispatch 或 caller resume。

    子线程随 CALL 返回结束：子帧以任何终态收尾（:meth:`complete_call`）、
    取消沿递归栈清理 CALL（:meth:`cancel_call`），以及派生成功之后、子帧开始
    运行之前的任何失败，都经子线程句柄同步结束子线程。主线程不由 Alice 结束。
    """

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        call_context_provider: CallContextProvider,
        *,
        frame_factory: FrameFactory,
        koakuma_config: KoakumaConfig,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_context_provider = call_context_provider
        self._frame_factory = frame_factory
        self._koakuma_config = koakuma_config

    async def begin_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        session: RunSession,
        run_output: AgentRunOutput,
    ) -> CallStartResult:
        """同步登记 CALL 后准备 callee；准备失败时直接恢复 caller。"""

        call = suspension.require_call_suspension()
        action_id = call.action_id
        call_request = call.call_request

        session.require_frame(caller_frame)

        record = session.register_call(caller_frame, action_id)
        record.begin_resolution()

        logger.info(
            "CALL suspend: target=%s, task=%r",
            call_request.target_alias,
            call_request.task[:80],
        )

        prepared = await self._prepare_callee(caller_frame, call_request)

        match prepared:
            case ExecutionFrame() as callee_frame:
                # 会话绑定失败属于编排不变量违规；子帧不会运行，先结束子线程。
                try:
                    session.register_callee_frame(callee_frame, record)
                except BaseException:
                    callee_frame.submit_operation.end()
                    raise
                await run_output.call_started(
                    CallOutputStarted(
                        agent_id=call_request.target_alias,
                        task=call_request.task,
                        iteration=caller_frame.progress.iteration,
                        action_id=action_id,
                        frame_id=callee_frame.runtime_scope.frame_id,
                    )
                )
                return DispatchCallee(callee_frame)
            case MTPCallResponse() as response:
                return await self._commit_response(
                    caller_frame,
                    suspension,
                    response,
                    record=record,
                    callee_frame=None,
                    callee_result=None,
                    run_output=run_output,
                )
            case _:
                raise AssertionError(f"Unhandled CALL preparation result: {prepared!r}")

    async def _prepare_callee(
        self,
        caller_frame: ExecutionFrame,
        call_request: MTPCallRequest,
    ) -> _PreparationResult:
        try:
            call_context = await self._call_context_provider.provide(
                caller_frame,
                call_request,
            )
        except Exception as error:
            response = preparation_error_response(call_request.target_alias, error, stage="context")
            if (
                response.error is not None
                and response.error.severity == MTPErrorSeverity.AGENT_FAULT
            ):
                logger.warning("CALL target resolution rejected: %s", response.error.code)
            else:
                logger.error("CALL context preparation failed", exc_info=True)
            return response

        callee = call_context.callee
        try:
            thread = callee.thread
            sub_profile: AgentProfile = thread.agent_profile
            # 子帧按被调用方的操作快照生成策略；仍禁止递归 CALL。
            policy = FrameExecutionPolicy.from_verbs(
                mtp_verbs_for_operations(thread.allowed_operations),
                denied_verbs={"CALL"},
            )
            # 子帧语言只解析一次：提示词与执行期的 MTP 回填使用同一值。
            language = resolve_language(profile_language=sub_profile.language).value
            messages = self._build_callee_messages(
                profile=sub_profile,
                language=language,
                task=call_request.task,
                shared_context=call_context.shared_context,
                permitted_verbs=policy.permitted_verbs,
            )
            # 子帧的观测标签取子线程自己的身份，不沿用调用方。
            scope = self._frame_factory.scope(
                labels=thread.labels,
                run_id=caller_frame.runtime_scope.run_id,
            )
            return self._frame_factory.create(
                FrameSpec(
                    runtime_scope=scope,
                    profile=sub_profile,
                    messages=messages,
                    topic_id=None,
                    execution_policy=policy,
                    submit_operation=callee,
                    language=language,
                )
            )
        except Exception as error:
            logger.error(
                "Sub-agent frame preparation failed for %r: %s",
                call_request.target_alias,
                error,
                exc_info=True,
            )
            callee.end()
            return preparation_error_response(call_request.target_alias, error)

    def _build_callee_messages(
        self,
        *,
        profile: AgentProfile,
        language: str,
        task: str,
        shared_context: str,
        permitted_verbs: frozenset[str],
    ) -> list[dict[str, str]]:
        """按被调用方语言排列教学、人设与共享上下文，最后放置任务原文。

        动词集合直接沿用子帧策略；递归 CALL 已在策略构造时排除。
        Profile 只提供人设，不参与操作权限判断。
        """
        mtp_prompt = render_mtp_teaching(
            self._koakuma_config, language=language, permitted_verbs=permitted_verbs
        )
        persona = render_persona(profile.persona, language=language)
        system_prompt = "\n\n".join(
            section for section in (mtp_prompt, persona, shared_context) if section
        )
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        return messages

    async def complete_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        callee_frame: ExecutionFrame,
        callee_result: FrameExecutionResult,
        *,
        session: RunSession,
        generation_options: dict[str, Any] | None = None,
        run_output: AgentRunOutput,
    ) -> CallCompletionResult:
        """把 callee outcome 收口为 caller 可消费的 CALL response。

        子帧已经结束运行，先同步结束子线程：此后以它的句柄提交都以线程已
        结束拒绝，子线程已提交的写入不受影响。
        """
        callee_frame.submit_operation.end()
        call = suspension.require_call_suspension()
        call_request = call.call_request
        action_id = call.action_id

        session.require_frame(caller_frame)
        session.require_frame(callee_frame)

        record = session.call_for_callee(callee_frame.runtime_scope.frame_id)

        if record.caller_frame_id != caller_frame.runtime_scope.frame_id:
            raise ValueError("Callee CALL record does not belong to the supplied caller frame.")
        if record.action_id != action_id:
            raise ValueError("Callee CALL record action does not match the suspension.")
        if callee_result.status == FrameExecutionStatus.SUSPENDED:
            raise ValueError("CallCoordinator.complete_call requires a terminal callee result.")

        effective_result, frame_products = self._finalize_callee(
            callee_frame,
            callee_result,
        )
        response = response_for_frame_result(
            call_request.target_alias,
            effective_result,
            reply="".join(callee_frame.progress.text_segments),
            artifact_aliases=frame_products.artifact_aliases,
            profile=callee_frame.agent_profile,
            generation_options=generation_options,
        )

        return await self._commit_response(
            caller_frame,
            suspension,
            response,
            record=record,
            callee_frame=callee_frame,
            callee_result=effective_result,
            run_output=run_output,
        )

    def cancel_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        session: RunSession,
    ) -> None:
        """协程取消时收尾未 apply 的 CALL，回填 cancelled response，并结束子线程。

        子线程的结束是同步调用，不在取消收尾中引入新的挂起点；子帧的在途操作
        已随取消中断，已登记的写入不受影响。
        """
        action_id = suspension.require_call_suspension().action_id
        record = session.require_call(caller_frame, action_id)
        if record.status in {CallRecordStatus.APPLIED, CallRecordStatus.CANCELLED}:
            return

        # RESOLVED 表示 callee 已完成逻辑收尾；只在仍为 RESOLVING 时清理 frame。
        if record.status == CallRecordStatus.SUSPENDED:
            record.begin_resolution()

        if record.status == CallRecordStatus.RESOLVING and record.callee_frame_id is not None:
            callee_frame = session.frames[record.callee_frame_id]
            try:
                self._agent_runtime.finalize_frame(
                    callee_frame,
                    FrameExecutionResult(status=FrameExecutionStatus.CANCELLED),
                )
            except Exception:
                logger.exception("Failed to finalize cancelled sub-agent frame")
        if record.status == CallRecordStatus.RESOLVING:
            record.mark_resolved()
        record.cancel()
        if record.callee_frame_id is not None:
            session.frames[record.callee_frame_id].submit_operation.end()

    async def _commit_response(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        response: MTPCallResponse,
        *,
        record: CallRecord,
        callee_frame: ExecutionFrame | None,
        callee_result: FrameExecutionResult | None,
        run_output: AgentRunOutput,
    ) -> CallCompletionResult:
        """在发布完成输出前，先提交调用方可消费的响应。"""
        call = suspension.require_call_suspension()
        call_request = call.call_request
        action_id = call.action_id

        record.mark_resolved()

        self._agent_runtime.apply_call_response(caller_frame, suspension, response)
        record.mark_applied()

        await self._emit_call_end(
            caller_frame,
            action_id=action_id,
            agent_alias=call_request.target_alias,
            response=response,
            callee_frame=callee_frame,
            callee_result=callee_result,
            run_output=run_output,
        )
        return ResumeCaller()

    def _finalize_callee(
        self,
        callee_frame: ExecutionFrame,
        callee_result: FrameExecutionResult,
    ) -> tuple[FrameExecutionResult, FrameProducts]:
        try:
            frame_products = self._agent_runtime.finalize_frame(
                callee_frame,
                callee_result,
            )
            return callee_result, frame_products
        except Exception as error:
            logger.error("Failed to finalize sub-agent frame: %s", error, exc_info=True)
            if callee_result.status == FrameExecutionStatus.CANCELLED:
                return callee_result, FrameProducts()

            failed_result = FrameExecutionResult(
                status=FrameExecutionStatus.FAILED,
                error=error,
            )
            if callee_result.status == FrameExecutionStatus.COMPLETED:
                try:
                    self._agent_runtime.finalize_frame(callee_frame, failed_result)
                except Exception:
                    logger.exception("Failed to clean up sub-agent frame after harvest error")
            return failed_result, FrameProducts()

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


__all__ = [
    "CallCompletionResult",
    "CallCoordinator",
    "CallStartResult",
    "DispatchCallee",
    "ResumeCaller",
]
