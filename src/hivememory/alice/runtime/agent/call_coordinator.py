from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.events import FrameEventSink, NullFrameEventSink
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    MTPExecutionContext,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import FrameProducts
from hivememory.alice.runtime.agent.event_sink import ScopedFrameEventSink
from hivememory.alice.runtime.agent.frame_factory import FrameFactory, FrameSpec
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
    from hivememory.agent_runtime.resolver import RuntimeAliasResolver
    from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
    from hivememory.alice.runtime.agent.run_session import RunSession
    from hivememory.alice.runtime.agent.runtime import AgentRuntime
    from hivememory.core.models import AgentProfile, Identity
    from hivememory.prompts.assembler import AgentPromptAssembler

logger = logging.getLogger(__name__)


class CallCoordinator:
    """Resolve one CALL and return the existing caller-facing response model."""

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

    async def resolve_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        session: RunSession,
        generation_options: dict[str, Any] | None = None,
        event_sink: FrameEventSink | None = None,
    ) -> MTPCallResponse:
        call_request = suspension.call_request
        if suspension.status != FrameExecutionStatus.SUSPENDED or call_request is None:
            raise ValueError("CALL coordinator requires a suspended frame result.")
        session.require_frame(caller_frame)

        action_id = suspension.suspend_action_id
        logger.info(
            "CALL suspend: target=%s, task=%r",
            call_request.target_alias,
            call_request.task[:80],
        )

        sub_frame: ExecutionFrame | None = None
        sub_profile: AgentProfile | None = None
        sub_result: FrameExecutionResult | None = None
        call_response: MTPCallResponse | None = None
        try:
            sub_profile = await self._agent_profile_resolver.resolve(
                call_request.target_alias,
                identity=caller_frame.identity,
            )
            shared_context = await self._fetch_context_refs_content(
                aliases=call_request.context_refs,
                identity=caller_frame.identity,
                language=getattr(caller_frame.agent_profile, "language", None),
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
            record = session.require_call(caller_frame, action_id)
            session.register_callee_frame(sub_frame, record)
            if event_sink is not None:
                await event_sink.emit(
                    {
                        "event": "sub_agent_start",
                        "data": {
                            "agent_id": call_request.target_alias,
                            "task": call_request.task,
                            "iteration": caller_frame.progress.iteration,
                            "action_id": action_id,
                            "scope": "sub",
                            "depth": 1,
                            "frame_id": sub_frame.runtime_scope.frame_id,
                        },
                    }
                )

            sub_sink: FrameEventSink = NullFrameEventSink()
            if event_sink is not None:
                sub_sink = ScopedFrameEventSink(
                    event_sink,
                    metadata=self._event_metadata_for_frame(sub_frame, action_id),
                )
            sub_result = await self._agent_runtime.run_frame(
                frame=sub_frame,
                generation_options=generation_options,
                event_sink=sub_sink,
                cancel_event=session.cancel_event,
            )

            call_response = self._call_response_for_frame(
                call_request,
                sub_result,
                cancelled=session.cancel_event.is_set(),
            )
        except MTPError as error:
            logger.warning("CALL rejected for %r: %s", call_request.target_alias, error.code)
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=error.to_error_info(),
            )
        except Exception as error:
            logger.error("Sub-agent execution failed: %s", error, exc_info=True)
            from hivememory.system.model_registry import ModelNotFoundError

            if isinstance(error, ModelNotFoundError):
                error_info = AgentModelUnavailableError(
                    params={
                        "agent_alias": call_request.target_alias,
                        "model_name": (
                            (generation_options or {}).get("model")
                            or getattr(sub_profile, "model_name", "unknown")
                        ),
                    },
                    cause=error,
                ).to_error_info()
            else:
                error_info = SubAgentExecutionError(
                    params={"agent_alias": call_request.target_alias},
                    cause=error,
                ).to_error_info()
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=error_info,
            )
        if call_response is None:
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=SubAgentExecutionError(
                    params={"agent_alias": call_request.target_alias},
                ).to_error_info(),
            )

        frame_products = FrameProducts()
        if sub_frame is not None:
            finalization_result = self._frame_result_for_finalization(
                sub_result,
                call_response,
            )
            try:
                frame_products = self._agent_runtime.finalize_frame(
                    sub_frame,
                    finalization_result,
                )
            except Exception as error:
                logger.error("Failed to finalize sub-agent frame: %s", error, exc_info=True)
                call_response = MTPCallResponse(
                    status=MTPResponseStatus.ERROR,
                    agent_alias=call_request.target_alias,
                    error=SubAgentExecutionError(
                        params={"agent_alias": call_request.target_alias},
                        cause=error,
                    ).to_error_info(),
                )
                sub_result = FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED,
                    error=error,
                )
                if finalization_result.status == FrameExecutionStatus.COMPLETED:
                    try:
                        self._agent_runtime.finalize_frame(sub_frame, sub_result)
                    except Exception:
                        logger.exception("Failed to clean up sub-agent frame after harvest error")

        if (
            sub_frame is not None
            and sub_result is not None
            and sub_result.status == FrameExecutionStatus.COMPLETED
            and call_response.status == MTPResponseStatus.SUCCESS
        ):
            call_response = call_response.model_copy(
                update={
                    "reply": "".join(sub_frame.progress.text_segments),
                    "artifact_aliases": list(frame_products.artifact_aliases),
                }
            )

        if event_sink is not None:
            end_data: dict[str, Any] = {
                "status": call_response.status.value,
                "final_text": (
                    call_response.reply if call_response.status == MTPResponseStatus.SUCCESS else ""
                ),
                "iteration": caller_frame.progress.iteration,
                "action_id": action_id,
                "scope": "sub",
                "depth": 1,
                "frame_id": sub_frame.runtime_scope.frame_id if sub_frame is not None else None,
                "agent_id": call_request.target_alias,
            }
            if sub_result is not None:
                end_data["terminal_status"] = sub_result.status.value
            if call_response.error is not None:
                end_data["error_code"] = call_response.error.code
            await event_sink.emit({"event": "sub_agent_end", "data": end_data})

        return call_response

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

    @staticmethod
    def _event_metadata_for_frame(
        frame: ExecutionFrame,
        action_id: str | None,
    ) -> dict[str, Any]:
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "agent_run_id": frame.runtime_scope.run_id,
            "action_id": action_id,
            "scope": "sub",
            "depth": 1,
            "agent_id": agent_id,
            "frame_id": frame.runtime_scope.frame_id,
        }


__all__ = ["CallCoordinator"]
