from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    MTPExecutionContext,
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
    from hivememory.agent_runtime.resolver import RuntimeAliasResolver
    from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler
    from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
    from hivememory.alice.runtime.agent.runtime import AgentRuntime
    from hivememory.core.models import AgentProfile, Identity

logger = logging.getLogger(__name__)

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]


class CallCoordinator:
    """Resolve one CALL and return the existing caller-facing response model."""

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        frame_scheduler: FrameScheduler,
        agent_profile_resolver: AgentProfileResolver,
        alias_resolver: RuntimeAliasResolver,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._frame_scheduler = frame_scheduler
        self._agent_profile_resolver = agent_profile_resolver
        self._alias_resolver = alias_resolver

    async def resolve_call(
        self,
        caller_frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        *,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        emit: EventEmitter | None = None,
    ) -> MTPCallResponse:
        call_request = suspension.call_request
        if suspension.status != FrameExecutionStatus.SUSPENDED or call_request is None:
            raise ValueError("CALL coordinator requires a suspended frame result.")

        action_id = suspension.suspend_action_id
        logger.info(
            "CALL suspend: target=%s, task=%r",
            call_request.target_alias,
            call_request.task[:80],
        )

        self._frame_scheduler.suspend_frame(caller_frame)
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
            sub_frame = await self._frame_scheduler.fork_sub_frame(
                parent_frame=caller_frame,
                agent_profile=sub_profile,
                task=call_request.task,
                shared_context=shared_context,
            )
            if emit is not None:
                await emit(
                    {
                        "event": "sub_agent_start",
                        "data": {
                            "agent_id": call_request.target_alias,
                            "task": call_request.task,
                            "iteration": caller_frame.progress.iteration,
                            "action_id": action_id,
                            "scope": "sub",
                            "depth": caller_frame.runtime_scope.depth + 1,
                            "frame_id": sub_frame.runtime_scope.frame_id,
                        },
                    }
                )

            if emit is None:
                sub_result = await self._agent_runtime.run_frame(
                    frame=sub_frame,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                )
            else:

                async def _sub_emit(event: dict[str, Any]) -> None:
                    await emit(event)

                sub_result = await self._agent_runtime.run_frame_emitting(
                    frame=sub_frame,
                    generation_options=generation_options,
                    stream_emitter=_sub_emit,
                    event_metadata=self._event_metadata_for_frame(sub_frame),
                    cancel_event=cancel_event,
                )

            call_response = self._call_response_for_frame(
                call_request,
                sub_result,
                cancelled=cancel_event is not None and cancel_event.is_set(),
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
        finally:
            self._frame_scheduler.resume_frame()

        if call_response is None:
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=SubAgentExecutionError(
                    params={"agent_alias": call_request.target_alias},
                ).to_error_info(),
            )

        if (
            sub_frame is not None
            and sub_result is not None
            and sub_result.status == FrameExecutionStatus.COMPLETED
            and call_response.status == MTPResponseStatus.SUCCESS
        ):
            try:
                sub_text = "".join(sub_frame.progress.text_segments)
                self._harvest_sub_frame_aliases(sub_frame)
                call_response = call_response.model_copy(
                    update={
                        "reply": sub_text,
                        "artifact_aliases": list(sub_frame.harvested_aliases),
                    }
                )
            except Exception as error:
                logger.error("Failed to harvest sub-agent result: %s", error, exc_info=True)
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
                self._cancel_frame(sub_frame)
        elif sub_frame is not None:
            self._cancel_frame(sub_frame)

        if emit is not None:
            end_data: dict[str, Any] = {
                "status": call_response.status.value,
                "final_text": (
                    call_response.reply if call_response.status == MTPResponseStatus.SUCCESS else ""
                ),
                "iteration": caller_frame.progress.iteration,
                "action_id": action_id,
                "scope": "sub",
                "depth": caller_frame.runtime_scope.depth + 1,
                "frame_id": sub_frame.runtime_scope.frame_id if sub_frame is not None else None,
                "agent_id": call_request.target_alias,
            }
            if sub_result is not None:
                end_data["terminal_status"] = sub_result.status.value
            if call_response.error is not None:
                end_data["error_code"] = call_response.error.code
            await emit({"event": "sub_agent_end", "data": end_data})

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

    def _cancel_frame(self, frame: ExecutionFrame) -> None:
        cancel_by_frame = getattr(self._agent_runtime, "cancel_tasks_by_frame", None)
        if callable(cancel_by_frame):
            cancel_by_frame(frame.runtime_scope.frame_id)

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

    def _harvest_sub_frame_aliases(self, frame: ExecutionFrame) -> None:
        from hivememory.core.mtp.models import MTPVerb

        harvested = set(frame.harvested_aliases)
        aliases_by_frame = getattr(self._agent_runtime, "aliases_by_frame", None)
        if callable(aliases_by_frame):
            for alias in aliases_by_frame(frame.runtime_scope.frame_id):
                if alias and alias not in harvested:
                    frame.harvested_aliases.append(alias)
                    harvested.add(alias)

        for event in frame.progress.turn_events:
            if event.kind == "tool_call" and event.tool_kind == MTPVerb.UPDATE.value:
                alias = event.target
                if alias and alias not in harvested:
                    frame.harvested_aliases.append(alias)
                    harvested.add(alias)

    @staticmethod
    def _event_metadata_for_frame(frame: ExecutionFrame) -> dict[str, Any]:
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "agent_run_id": frame.runtime_scope.run_id,
            "action_id": None,
            "scope": "sub",
            "depth": frame.runtime_scope.depth,
            "agent_id": agent_id,
            "frame_id": frame.runtime_scope.frame_id,
        }


__all__ = ["CallCoordinator"]
