"""
AgentOrchestrator - 多智能体编排驱动器

职责（编排层）：
    - 造主帧、驱动引擎循环
    - 收到 SUSPENDED 时：fork 子帧 → 驱动引擎跑子帧 → resume → harvest → 组 CALL response → 重入
    - COMPLETED 时从 frame.progress 聚合 AgentRunResult
    - 流式模式下负责 sub_agent_start/end 事件与子帧事件透传

不变量：本模块不得反向 import alice/ 以外的子系统。
见 docs/archive/plans/implementation/agent-loop-decoupling.md §3.2 / §4 Phase 1+2。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    MTPExecutionContext,
)
from hivememory.alice.runtime.agent.call_coordinator import CallCoordinator
from hivememory.alice.runtime.agent.run_driver import RunDriver
from hivememory.core.models import TurnEvent
from hivememory.core.mtp import MTPCallResponse, MTPFormatter, MTPResponseStatus
from hivememory.core.mtp.exceptions import (
    AgentModelUnavailableError,
    MTPError,
    SubAgentBudgetExhaustedError,
    SubAgentExecutionError,
    SubAgentUnexpectedSuspendError,
)
from hivememory.core.protocol.models import AgentRunResult, AgentRunStatus
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


class AgentOrchestrator:
    """
    多智能体编排驱动器。

    持有单 Agent 运行时门面（AgentRuntime）和编排组件（FrameScheduler /
    AgentProfileResolver / RuntimeAliasResolver），负责：
      1. 造主帧（create_main_frame）
      2. 通过门面 run_frame(main_frame) 跑单 Agent
      3. SUSPENDED → 重入序列（append CALL 文本 → fork/跑子帧/resume/harvest/组 CALL response → append 回填）
      4. COMPLETED → 从 frame.progress 聚合 AgentRunResult

    编排只调门面 API 跑 frame，不直接接触 loop_executor 或迭代上限等引擎细节。
    """

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
        self._mtp_formatter = MTPFormatter()
        self._call_coordinator = CallCoordinator(
            agent_runtime,
            frame_scheduler,
            agent_profile_resolver,
            alias_resolver,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def run_agent(
        self,
        messages: list[dict[str, str]],
        identity: Identity,
        topic_id: str,
        generation_options: dict[str, Any] | None = None,
        agent_profile: AgentProfile | None = None,
        cancel_event=None,
    ) -> AgentRunResult:
        main_frame = self._frame_scheduler.create_main_frame(
            agent_profile=agent_profile,
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )
        self._record_initial_user_event(main_frame, messages)
        driver = RunDriver(self._agent_runtime, self._call_coordinator)

        engine_result = await driver.run(
            main_frame,
            generation_options=generation_options,
            cancel_event=cancel_event,
        )
        return self._assemble_agent_run_result(
            main_frame,
            engine_result=engine_result,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        messages: list[dict[str, str]],
        identity: Identity,
        topic_id: str,
        generation_options: dict[str, Any] | None = None,
        agent_profile: AgentProfile | None = None,
        cancel_event=None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        main_frame = self._frame_scheduler.create_main_frame(
            agent_profile=agent_profile,
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )
        self._record_initial_user_event(main_frame, messages)

        driver = RunDriver(self._agent_runtime, self._call_coordinator)
        event_metadata = self._event_metadata_for_frame(main_frame)

        async for event in driver.run_stream(
            main_frame,
            generation_options=generation_options,
            cancel_event=cancel_event,
            event_metadata=event_metadata,
        ):
            yield event

        terminal_result = driver.terminal_result
        if terminal_result is None:
            raise RuntimeError("Run driver ended without a terminal result.")
        yield {
            "event": "done",
            "data": {
                **self._assemble_agent_run_result(
                    main_frame,
                    engine_result=terminal_result,
                    cancel_event=cancel_event,
                ).model_dump(),
                **event_metadata,
                "stream_sequence": driver.next_stream_sequence,
            },
        }

    def _record_initial_user_event(
        self,
        frame: ExecutionFrame,
        messages: list[dict[str, str]],
    ) -> None:
        """
        记录初始用户消息（如果存在）作为首个 TurnEvent。
        """
        content = self._current_user_message(messages)
        if not content:
            return
        if any(event.kind == "user_message" for event in frame.progress.turn_events):
            return

        frame.progress.turn_events = [
            event.model_copy(update={"sequence": event.sequence + 1})
            for event in frame.progress.turn_events
        ]

        frame.progress.turn_events.insert(
            0,
            TurnEvent(
                kind="user_message",
                sequence=0,
                role="user",
                content=content,
            ),
        )
        frame.progress.sequence = max(
            frame.progress.sequence + 1,
            max((event.sequence for event in frame.progress.turn_events), default=-1) + 1,
        )

    def _current_user_message(self, messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    # ------------------------------------------------------------------
    # 内部：SUSPENDED 重入序列
    # ------------------------------------------------------------------

    async def _handle_suspend(
        self,
        main_frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        generation_options: dict[str, Any] | None,
        emit: Callable | None = None,
        cancel_event=None,
    ) -> None:
        """Compatibility entrypoint; new runs use RunDriver + CallCoordinator."""
        apply_response = getattr(self._agent_runtime, "apply_call_response", None)
        if callable(apply_response):
            response = await self._call_coordinator.resolve_call(
                main_frame,
                engine_result,
                generation_options=generation_options,
                cancel_event=cancel_event,
                emit=emit,
            )
            if cancel_event is None or not cancel_event.is_set():
                apply_response(main_frame, engine_result, response)
            return
        await self._legacy_handle_suspend(
            main_frame=main_frame,
            engine_result=engine_result,
            generation_options=generation_options,
            emit=emit,
            cancel_event=cancel_event,
        )

    async def _legacy_handle_suspend(
        self,
        main_frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        generation_options: dict[str, Any] | None,
        emit: Callable | None = None,
        cancel_event=None,
    ) -> None:
        cr = engine_result.call_request
        action_id = engine_result.suspend_action_id
        suspend_text = engine_result.suspend_assistant_text or ""

        logger.info(f"CALL suspend: target={cr.target_alias}, task='{cr.task[:80]}'")

        main_frame.working_history.append({"role": "assistant", "content": suspend_text})

        self._frame_scheduler.suspend_frame(main_frame)
        sub_result_text = ""
        sub_frame = None
        sub_profile = None
        sub_execution_result: FrameExecutionResult | None = None
        call_response: MTPCallResponse | None = None
        try:
            sub_profile = await self._agent_profile_resolver.resolve(
                cr.target_alias,
                identity=main_frame.identity,
            )
            shared_context = await self._fetch_context_refs_content(
                aliases=cr.context_refs,
                identity=main_frame.identity,
                language=getattr(main_frame.agent_profile, "language", None),
            )
            sub_frame = await self._frame_scheduler.fork_sub_frame(
                parent_frame=main_frame,
                agent_profile=sub_profile,
                task=cr.task,
                shared_context=shared_context,
            )
            if emit is not None:
                await emit(
                    {
                        "event": "sub_agent_start",
                        "data": {
                            "agent_id": cr.target_alias,
                            "task": cr.task,
                            "iteration": main_frame.progress.iteration,
                            "action_id": action_id,
                            "scope": "sub",
                            "depth": main_frame.runtime_scope.depth + 1,
                            "frame_id": sub_frame.runtime_scope.frame_id,
                        },
                    }
                )

            if emit is None:
                sub_execution_result = await self._agent_runtime.run_frame(
                    frame=sub_frame,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                )
            else:

                async def _sub_emit(sub_event: dict[str, Any]) -> None:
                    await emit(sub_event)

                sub_execution_result = await self._agent_runtime.run_frame_emitting(
                    frame=sub_frame,
                    generation_options=generation_options,
                    stream_emitter=_sub_emit,
                    event_metadata=self._event_metadata_for_frame(sub_frame),
                    cancel_event=cancel_event,
                )

            call_response = self._call_response_for_sub_frame(
                call_request=cr,
                execution_result=sub_execution_result,
            )

        except MTPError as e:
            logger.warning("CALL rejected for %r: %s", cr.target_alias, e.code)
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=cr.target_alias,
                error=e.to_error_info(),
            )
        except Exception as e:
            logger.error(f"Sub-agent execution failed: {e}", exc_info=True)
            from hivememory.system.model_registry import ModelNotFoundError

            if isinstance(e, ModelNotFoundError):
                error = AgentModelUnavailableError(
                    params={
                        "agent_alias": cr.target_alias,
                        "model_name": (
                            (generation_options or {}).get("model")
                            or getattr(sub_profile, "model_name", "unknown")
                        ),
                    },
                    cause=e,
                ).to_error_info()
            else:
                error = SubAgentExecutionError(
                    params={"agent_alias": cr.target_alias},
                    cause=e,
                ).to_error_info()
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=cr.target_alias,
                error=error,
            )
        finally:
            self._frame_scheduler.resume_frame()

        if call_response is None:
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=cr.target_alias,
                error=SubAgentExecutionError(
                    params={"agent_alias": cr.target_alias},
                ).to_error_info(),
            )

        if (
            sub_frame is not None
            and sub_execution_result is not None
            and sub_execution_result.status == FrameExecutionStatus.COMPLETED
            and call_response.status == MTPResponseStatus.SUCCESS
        ):
            try:
                sub_result_text = "".join(sub_frame.progress.text_segments)
                self._harvest_sub_frame_aliases(sub_frame)
                for alias in sub_frame.harvested_aliases:
                    if alias not in main_frame.harvested_aliases:
                        main_frame.harvested_aliases.append(alias)
                call_response = call_response.model_copy(
                    update={
                        "reply": sub_result_text,
                        "artifact_aliases": sub_frame.harvested_aliases,
                    }
                )
            except Exception as e:
                logger.error("Failed to harvest sub-agent result: %s", e, exc_info=True)
                call_response = MTPCallResponse(
                    status=MTPResponseStatus.ERROR,
                    agent_alias=cr.target_alias,
                    error=SubAgentExecutionError(
                        params={"agent_alias": cr.target_alias},
                        cause=e,
                    ).to_error_info(),
                )
                sub_execution_result = FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED,
                    error=e,
                )
                cancel_by_frame = getattr(
                    self._agent_runtime,
                    "cancel_tasks_by_frame",
                    None,
                )
                if callable(cancel_by_frame):
                    cancel_by_frame(sub_frame.runtime_scope.frame_id)
        elif sub_frame is not None:
            cancel_by_frame = getattr(self._agent_runtime, "cancel_tasks_by_frame", None)
            if callable(cancel_by_frame):
                cancel_by_frame(sub_frame.runtime_scope.frame_id)

        if emit is not None:
            end_data = {
                "status": call_response.status.value,
                "final_text": (
                    sub_result_text if call_response.status == MTPResponseStatus.SUCCESS else ""
                ),
                "iteration": main_frame.progress.iteration,
                "action_id": action_id,
                "scope": "sub",
                "depth": main_frame.runtime_scope.depth + 1,
                "frame_id": sub_frame.runtime_scope.frame_id if sub_frame is not None else None,
                "agent_id": cr.target_alias,
            }
            if sub_execution_result is not None:
                end_data["terminal_status"] = sub_execution_result.status.value
            if call_response.error is not None:
                end_data["error_code"] = call_response.error.code
            await emit({"event": "sub_agent_end", "data": end_data})

        formatted_call_response = self._mtp_formatter.format_call_response(
            call_response,
            getattr(main_frame.agent_profile, "language", None),
        )

        # iv. append CALL response + tool_result TurnEvent
        main_frame.working_history.append(
            {
                "role": "user",
                "content": formatted_call_response,
            }
        )

        # 找到对应的 tool_call 事件并同步最终 CALL 状态。
        matched_action = False
        for index, ev in enumerate(main_frame.progress.turn_events):
            if ev.kind == "tool_call" and ev.action_id == action_id:
                main_frame.progress.turn_events[index] = ev.model_copy(
                    update={"status": call_response.status.value}
                )
                matched_action = True
                break
        if action_id is not None and not matched_action:
            raise RuntimeError(
                f"CALL result has no matching tool_call event: action_id={action_id}"
            )

        main_frame.progress.turn_events.append(
            TurnEvent(
                kind="tool_result",
                sequence=main_frame.progress.sequence,
                role="user",
                content=formatted_call_response,
                action_id=action_id,
                tool_kind="CALL",
                tool_name=cr.target_alias,
                status=call_response.status.value,
                render_as="system_call_response",
            )
        )
        main_frame.progress.sequence += 1

    def _call_response_for_sub_frame(
        self,
        *,
        call_request,
        execution_result: FrameExecutionResult,
    ) -> MTPCallResponse:
        """Map a child frame terminal signal to the parent-facing CALL result."""
        if execution_result.status == FrameExecutionStatus.COMPLETED:
            return MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias=call_request.target_alias,
            )
        if execution_result.status == FrameExecutionStatus.CANCELLED:
            return MTPCallResponse(
                status=MTPResponseStatus.CANCELLED,
                agent_alias=call_request.target_alias,
            )
        if execution_result.status == FrameExecutionStatus.BUDGET_EXHAUSTED:
            return MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=SubAgentBudgetExhaustedError(
                    params={"agent_alias": call_request.target_alias},
                ).to_error_info(),
            )
        if execution_result.status == FrameExecutionStatus.SUSPENDED:
            return MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=call_request.target_alias,
                error=SubAgentUnexpectedSuspendError(
                    params={"agent_alias": call_request.target_alias},
                    cause=execution_result.error,
                ).to_error_info(),
            )
        return MTPCallResponse(
            status=MTPResponseStatus.ERROR,
            agent_alias=call_request.target_alias,
            error=SubAgentExecutionError(
                params={"agent_alias": call_request.target_alias},
                cause=execution_result.error,
            ).to_error_info(),
        )

    # ------------------------------------------------------------------
    # 内部：辅助方法（从 loop_executor 迁入）
    # ------------------------------------------------------------------

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
            except Exception as e:
                logger.warning(f"Failed to resolve context_ref {alias}: {e}")
                continue
            if resolved.kind in {"pending", "redirect", "atom"} and (
                resolved.pending is not None or resolved.atom is not None
            ):
                sources.append(resolved)
            else:
                logger.warning(f"Context ref alias not found: {alias}")
        if not sources:
            logger.warning(f"No rendered context returned for context_refs: {aliases}")
            return ""
        return compiler.compile(
            sources,
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
            MemoryCompileOptions(language=language),
        ).text

    def _harvest_sub_frame_aliases(self, sub_frame: ExecutionFrame) -> None:
        """从子帧 PendingAtomRuntime 重建 harvested_aliases。

        通过 frame_id 过滤子帧的 PendingAtom，收集 pending_alias 用于 CALL response artifacts。
        UPDATE fallback：从 tool_call TurnEvent.target 补充尚未注册为 pending 的 alias。
        """
        from hivememory.core.mtp.models import MTPVerb

        harvested = set(sub_frame.harvested_aliases)

        # WRITE/UPDATE aliases from PendingAtomRuntime（主要路径）
        frame_id = sub_frame.runtime_scope.frame_id
        for alias in self._agent_runtime.aliases_by_frame(frame_id):
            if alias and alias not in harvested:
                sub_frame.harvested_aliases.append(alias)
                harvested.add(alias)

        # UPDATE fallback: target alias when no pending_alias was generated
        for ev in sub_frame.progress.turn_events:
            if ev.kind == "tool_call" and ev.tool_kind == MTPVerb.UPDATE.value:
                alias = ev.target
                if alias and alias not in harvested:
                    sub_frame.harvested_aliases.append(alias)
                    harvested.add(alias)

    def _assemble_agent_run_result(
        self,
        frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        cancel_event=None,
    ) -> AgentRunResult:
        p = frame.progress
        cancelled = engine_result.status == FrameExecutionStatus.CANCELLED or (
            cancel_event is not None and cancel_event.is_set()
        )
        completed = engine_result.status == FrameExecutionStatus.COMPLETED and not cancelled
        if not completed:
            self._agent_runtime.cancel_tasks_by_run(frame.runtime_scope.run_id)
            tasks = []
        else:
            tasks = self._agent_runtime.collect_tasks_by_run(frame.runtime_scope.run_id)
        if cancelled:
            run_status = AgentRunStatus.CANCELLED
        elif completed:
            run_status = AgentRunStatus.COMPLETED
        else:
            run_status = AgentRunStatus.FAILED
        return AgentRunResult(
            status=run_status,
            final_text="".join(p.text_segments),
            mtp_iterations=max(0, p.iteration - 1),
            total_iterations=p.iteration,
            turn_events=p.turn_events,
            materialize_tasks=tasks,
            # frame.progress.model_used 由 AgentRuntime._resolve_model_for_frame 写入
            # 空字符串表示注册表未启用（兼容无注册表的场景）
            model_used=p.model_used,
        )

    @staticmethod
    def _event_metadata_for_frame(frame: ExecutionFrame) -> dict[str, Any]:
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "agent_run_id": frame.runtime_scope.run_id,
            "action_id": None,
            "scope": "sub" if frame.is_sub_frame() else "main",
            "depth": frame.runtime_scope.depth,
            "agent_id": agent_id,
            "frame_id": frame.runtime_scope.frame_id,
        }


__all__ = ["AgentOrchestrator"]
