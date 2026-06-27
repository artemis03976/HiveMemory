"""
AgentOrchestrator - 多智能体编排驱动器

职责（编排层）：
    - 造主帧、驱动引擎循环
    - 收到 SUSPENDED 时：fork 子帧 → 驱动引擎跑子帧 → resume → harvest → 组 CALL response → 重入
    - COMPLETED 时从 frame.progress 聚合 AgentRunResult
    - 流式模式下负责 sub_agent_start/end 事件与子帧事件透传

不变量：本模块不得反向 import alice/ 以外的子系统。
见 docs/mod/AgentLoopDecouplingDesign.md §3.2 / §4 Phase 1+2。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    MTPExecutionContext,
)
from hivememory.core.models import TurnEvent
from hivememory.core.mtp import MTPCallResponse, MTPFormatter, MTPResponseStatus
from hivememory.core.mtp.exceptions import SubAgentExecutionError
from hivememory.core.protocol.models import AgentRunResult, AgentRunStatus
from hivememory.engines.memory_compiler import (
    CompiledMemory,
    MemoryCompiler,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeTarget,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler
    from hivememory.alice.runtime.agent.runtime import AgentRuntime
    from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
    from hivememory.agent_runtime.resolver import RuntimeAliasResolver
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
        agent_runtime: "AgentRuntime",
        frame_scheduler: "FrameScheduler",
        agent_profile_resolver: "AgentProfileResolver",
        alias_resolver: "RuntimeAliasResolver",
    ) -> None:
        self._agent_runtime = agent_runtime
        self._frame_scheduler = frame_scheduler
        self._agent_profile_resolver = agent_profile_resolver
        self._alias_resolver = alias_resolver
        self._mtp_formatter = MTPFormatter()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def run_agent(
        self,
        messages: List[Dict[str, str]],
        identity: "Identity",
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile: Optional["AgentProfile"] = None,
        cancel_event=None,
    ) -> AgentRunResult:
        main_frame = self._frame_scheduler.create_main_frame(
            agent_profile=agent_profile,
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )
        while True:
            engine_result = await self._agent_runtime.run_frame(
                frame=main_frame,
                generation_options=generation_options,
                cancel_event=cancel_event,
            )
            if engine_result.status == FrameExecutionStatus.SUSPENDED:
                await self._handle_suspend(
                    main_frame=main_frame,
                    engine_result=engine_result,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                )
                continue
            break
        return self._assemble_agent_run_result(main_frame, cancel_event=cancel_event)

    async def run_agent_stream(
        self,
        messages: List[Dict[str, str]],
        identity: "Identity",
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile: Optional["AgentProfile"] = None,
        cancel_event=None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        main_frame = self._frame_scheduler.create_main_frame(
            agent_profile=agent_profile,
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )

        queue: asyncio.Queue = asyncio.Queue()

        async def _emit(event: Dict[str, Any]) -> None:
            await queue.put(event)

        async def on_suspend(engine_result: FrameExecutionResult) -> None:
            await self._handle_suspend(
                main_frame=main_frame,
                engine_result=engine_result,
                generation_options=generation_options,
                emit=_emit,
                cancel_event=cancel_event,
            )

        async def _runner() -> None:
            try:
                async for event in self._agent_runtime.run_frame_stream(
                    frame=main_frame,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                    on_suspend=on_suspend,
                ):
                    await queue.put(event)
                await queue.put({"event": "done", "data": self._assemble_agent_run_result(main_frame, cancel_event=cancel_event).model_dump()})
            finally:
                await queue.put(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            # 不在正常完成路径下 set 外部传入的 cancel_event（否则会污染调用方共享 token）。
            # cancel_event 由调用方（ChatApplicationService）独占管理。
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                await task

    # ------------------------------------------------------------------
    # 内部：SUSPENDED 重入序列
    # ------------------------------------------------------------------

    async def _handle_suspend(
        self,
        main_frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        generation_options: Optional[Dict[str, Any]],
        emit: Optional[Callable] = None,
        cancel_event=None,
    ) -> None:
        cr = engine_result.call_request
        action_id = engine_result.suspend_action_id
        suspend_text = engine_result.suspend_assistant_text or ""

        logger.info(f"CALL suspend: target={cr.target_alias}, task='{cr.task[:80]}'")

        main_frame.working_history.append(
            {"role": "assistant", "content": suspend_text}
        )

        if emit is not None:
            await emit({"event": "sub_agent_start", "data": {
                "agent_id": cr.target_alias,
                "task": cr.task,
                "iteration": main_frame.progress.iteration,
                "scope": "sub",
                "depth": main_frame.runtime_scope.depth + 1,
                "frame_id": None,
            }})

        self._frame_scheduler.suspend_frame(main_frame)
        sub_result_text = ""
        sub_frame = None
        try:
            sub_profile = await self._agent_profile_resolver.resolve(cr.target_alias)
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

            if emit is None:
                await self._agent_runtime.run_frame(
                    frame=sub_frame,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                )
            else:
                async def _sub_emit(sub_event: Dict[str, Any]) -> None:
                    await emit(sub_event)

                await self._agent_runtime.run_frame_emitting(
                    frame=sub_frame,
                    generation_options=generation_options,
                    stream_emitter=_sub_emit,
                    cancel_event=cancel_event,
                )

            self._frame_scheduler.resume_frame()
            sub_result_text = "".join(sub_frame.progress.text_segments)

            self._harvest_sub_frame_aliases(sub_frame)

            for alias in sub_frame.harvested_aliases:
                if alias not in main_frame.harvested_aliases:
                    main_frame.harvested_aliases.append(alias)

            call_response = MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias=cr.target_alias,
                reply=sub_result_text,
                artifact_aliases=sub_frame.harvested_aliases,
            )

            if emit is not None:
                await emit({"event": "sub_agent_end", "data": {
                    "status": "success",
                    "final_text": sub_result_text,
                    "iteration": main_frame.progress.iteration,
                    "scope": "sub",
                    "depth": main_frame.runtime_scope.depth + 1,
                    "frame_id": sub_frame.runtime_scope.frame_id,
                    "agent_id": cr.target_alias,
                }})

        except Exception as e:
            logger.error(f"Sub-agent execution failed: {e}", exc_info=True)
            self._frame_scheduler.resume_frame()
            error = SubAgentExecutionError(
                params={"agent_alias": cr.target_alias},
                cause=e,
            ).to_error_info()
            call_response = MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias=cr.target_alias,
                error=error,
            )
            if emit is not None:
                await emit({"event": "sub_agent_end", "data": {
                    "status": "error",
                    "iteration": main_frame.progress.iteration,
                    "scope": "sub",
                    "depth": main_frame.runtime_scope.depth + 1,
                    "frame_id": None,
                    "agent_id": cr.target_alias,
                }})

        formatted_call_response = self._mtp_formatter.format_call_response(
            call_response,
            getattr(main_frame.agent_profile, "language", None),
        )

        # iv. append CALL response + tool_result TurnEvent
        main_frame.working_history.append({
            "role": "user",
            "content": formatted_call_response,
        })

        # 找到对应的 tool_call 事件并标记 success
        for ev in main_frame.progress.turn_events:
            if ev.kind == "tool_call" and ev.action_id == action_id:
                ev.status = "success"
                break

        main_frame.progress.turn_events.append(TurnEvent(
            kind="tool_result",
            sequence=main_frame.progress.sequence,
            role="user",
            content=formatted_call_response,
            action_id=action_id,
            tool_kind="CALL",
            tool_name=cr.target_alias,
            status=call_response.status.value,
            render_as="system_call_response",
        ))
        main_frame.progress.sequence += 1

    # ------------------------------------------------------------------
    # 内部：辅助方法（从 loop_executor 迁入）
    # ------------------------------------------------------------------

    async def _fetch_context_refs_content(
        self,
        aliases: List[str],
        identity: "Identity",
        language: Optional[str] = None,
    ) -> str:
        if not aliases:
            return ""
        compiler = MemoryCompiler()
        artifacts: List[CompiledMemory] = []
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
                artifact = compiler.compile(
                    resolved,
                    MemoryCompileTarget.SHARED_CONTEXT,
                    MemoryCompileOptions(requested_alias=alias, language=language),
                )
                artifacts.append(artifact)
            else:
                logger.warning(f"Context ref alias not found: {alias}")
        if not artifacts:
            logger.warning(f"No rendered context returned for context_refs: {aliases}")
            return ""
        return compiler.compile(
            artifacts,
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
        cancel_event=None,
    ) -> AgentRunResult:
        p = frame.progress
        cancelled = cancel_event is not None and cancel_event.is_set()
        if cancelled:
            self._agent_runtime.cancel_tasks_by_run(frame.runtime_scope.run_id)
            tasks = []
        else:
            tasks = self._agent_runtime.collect_tasks_by_run(frame.runtime_scope.run_id)
        return AgentRunResult(
            status=(
                AgentRunStatus.CANCELLED
                if cancelled
                else AgentRunStatus.COMPLETED
            ),
            final_text="".join(p.text_segments),
            mtp_iterations=max(0, p.iteration - 1),
            total_iterations=p.iteration,
            turn_events=p.turn_events,
            materialize_tasks=tasks,
        )


__all__ = ["AgentOrchestrator"]
