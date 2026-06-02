"""
AgentOrchestrator - 多智能体编排驱动器

职责（编排层）：
    - 造主帧、驱动引擎循环
    - 收到 SUSPENDED 时：fork 子帧 → 驱动引擎跑子帧 → resume → harvest → 组 IPC → 重入
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
from hivememory.core.protocol.models import AgentRunResult
from hivememory.engines.memory_compiler import (
    CompiledMemoryArtifact,
    MemoryCompiler,
    MemoryCompileOptions,
    MemoryCompileTarget,
    MemoryEnvelopeTarget,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler
    from hivememory.agent_runtime.loop_executor import KernelLoopExecutor
    from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
    from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
    from hivememory.agent_runtime.resolver import RuntimeAliasResolver
    from hivememory.core.models import AgentProfile, Identity

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    多智能体编排驱动器。

    持有引擎（KernelLoopExecutor）和编排组件（FrameScheduler /
    AgentProfileResolver / RuntimeAliasResolver），承接原 AgentRuntime 的
    run_agent / run_agent_stream，负责：
      1. 造主帧（create_main_frame）
      2. 循环驱动引擎 execute_frame(main_frame)
      3. SUSPENDED → 重入序列（append CALL文本 → fork/跑子帧/resume/harvest/组IPC → append IPC）
      4. COMPLETED → 从 frame.progress 聚合 AgentRunResult
    """

    def __init__(
        self,
        loop_executor: "KernelLoopExecutor",
        frame_scheduler: "FrameScheduler",
        agent_profile_resolver: "AgentProfileResolver",
        alias_resolver: "RuntimeAliasResolver",
        pending_runtime: Optional["PendingAtomRuntime"] = None,
    ) -> None:
        self._loop_executor = loop_executor
        self._frame_scheduler = frame_scheduler
        self._agent_profile_resolver = agent_profile_resolver
        self._alias_resolver = alias_resolver
        self._pending_runtime = pending_runtime

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
        max_iter = self._loop_executor.config.max_loop_iterations
        main_frame = self._frame_scheduler.create_main_frame(
            agent_profile=agent_profile,
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )
        while True:
            engine_result = await self._loop_executor.execute_frame(
                frame=main_frame,
                max_iterations=max_iter,
                generation_options=generation_options,
                cancel_event=cancel_event,
            )
            if engine_result.status == FrameExecutionStatus.SUSPENDED:
                await self._handle_suspend(
                    main_frame=main_frame,
                    engine_result=engine_result,
                    max_iter=max_iter,
                    generation_options=generation_options,
                )
                continue
            break
        return self._assemble_agent_run_result(main_frame)

    async def run_agent_stream(
        self,
        messages: List[Dict[str, str]],
        identity: "Identity",
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile: Optional["AgentProfile"] = None,
        cancel_event=None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        max_iter = self._loop_executor.config.max_loop_iterations
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
                max_iter=max_iter,
                generation_options=generation_options,
                emit=_emit,
            )

        async def _runner() -> None:
            try:
                async for event in self._loop_executor.execute_frame_stream(
                    frame=main_frame,
                    max_iterations=max_iter,
                    generation_options=generation_options,
                    cancel_event=cancel_event,
                    on_suspend=on_suspend,
                ):
                    await queue.put(event)
                await queue.put({"event": "done", "data": self._assemble_agent_run_result(main_frame).model_dump()})
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
            if cancel_event is not None:
                cancel_event.set()
            await task

    # ------------------------------------------------------------------
    # 内部：SUSPENDED 重入序列
    # ------------------------------------------------------------------

    async def _handle_suspend(
        self,
        main_frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        max_iter: int,
        generation_options: Optional[Dict[str, Any]],
        emit: Optional[Callable] = None,
    ) -> None:
        cr = engine_result.call_request
        action_id = engine_result.suspend_action_id
        suspend_text = engine_result.suspend_assistant_text or ""

        logger.info(f"CALL suspend: target={cr.target_alias}, task='{cr.task[:80]}'")

        main_frame.working_history.append(
            {"role": "assistant", "content": suspend_text + "⟫"}
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
                await self._loop_executor.execute_frame(
                    frame=sub_frame,
                    max_iterations=max_iter,
                    generation_options=generation_options,
                )
            else:
                async def _sub_emit(sub_event: Dict[str, Any]) -> None:
                    await emit(sub_event)

                await self._loop_executor.execute_frame(
                    frame=sub_frame,
                    max_iterations=max_iter,
                    generation_options=generation_options,
                    stream_emitter=_sub_emit,
                    use_stream_generation=True,
                )

            self._frame_scheduler.resume_frame()
            sub_result_text = "".join(sub_frame.progress.text_segments)

            self._harvest_sub_frame_aliases(sub_frame)

            for alias in sub_frame.harvested_aliases:
                if alias not in main_frame.harvested_aliases:
                    main_frame.harvested_aliases.append(alias)

            ipc_response = self._assemble_ipc_return(sub_result_text, sub_frame.harvested_aliases)

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
            ipc_response = (
                '<mtp_response status="error" type="ipc_return">\n'
                f'[Sub-Agent Error]: The sub-agent "{cr.target_alias}" encountered '
                f'an error and could not complete the task.\n'
                f'Action: Try a different approach or continue without the sub-agent.\n'
                '</mtp_response>'
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

        # iv. append IPC + tool_result TurnEvent
        main_frame.working_history.append({
            "role": "user",
            "content": f"[System IPC Return]\n{ipc_response}",
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
            content=ipc_response,
            action_id=action_id,
            tool_kind="CALL",
            tool_name=cr.target_alias,
            status="success",
            render_as="system_ipc_return",
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
        artifacts: List[CompiledMemoryArtifact] = []
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
        return compiler.wrap(
            artifacts,
            envelope_target=MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
            options=MemoryCompileOptions(language=language),
        ).text

    def _harvest_sub_frame_aliases(self, sub_frame: ExecutionFrame) -> None:
        """从子帧 PendingAtomRuntime 重建 harvested_aliases。

        通过 frame_id 过滤子帧的 PendingAtom，收集 pending_alias 用于 IPC [Artifacts]。
        UPDATE fallback：从 tool_call TurnEvent.target 补充尚未注册为 pending 的 alias。
        """
        from hivememory.core.mtp.models import MTPVerb
        harvested = set(sub_frame.harvested_aliases)

        # WRITE/UPDATE aliases from PendingAtomRuntime（主要路径）
        if self._pending_runtime is not None:
            frame_id = sub_frame.runtime_scope.frame_id
            for atom in self._pending_runtime.all_atoms():
                if atom.runtime_scope.frame_id == frame_id:
                    alias = atom.pending_alias
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

    def _assemble_ipc_return(self, sub_final_text: str, harvested_aliases: List[str]) -> str:
        lines = ['<mtp_response status="success" type="ipc_return">']
        lines.append("[Sub-Agent Reply]:")
        lines.append(sub_final_text)
        if harvested_aliases:
            lines.append("")
            lines.append("[Artifacts Generated / Updated]:")
            for alias in harvested_aliases:
                if alias.startswith("draft_") or alias.startswith("rev_"):
                    lines.append(f"- {alias} (pending, readable now)")
                else:
                    lines.append(f"- {alias}")
        lines.append("</mtp_response>")
        return "\n".join(lines)

    def _assemble_agent_run_result(self, frame: ExecutionFrame) -> AgentRunResult:
        p = frame.progress
        run_id = frame.runtime_scope.run_id
        tasks = (
            self._pending_runtime.tasks_by_run(run_id)
            if self._pending_runtime is not None
            else []
        )
        return AgentRunResult(
            final_text="".join(p.text_segments),
            mtp_iterations=max(0, p.iteration - 1),
            total_iterations=p.iteration,
            turn_events=p.turn_events,
            materialize_tasks=tasks,
        )


__all__ = ["AgentOrchestrator"]
