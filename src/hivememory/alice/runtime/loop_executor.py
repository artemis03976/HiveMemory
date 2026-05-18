"""
Kernel Loop Executor - 帧栈驱动的递归生成循环执行器

职责：
    - 管理主 Agent 和子 Agent 的递归执行循环
    - 处理 CALL 指令的挂起/恢复
    - 自动收割子帧生成的记忆别名
    - 组装 IPC 返回 payload

Phase A→B→C→D 循环：
    A. LLM 生成
    B. 自然停止检测
    C. MTP 执行 (可能触发 CALL → 子帧派生)
    D. 回填 & 继续

作者: HiveMemory Team
版本: 3.0 (Phase 2 重构)
"""

import json
import logging
import asyncio
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Callable, Awaitable

from hivememory.core.protocol.models import ChatResult
from hivememory.alice.runtime.execution_frame import ExecutionFrame
from hivememory.core.mtp.models import MTPVerb
from hivememory.core.models import TraceItem, TurnEvent

if TYPE_CHECKING:
    from hivememory.alice.runtime.host import AgentRuntimeHost
    from hivememory.alice.runtime.worker_agent import WorkerAgentService

logger = logging.getLogger(__name__)


class KernelLoopExecutor:
    """
    帧栈驱动的递归生成循环执行器

    封装了 Phase 2 多智能体系统的核心执行逻辑，支持：
    - 主 Agent 递归 MTP 执行
    - 子 Agent 调用 (CALL 指令 → 帧挂起/恢复)
    - 自动收割 (WRITE/UPDATE 别名跟踪)
    - 黑盒隔离 (子 Agent 细节不污染主 Agent)
    """

    def __init__(
        self,
        runtime_host: "AgentRuntimeHost",
        worker_agent: "WorkerAgentService",
    ):
        self._host = runtime_host
        self.worker_agent = worker_agent

    def _namespace_for_frame(self, frame: ExecutionFrame) -> Dict[str, Any]:
        """构造事件命名空间元数据。"""
        # 统一事件命名空间：前端通过 scope/depth 区分主/子 Agent，
        # 不再依赖 sub_token/sub_mtp_* 这类事件名分叉。
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "scope": "sub" if frame.is_sub_frame() else "main",
            "depth": frame.depth,
            "agent_id": agent_id,
            "frame_id": frame.process_id,
        }

    async def execute_main_frame(
        self,
        messages: List[Dict[str, str]],
        max_iterations: Optional[int] = None,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        topic_id: Optional[str] = None,
        identity=None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ChatResult:
        """
        执行主帧的递归生成循环

        Args:
            messages: 初始 messages
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            agent_profile: 人偶图纸配置
            topic_id: 话题 ID
            identity: 完整身份标识

        Returns:
            ChatResult: 递归生成循环的完整结果
        """
        max_iter = max_iterations or self._host.config.koakuma.max_recursion_depth

        main_frame = self._host.frame_scheduler.create_main_frame(
            agent_profile=agent_profile or await self._host.load_agent_profile("omni_doll"),
            frame=main_frame,
            max_iterations=max_iter,
            generation_options=generation_options,
            cancel_event=cancel_event,
        )

    async def execute_main_frame_stream(
        self,
        messages: List[Dict[str, str]],
        max_iterations: Optional[int] = None,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        topic_id: Optional[str] = None,
        identity=None,
        cancel_event: Optional[asyncio.Event] = None,
    ):
        """
        执行主帧的流式递归生成循环

        与 execute_main_frame 相同，但以流式方式 yield SSE 事件。

        Args:
            messages: 初始 messages
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            agent_profile: 人偶图纸配置
            topic_id: 话题 ID
            identity: 完整身份标识

        Yields:
            Dict[str, Any]: SSE 事件 {"event": str, "data": dict}
        """
        max_iter = max_iterations or self._host.config.koakuma.max_recursion_depth

        main_frame = self._host.frame_scheduler.create_main_frame(
            agent_profile=agent_profile or await self._host.load_agent_profile("omni_doll"),
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )

        async for event in self.execute_frame_stream(
            frame=main_frame,
            max_iterations=max_iter,
            generation_options=generation_options,
            cancel_event=cancel_event,
        ):
            yield event

    async def execute_frame(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
        stream_emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        use_stream_generation: bool = False,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ChatResult:
        """
        执行单个帧的递归循环

        这是 Phase 2 的核心方法，同时服务于主 Agent 和子 Agent。
        子 Agent 的 CALL 触发递归调用此方法。

        Phase A→B→C→D:
        A. LLM 生成
        B. 自然停止检测
        C. MTP 执行 (SUSPEND → 子帧派生)
        D. 回填 & 继续

        Args:
            frame: 执行帧
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            stream_emitter: 可选事件发射器（用于 SSE 流式输出）
            use_stream_generation: 是否使用 generate_stream（流式模式）

        Returns:
            ChatResult: 执行结果
        """
        text_segments: List[str] = []
        mtp_commands: List[str] = []
        turn_events: List[TurnEvent] = []
        _seq = 0
        iteration = 0

        self._host.koakuma.set_current_identity(frame.identity)
        self._host.koakuma.set_active_profile(frame.agent_profile)
        self._host.koakuma.set_current_depth(frame.depth)
        self._host.koakuma.reset_interaction_state()

        while iteration < max_iterations:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Generation cancelled by user")
                break

            iteration += 1

            result = None
            if use_stream_generation:
                # 流式模式：逐 chunk 推送 token，但最终仍收敛为一个 result，
                # 后续 MTP 拦截/执行逻辑与非流式共用同一骨架。
                async for chunk in self.worker_agent.generate_stream(
                    frame.working_history,
                    cancel_event=cancel_event,
                    **(generation_options or {}),
                ):
                    if chunk.is_final:
                        result = chunk.result
                        break
                    if (
                        stream_emitter is not None
                        and not chunk.mtp_detected
                        and chunk.delta
                    ):
                        token_data = {"content": chunk.delta}
                        token_data.update(self._namespace_for_frame(frame))
                        await stream_emitter(
                            {"event": "token", "data": token_data}
                        )
                if result is None:
                    break
            else:
                result = await self.worker_agent.generate_async(
                    frame.working_history,
                    **(generation_options or {}),
                )

            if not result.was_mtp_interrupted:
                text_segments.append(result.text)
                turn_events.append(TurnEvent(
                    kind="assistant_message",
                    sequence=_seq,
                    role="assistant",
                    content=result.text,
                ))
                _seq += 1
                break

            text_segments.append(result.prefix_text)
            if result.prefix_text:
                turn_events.append(TurnEvent(
                    kind="assistant_message",
                    sequence=_seq,
                    role="assistant",
                    content=result.prefix_text,
                ))
                _seq += 1
            raw_hint = result.mtp_fragment
            verb_hint = "UNKNOWN"
            target_hint = ""
            args_hint: Dict[str, Any] = {}

            # MTP 的语义解析以 KoakumaRuntime 返回结果为唯一真相来源。
            # LoopExecutor 不再自行 parse 指令字符串，避免双真相漂移。
            mtp_result = await self._host.koakuma.intercept_and_execute(result.text)
            if mtp_result is not None and mtp_result.command:
                verb_hint = mtp_result.command.verb.value
                target_hint, args_hint, raw_hint = self._extract_command_info(
                    mtp_result.command, raw_hint
                )
            action_id = f"action_{iteration}_{_seq}"
            command_event = TurnEvent(
                kind="tool_call",
                sequence=_seq,
                role="assistant",
                content=result.text,
                action_id=action_id,
                tool_kind=verb_hint,
                tool_name=target_hint if target_hint else None,
                tool_args=args_hint or None,
                target=target_hint if target_hint else None,
            )
            turn_events.append(command_event)
            _seq += 1
            if stream_emitter is not None:
                mtp_start_data = {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "iteration": iteration,
                }
                mtp_start_data.update(self._namespace_for_frame(frame))
                await stream_emitter(
                    {
                        "event": "mtp_start",
                        "data": mtp_start_data,
                    }
                )

            if mtp_result is None:
                text_segments.append(result.mtp_fragment)
                command_event.status = "failed"
                turn_events.append(TurnEvent(
                    kind="tool_result",
                    sequence=_seq,
                    role="user",
                    content=result.mtp_fragment,
                    action_id=action_id,
                    tool_kind=verb_hint,
                    tool_name=target_hint if target_hint else None,
                    status="failed",
                ))
                _seq += 1
                if stream_emitter is not None:
                    mtp_failed_data = {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": "failed",
                        "iteration": iteration,
                    }
                    mtp_failed_data.update(self._namespace_for_frame(frame))
                    await stream_emitter(
                        {
                            "event": "mtp_result",
                            "data": mtp_failed_data,
                        }
                    )
                break

            if mtp_result.response_status == "suspend":
                if stream_emitter is not None:
                    mtp_suspend_data = {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": mtp_result.response_status,
                        "iteration": iteration,
                    }
                    mtp_suspend_data.update(self._namespace_for_frame(frame))
                    await stream_emitter(
                        {
                            "event": "mtp_result",
                            "data": mtp_suspend_data,
                        }
                    )
                stream_events: Optional[List[Dict[str, Any]]] = (
                    [] if stream_emitter is not None else None
                )
                # CALL 的挂起/恢复与子帧执行统一交给 _execute_call，
                # execute_frame 仅负责主循环编排与历史回填。
                ipc_response = await self._execute_call(
                    frame=frame,
                    mtp_result=mtp_result,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                    stream_events=stream_events,
                    iteration=iteration,
                )
                if stream_emitter is not None and stream_events:
                    for event in stream_events:
                        await stream_emitter(event)

                frame.working_history.append(
                    {"role": "assistant", "content": result.text + "⟫"}
                )
                frame.working_history.append({
                    "role": "user",
                    "content": f"[System IPC Return]\n{ipc_response}",
                })
                command_event.status = "success"
                turn_events.append(TurnEvent(
                    kind="tool_result",
                    sequence=_seq,
                    role="user",
                    content=ipc_response,
                    action_id=action_id,
                    tool_kind="CALL",
                    tool_name=target_hint if target_hint else None,
                    status="success",
                    render_as="system_ipc_return",
                ))
                _seq += 1
                mtp_commands.append("CALL")
                continue

            mtp_commands.append(verb_hint)
            command_event.status = mtp_result.response_status
            if stream_emitter is not None:
                mtp_result_data = {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "status": mtp_result.response_status,
                    "iteration": iteration,
                }
                mtp_result_data.update(self._namespace_for_frame(frame))
                await stream_emitter(
                    {
                        "event": "mtp_result",
                        "data": mtp_result_data,
                    }
                )

            frame.working_history.append(
                {"role": "assistant", "content": result.text + "⟫"}
            )
            frame.working_history.append({
                "role": "user",
                "content": f"[System MTP Execution Result]\n{mtp_result.formatted_response}",
            })
            turn_events.append(TurnEvent(
                kind="tool_result",
                sequence=_seq,
                role="user",
                content=mtp_result.formatted_response,
                action_id=action_id,
                tool_kind=verb_hint,
                tool_name=target_hint if target_hint else None,
                target=target_hint if target_hint else None,
                status=mtp_result.response_status,
                render_as="system_tool_result",
            ))
            _seq += 1

            if frame.is_sub_frame() and mtp_result.command:
                self._try_harvest_alias(frame, mtp_result)

        loop_result = ChatResult(
            final_text="".join(text_segments),
            mtp_iterations=max(0, iteration - 1),
            total_iterations=iteration,
            mtp_commands_executed=mtp_commands,
            turn_events=turn_events,
        )
        if stream_emitter is not None:
            await stream_emitter({"event": "done", "data": loop_result.model_dump()})
        return loop_result

    async def execute_frame_stream(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ):
        """
        执行单个帧的流式递归循环

        与 execute_frame 相同的核心逻辑，但逐 token yield LLM 生成内容。

        Yields:
            Dict[str, Any]: SSE 事件
                - {"event": "token", "data": {"content": str}}
                - {"event": "mtp_start", "data": {...}}
                - {"event": "mtp_result", "data": {...}}
                - {"event": "sub_agent_start", "data": {...}}
                - {"event": "sub_agent_end", "data": {...}}
                - 所有 token/mtp* 事件通过 data.scope 区分 main/sub
                - {"event": "done", "data": ChatResult}
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _emit(event: Dict[str, Any]) -> None:
            await queue.put(event)

        async def _runner() -> None:
            try:
                # 关键设计：流式仅是 execute_frame 的“输出策略”，
                # 不再维护第二套循环实现。
                await self.execute_frame(
                    frame=frame,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                    stream_emitter=_emit,
                    use_stream_generation=True,
                    cancel_event=cancel_event,
                )
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

    def _extract_command_info(self, command, raw_hint):
        """从 MTP 命令中提取信息"""
        if command.target.is_wildcard:
            target_hint = "*"
        elif command.target.aliases:
            target_hint = ",".join(command.target.aliases)
        else:
            target_hint = ""
        args_hint = dict(command.args or {})
        raw_hint = command.raw_text or raw_hint
        return target_hint, args_hint, raw_hint

    async def _execute_call(
        self,
        frame: ExecutionFrame,
        mtp_result,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
        stream_events: Optional[List[Dict[str, Any]]] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        执行 CALL 指令并返回 IPC payload（统一的流式/非流式实现）

        当提供 stream_events 时，会额外写入子 Agent 的流式事件：
        sub_agent_start / sub_* / sub_agent_end。

        Args:
            frame: 当前帧（将被挂起）
            mtp_result: MTP 执行结果（含 CALL 参数）
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            stream_events: 可选事件容器（流式模式）
            iteration: 当前迭代次数（流式事件展示用）

        Returns:
            str: 格式化的 IPC 返回 payload (XML 格式)
        """
        call_params = json.loads(mtp_result.response_content)
        target_alias = call_params["target_alias"]
        task = call_params["task"]
        context_refs = call_params.get("context_refs", [])

        logger.info(
            f"CALL suspend: target={target_alias}, task='{task[:80]}...'"
        )

        if stream_events is not None:
            sub_start_data = {
                "agent_id": target_alias,
                "task": task,
                "iteration": iteration,
                "scope": "sub",
                "depth": frame.depth + 1,
                "frame_id": None,
            }
            stream_events.append({"event": "sub_agent_start", "data": sub_start_data})

        self._host.frame_scheduler.suspend_frame(frame)

        try:
            sub_frame = await self._host.frame_scheduler.fork_sub_frame(
                parent_frame=frame,
                target_alias=target_alias,
                task=task,
                context_refs=context_refs,
            )

            if stream_events is None:
                sub_result = await self.execute_frame(
                    frame=sub_frame,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                )
            else:
                async def _sub_emit(sub_event: Dict[str, Any]) -> None:
                    if sub_event["event"] == "done":
                        return
                    # 子帧事件不再改名为 sub_token/sub_mtp_*，
                    # 统一沿用 token/mtp_*，通过 data.scope=sub 区分。
                    stream_events.append(sub_event)

                sub_result = await self.execute_frame(
                    frame=sub_frame,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                    stream_emitter=_sub_emit,
                    use_stream_generation=True,
                )

            self._host.frame_scheduler.resume_frame()

            self._host.koakuma.set_current_identity(frame.identity)
            self._host.koakuma.set_active_profile(frame.agent_profile)
            self._host.koakuma.set_current_depth(frame.depth)

            self._host.koakuma._current_traces.append(TraceItem(
                action="CALL",
                target=target_alias,
                status="success",
            ))

            if stream_events is not None:
                sub_end_data = {
                    "status": "success",
                    "final_text": sub_result.final_text,
                    "iteration": iteration,
                    "scope": "sub",
                    "depth": frame.depth + 1,
                    "frame_id": sub_frame.process_id,
                    "agent_id": target_alias,
                }
                stream_events.append({"event": "sub_agent_end", "data": sub_end_data})

            return self._assemble_ipc_return(
                sub_result=sub_result,
                harvested_aliases=sub_frame.harvested_aliases,
            )

        except Exception as e:
            logger.error(f"Sub-agent execution failed: {e}", exc_info=True)

            self._host.frame_scheduler.resume_frame()
            self._host.koakuma.set_current_identity(frame.identity)
            self._host.koakuma.set_active_profile(frame.agent_profile)
            self._host.koakuma.set_current_depth(frame.depth)

            self._host.koakuma._current_traces.append(TraceItem(
                action="CALL",
                target=target_alias,
                status="error",
            ))

            if stream_events is not None:
                sub_end_err_data = {
                    "status": "error",
                    "iteration": iteration,
                    "scope": "sub",
                    "depth": frame.depth + 1,
                    "frame_id": None,
                    "agent_id": target_alias,
                }
                stream_events.append({"event": "sub_agent_end", "data": sub_end_err_data})

            return (
                '<mtp_response status="error" type="ipc_return">\n'
                f'[Sub-Agent Error]: The sub-agent "{target_alias}" encountered '
                f'an error and could not complete the task.\n'
                f'Action: Try a different approach or continue without the sub-agent.\n'
                '</mtp_response>'
            )

    def _assemble_ipc_return(
        self,
        sub_result: ChatResult,
        harvested_aliases: List[str],
    ) -> str:
        """
        组装 IPC 返回 payload (XML 格式)

        将子 Agent 的自然语言回复与自动收割的记忆指针混合打包。

        Args:
            sub_result: 子 Agent 执行结果
            harvested_aliases: 子 Agent 生成的记忆别名列表

        Returns:
            str: 格式化的 IPC 返回 payload
        """
        lines = ['<mtp_response status="success" type="ipc_return">']
        lines.append("[Sub-Agent Reply]:")
        lines.append(sub_result.final_text)

        if harvested_aliases:
            lines.append("")
            lines.append("[Artifacts Generated / Updated]:")
            for alias in harvested_aliases:
                atom = self._host.koakuma.atom_cache.get_atom_by_alias(alias)
                if atom and hasattr(atom, 'index') and atom.index.summary:
                    summary = atom.index.summary[:60]
                    lines.append(f"- {alias} ({summary})")
                else:
                    lines.append(f"- {alias}")

        lines.append("</mtp_response>")
        return "\n".join(lines)

    def _try_harvest_alias(self, frame: ExecutionFrame, mtp_result) -> None:
        """
        尝试从 MTP 执行结果中收割别名 (仅子帧)

        当子 Agent 执行 WRITE/UPDATE 时，提取生成的别名并
        添加到帧的 harvested_aliases 列表中。

        Args:
            frame: 当前子帧
            mtp_result: MTP 执行结果
        """
        if not mtp_result.command:
            return

        verb = mtp_result.command.verb
        if verb not in (MTPVerb.WRITE, MTPVerb.UPDATE):
            return

        if verb == MTPVerb.UPDATE:
            alias = mtp_result.command.target.single_alias
            if alias:
                frame.add_harvested_alias(alias)
                logger.debug(f"Harvested UPDATE alias: {alias}")

        elif verb == MTPVerb.WRITE:
            alias = self._host.koakuma.get_last_generated_alias()
            if alias:
                frame.add_harvested_alias(alias)
                logger.debug(f"Harvested WRITE alias: {alias}")


__all__ = ["KernelLoopExecutor"]
