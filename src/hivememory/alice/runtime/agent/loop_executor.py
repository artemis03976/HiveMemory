"""
Kernel Loop Executor - 纯单 Agent 执行循环总控

职责（引擎层，agent-数量无关）：
    - 驱动单个 ExecutionFrame 的 generate → MTP → 回填循环
    - 命中 CALL 时返回 FrameExecutionResult(SUSPENDED)，不自我编排
    - 累积产物写入 frame.progress（PCB），支持 CALL 后重入续接

Phase A→B→C→D 循环：
    A. LLM 生成
    B. 自然停止检测
    C. MTP 执行（CALL → 返回 SUSPENDED，交还编排）
    D. 回填 & 继续

不变量：本模块不得出现 sub-agent / topology / 下一个该调谁 词汇。
见 docs/mod/AgentLoopDecouplingDesign.md §3 / §4 Phase 1+2。
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Callable, Awaitable

from hivememory.alice.runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    CallRequest,
    MTPExecutionContext,
)
from hivememory.core.models import TurnEvent
from hivememory.system.config import AgentRuntimeConfig

import json

if TYPE_CHECKING:
    from hivememory.alice.runtime.agent.mtp_executor import MTPExecutor
    from hivememory.alice.runtime.agent.worker_agent import WorkerAgentService

logger = logging.getLogger(__name__)


class KernelLoopExecutor:
    """
    纯单 Agent 执行循环总控。

    接受一个 ExecutionFrame（PCB），驱动 generate→MTP→回填循环直到：
    - 自然收敛 → 返回 FrameExecutionResult(COMPLETED)
    - 命中 CALL → 返回 FrameExecutionResult(SUSPENDED)，控制权交还编排

    累积产物（text_segments / turn_events / write_foci / update_foci /
    pending_aliases / iteration / sequence）全部写在 frame.progress 上，
    重入同一 frame 时自然续接，编号连续。
    """

    def __init__(
        self,
        worker_agent: "WorkerAgentService",
        mtp_executor: "MTPExecutor",
        config: AgentRuntimeConfig,
    ):
        self.worker_agent = worker_agent
        self._mtp_executor = mtp_executor
        self.config = config

    def _namespace_for_frame(self, frame: ExecutionFrame) -> Dict[str, Any]:
        """构造事件命名空间元数据。"""
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "scope": "sub" if frame.is_sub_frame() else "main",
            "depth": frame.runtime_scope.depth,
            "agent_id": agent_id,
            "frame_id": frame.runtime_scope.frame_id,
        }

    async def execute_frame(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
        stream_emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        use_stream_generation: bool = False,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> FrameExecutionResult:
        """
        执行单个帧的循环，直到自然收敛或命中 CALL。

        累积产物写入 frame.progress；重入同一 frame 时续接。
        命中 CALL 时不 fork、不 resume、不组 IPC——直接返回 SUSPENDED。

        Returns:
            FrameExecutionResult: COMPLETED（自然收敛）或 SUSPENDED（命中 CALL）
        """
        p = frame.progress  # PCB 累积器，重入时续接

        while p.iteration < max_iterations:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Generation cancelled by user")
                break

            p.iteration += 1

            result = None
            if use_stream_generation:
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
                        await stream_emitter({"event": "token", "data": token_data})
                if result is None:
                    break
            else:
                result = await self.worker_agent.generate_async(
                    frame.working_history,
                    **(generation_options or {}),
                )

            if not result.was_mtp_interrupted:
                p.text_segments.append(result.text)
                p.turn_events.append(TurnEvent(
                    kind="assistant_message",
                    sequence=p.sequence,
                    role="assistant",
                    content=result.text,
                ))
                p.sequence += 1
                break

            p.text_segments.append(result.prefix_text)
            if result.prefix_text:
                p.turn_events.append(TurnEvent(
                    kind="assistant_message",
                    sequence=p.sequence,
                    role="assistant",
                    content=result.prefix_text,
                ))
                p.sequence += 1

            verb_hint = "UNKNOWN"
            target_hint = ""
            args_hint: Dict[str, Any] = {}
            raw_hint = result.mtp_fragment

            action_id = f"action_{p.iteration}_{p.sequence}"
            mtp_context = MTPExecutionContext(
                identity=frame.identity,
                agent_profile=frame.agent_profile,
                runtime_scope=frame.runtime_scope.with_action(action_id),
            )
            mtp_result = await self._mtp_executor.intercept_and_execute(
                result.text,
                context=mtp_context,
            )

            if mtp_result is not None and mtp_result.command:
                verb_hint = mtp_result.command.verb.value
                target_hint, args_hint, raw_hint = self._extract_command_info(
                    mtp_result.command, raw_hint
                )

            command_event = TurnEvent(
                kind="tool_call",
                sequence=p.sequence,
                role="assistant",
                content=result.text,
                action_id=action_id,
                tool_kind=verb_hint,
                tool_name=target_hint if target_hint else None,
                tool_args=args_hint or None,
                target=target_hint if target_hint else None,
            )
            p.turn_events.append(command_event)
            p.sequence += 1

            if stream_emitter is not None:
                mtp_start_data = {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "iteration": p.iteration,
                }
                mtp_start_data.update(self._namespace_for_frame(frame))
                await stream_emitter({"event": "mtp_start", "data": mtp_start_data})

            if mtp_result is None:
                p.text_segments.append(result.mtp_fragment)
                command_event.status = "failed"
                p.turn_events.append(TurnEvent(
                    kind="tool_result",
                    sequence=p.sequence,
                    role="user",
                    content=result.mtp_fragment,
                    action_id=action_id,
                    tool_kind=verb_hint,
                    tool_name=target_hint if target_hint else None,
                    status="failed",
                ))
                p.sequence += 1
                if stream_emitter is not None:
                    mtp_failed_data = {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": "failed",
                        "iteration": p.iteration,
                    }
                    mtp_failed_data.update(self._namespace_for_frame(frame))
                    await stream_emitter({"event": "mtp_result", "data": mtp_failed_data})
                break

            if mtp_result.response_status == "suspend":
                # CALL 陷入：引擎把控制权交还编排，自己不 fork / resume / 组 IPC。
                # 编排负责：append working_history(CALL文本+⟫) → fork子帧 → 跑子帧
                # → resume → harvest → 组IPC → append IPC → 重入本帧。
                if stream_emitter is not None:
                    mtp_suspend_data = {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": mtp_result.response_status,
                        "iteration": p.iteration,
                    }
                    mtp_suspend_data.update(self._namespace_for_frame(frame))
                    await stream_emitter({"event": "mtp_result", "data": mtp_suspend_data})

                call_params = json.loads(mtp_result.response_content)
                return FrameExecutionResult(
                    status=FrameExecutionStatus.SUSPENDED,
                    call_request=CallRequest(
                        target_alias=call_params["target_alias"],
                        task=call_params["task"],
                        context_refs=call_params.get("context_refs", []),
                    ),
                    suspend_assistant_text=result.text,
                    suspend_action_id=action_id,
                )

            command_event.status = mtp_result.response_status
            if stream_emitter is not None:
                mtp_result_data = {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "status": mtp_result.response_status,
                    "iteration": p.iteration,
                }
                mtp_result_data.update(self._namespace_for_frame(frame))
                await stream_emitter({"event": "mtp_result", "data": mtp_result_data})

            frame.working_history.append(
                {"role": "assistant", "content": result.text + "⟫"}
            )
            frame.working_history.append({
                "role": "user",
                "content": f"[System MTP Execution Result]\n{mtp_result.formatted_response}",
            })
            p.turn_events.append(TurnEvent(
                kind="tool_result",
                sequence=p.sequence,
                role="user",
                content=mtp_result.formatted_response,
                action_id=action_id,
                tool_kind=verb_hint,
                tool_name=target_hint if target_hint else None,
                target=target_hint if target_hint else None,
                status=mtp_result.response_status,
                render_as="system_tool_result",
            ))
            p.sequence += 1

        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    async def execute_frame_stream(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
        # 编排注入的回调：当引擎遇到 SUSPEND 时，编排处理子帧并回填 IPC，
        # 然后引擎继续本段流。签名: (FrameExecutionResult) -> None（异步）。
        on_suspend: Optional[Callable[["FrameExecutionResult"], Awaitable[None]]] = None,
    ):
        """
        执行单个帧的流式循环。

        遇到 CALL SUSPEND 时：
        1. 发出 mtp_result(status=suspend) 事件（已在 execute_frame 内完成）
        2. 调用 on_suspend(result) 让编排处理子帧（sub_agent_start/end、IPC 回填）
        3. 编排回填 working_history 后，引擎重入同一 frame 继续流式输出

        Yields:
            Dict[str, Any]: SSE 事件
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _emit(event: Dict[str, Any]) -> None:
            await queue.put(event)

        async def _runner() -> None:
            try:
                while True:
                    engine_result = await self.execute_frame(
                        frame=frame,
                        max_iterations=max_iterations,
                        generation_options=generation_options,
                        stream_emitter=_emit,
                        use_stream_generation=True,
                        cancel_event=cancel_event,
                    )
                    if engine_result.status == FrameExecutionStatus.SUSPENDED:
                        if on_suspend is not None:
                            await on_suspend(engine_result)
                        # 重入同一 frame（PCB 续接），继续流式输出
                        continue
                    # COMPLETED
                    break
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


__all__ = ["KernelLoopExecutor"]
