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
见 docs/archive/plans/implementation/agent-loop-decoupling.md §3 / §4 Phase 1+2。
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.events import (
    FrameEventSink,
    NullFrameEventSink,
)
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    GenerationResult,
    MTPExecutionContext,
)
from hivememory.core.models import TurnEvent
from hivememory.system.config import AgentRuntimeConfig

if TYPE_CHECKING:
    from hivememory.agent_runtime.mtp.mtp_executor import MTPExecutor
    from hivememory.agent_runtime.worker_agent import WorkerAgentService

logger = logging.getLogger(__name__)


class AgentLoopExecutor:
    """
    纯单 Agent 执行循环总控。

    接受一个 ExecutionFrame（PCB），驱动 generate→MTP→回填循环直到：
    - 自然收敛 → 返回 FrameExecutionResult(COMPLETED)
    - 命中 CALL → 返回 FrameExecutionResult(SUSPENDED)，控制权交还编排

    累積产物（text_segments / turn_events / pending_aliases / iteration / sequence）全部写在 frame.progress 上，
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

    async def _generate_turn(
        self,
        frame: ExecutionFrame,
        generation_options: dict[str, Any] | None,
        sink: FrameEventSink,
        cancel_event: asyncio.Event | None,
    ) -> GenerationResult | FrameExecutionResult:
        if not sink.wants_token_stream:
            return await self.worker_agent.generate_async(
                frame.working_history,
                cancel_event=cancel_event,
                **(generation_options or {}),
            )

        result: GenerationResult | None = None
        async for chunk in self.worker_agent.generate_stream(
            frame.working_history,
            cancel_event=cancel_event,
            **(generation_options or {}),
        ):
            if chunk.is_final:
                result = chunk.result
                break
            if not chunk.mtp_detected and chunk.delta:
                await sink.emit({"event": "token", "data": {"content": chunk.delta}})
        if result is not None:
            return result
        if cancel_event is not None and cancel_event.is_set():
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        return FrameExecutionResult(
            status=FrameExecutionStatus.FAILED,
            error=RuntimeError("Streaming generation ended without a final result."),
        )

    async def execute_frame(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: dict[str, Any] | None = None,
        event_sink: FrameEventSink | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> FrameExecutionResult:
        """
        执行单个帧的循环，直到自然收敛、命中 CALL 或进入其他明确终态。

        累积产物写入 frame.progress；重入同一 frame 时续接。
        命中 CALL 时不 fork、不 resume、不组 CALL response，直接返回 SUSPENDED。

        Returns:
            FrameExecutionResult: 帧停止原因；只有自然收敛返回 COMPLETED。
        """
        p = frame.progress  # PCB 累积器，重入时续接
        sink = event_sink or NullFrameEventSink()

        while p.iteration < max_iterations:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Generation cancelled by user")
                return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)

            p.iteration += 1

            result = await self._generate_turn(
                frame,
                generation_options,
                sink,
                cancel_event,
            )
            if isinstance(result, FrameExecutionResult):
                return result

            if result.finish_reason == "cancelled":
                logger.info("Generation cancelled by user")
                return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)

            if not result.was_mtp_interrupted:
                p.text_segments.append(result.text)
                p.turn_events.append(
                    TurnEvent(
                        kind="assistant_message",
                        sequence=p.sequence,
                        role="assistant",
                        content=result.text,
                    )
                )
                p.sequence += 1
                return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

            p.text_segments.append(result.prefix_text)
            if result.prefix_text:
                p.turn_events.append(
                    TurnEvent(
                        kind="assistant_message",
                        sequence=p.sequence,
                        role="assistant",
                        content=result.prefix_text,
                    )
                )
                p.sequence += 1

            verb_hint = "UNKNOWN"
            target_hint = ""
            args_hint: dict[str, Any] = {}
            raw_hint = result.mtp_fragment

            action_id = f"action_{p.iteration}_{p.sequence}"
            mtp_context = MTPExecutionContext(
                identity=frame.identity,
                agent_profile=frame.agent_profile,
                runtime_scope=frame.runtime_scope.with_action(action_id),
                execution_policy=frame.execution_policy,
            )
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Generation cancelled before MTP execution")
                return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)

            mtp_result = await self._mtp_executor.intercept_and_execute(
                result.text,
                context=mtp_context,
                cancel_event=cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Generation cancelled after MTP execution")
                return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)

            if mtp_result is not None and mtp_result.response_status == "cancelled":
                logger.info("MTP execution cancelled")
                return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)

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

            if event_sink is not None:
                mtp_start_data = {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "iteration": p.iteration,
                    "action_id": action_id,
                }
                await sink.emit({"event": "mtp_start", "data": mtp_start_data})

            if mtp_result is None:
                p.text_segments.append(result.mtp_fragment)
                p.turn_events[-1] = command_event.model_copy(update={"status": "failed"})
                p.turn_events.append(
                    TurnEvent(
                        kind="tool_result",
                        sequence=p.sequence,
                        role="user",
                        content=result.mtp_fragment,
                        action_id=action_id,
                        tool_kind=verb_hint,
                        tool_name=target_hint if target_hint else None,
                        status="failed",
                    )
                )
                p.sequence += 1
                if event_sink is not None:
                    mtp_failed_data = {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": "failed",
                        "iteration": p.iteration,
                        "action_id": action_id,
                    }
                    await sink.emit({"event": "mtp_result", "data": mtp_failed_data})
                return FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED,
                    error=RuntimeError("MTP execution returned no result."),
                )

            if mtp_result.response_status == "suspend":
                # CALL 陷入：引擎把控制权交还编排，自己不 fork / resume / 组 CALL response。
                # 编排负责：append 已归一化的 CALL 文本 → fork子帧 → 跑子帧
                # → resume → harvest → 组 CALL response → append 回填 → 重入本帧。
                if event_sink is not None:
                    mtp_suspend_data = {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": mtp_result.response_status,
                        "iteration": p.iteration,
                        "action_id": action_id,
                    }
                    await sink.emit({"event": "mtp_result", "data": mtp_suspend_data})

                if mtp_result.call_request is None:
                    raise RuntimeError("CALL suspend response missing call_request.")
                return FrameExecutionResult(
                    status=FrameExecutionStatus.SUSPENDED,
                    call_request=mtp_result.call_request,
                    suspend_assistant_text=result.text,
                    suspend_action_id=action_id,
                )

            p.turn_events[-1] = command_event.model_copy(
                update={"status": mtp_result.response_status}
            )
            if event_sink is not None:
                mtp_result_data = {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "status": mtp_result.response_status,
                    "iteration": p.iteration,
                    "action_id": action_id,
                }
                await sink.emit({"event": "mtp_result", "data": mtp_result_data})

            frame.working_history.append({"role": "assistant", "content": result.text})
            frame.working_history.append(
                {
                    "role": "user",
                    "content": mtp_result.formatted_response,
                }
            )
            p.turn_events.append(
                TurnEvent(
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
                )
            )
            p.sequence += 1

        if cancel_event is not None and cancel_event.is_set():
            return FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        return FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)

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


__all__ = ["AgentLoopExecutor"]
