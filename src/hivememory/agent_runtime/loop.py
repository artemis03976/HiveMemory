"""
Agent Loop Executor - 纯单 Agent 执行循环总控

职责（引擎层，agent-数量无关）：
    - 驱动单个 ExecutionFrame 的 generate → MTP → 回填循环
    - 命中 CALL 时返回 FrameExecutionResult(SUSPENDED)，不自我编排
    - 累积产物写入 frame.progress（PCB），支持 CALL 后重入续接

每轮依次执行：
    A. LLM 生成
    B. 自然停止检测
    C. MTP 执行（CALL → 返回 SUSPENDED，交还编排）
    D. 回填 & 继续

本模块只推进当前帧，帧之间的调度由调用方负责。
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.model.budget import (
    TurnContextBudgetExceededError,
    estimate_context_tokens,
)
from hivememory.agent_runtime.model.generator import GenerationResult
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.output import (
    FrameOutputSink,
    MTPFinished,
    MTPStarted,
    TokenDelta,
)
from hivememory.config.alice import AgentRuntimeConfig
from hivememory.core.models import TurnEvent, freeze_mapping
from hivememory.core.models.interaction import ToolResultStatus

if TYPE_CHECKING:
    from hivememory.agent_runtime.model.generator import MTPGenerator
    from hivememory.agent_runtime.mtp.runtime import MTPExecutor
    from hivememory.infrastructure.llm.base import LLMRequestParams

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
        generator: "MTPGenerator",
        mtp_executor: "MTPExecutor",
        config: AgentRuntimeConfig,
    ):
        self.generator = generator
        self._mtp_executor = mtp_executor
        self.config = config

    async def _generate_turn(
        self,
        frame: ExecutionFrame,
        params: "LLMRequestParams",
        sink: FrameOutputSink,
    ) -> GenerationResult | FrameExecutionResult:
        """执行一轮流式 LLM 生成：自然语言增量逐条送达 sink，并缓冲到帧进度。

        生成总是流式，停止时已送达的文字可以封为未完成回复；交付方不需要
        增量时由 sink 自行丢弃。
        """
        result: GenerationResult | None = None
        # 生成器负责关闭 LLM 服务的流，服务再关闭底层响应；本层只关闭生成器的流。
        chunks = self.generator.generate(frame.working_history, params, stream=True)

        try:
            while True:
                try:
                    chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.error("Streaming frame generation failed: %s", error, exc_info=True)
                    return FrameExecutionResult(
                        status=FrameExecutionStatus.FAILED,
                        error=error,
                    )
                if chunk.is_final:
                    result = chunk.result
                    break
                if not chunk.mtp_detected and chunk.delta:
                    await sink.send(TokenDelta(content=chunk.delta))
                    frame.progress.streaming_text.append(chunk.delta)
        finally:
            close = getattr(chunks, "aclose", None)
            if callable(close):
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("关闭 LLM stream 失败", exc_info=True)

        if result is not None:
            return result
        return FrameExecutionResult(
            status=FrameExecutionStatus.FAILED,
            error=RuntimeError("Streaming generation ended without a final result."),
        )

    async def execute_frame(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        output_sink: FrameOutputSink,
    ) -> FrameExecutionResult:
        """
        执行单个帧的循环，直到自然收敛、命中 CALL 或进入其他明确终态。

        累积产物写入 frame.progress；重入同一 frame 时续接。
        命中 CALL 时不 fork、不 resume、不组 CALL response，直接返回 SUSPENDED。
        帧必须已解析模型：请求参数与输入预算都取自 ``frame.resolved_model``。

        Returns:
            FrameExecutionResult: 帧停止原因；只有自然收敛返回 COMPLETED。
        """
        resolved_model = frame.resolved_model
        if resolved_model is None:
            raise ValueError("Frame model must be resolved before execution.")
        p = frame.progress  # PCB 累积器，重入时续接

        while p.iteration < max_iterations:
            # 工具结果与 CALL 回填都可能扩大历史；必须在下一次生成前重新检查。
            estimated_tokens = estimate_context_tokens(frame.working_history)
            limit = resolved_model.input_token_limit
            if estimated_tokens > limit:
                return FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED,
                    error=TurnContextBudgetExceededError(estimated_tokens, limit),
                )
            p.iteration += 1

            try:
                result = await self._generate_turn(frame, resolved_model.params, output_sink)
            except asyncio.CancelledError:
                self._seal_partial_text(frame)
                raise
            if isinstance(result, FrameExecutionResult):
                self._seal_partial_text(frame)
                return result
            # 完整结果已经包含流出的文字，随后记录时不能再次追加缓冲。
            p.streaming_text.clear()

            if not result.was_mtp_interrupted:
                p.text_segments.append(result.text)
                p.turn_events.append(
                    TurnEvent(kind="assistant_message", sequence=p.sequence, content=result.text)
                )
                p.sequence += 1
                return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

            # 指令前正文只记在 assistant_message 中，调用事件只存指令片段。
            p.text_segments.append(result.prefix_text)
            if result.prefix_text:
                p.turn_events.append(
                    TurnEvent(
                        kind="assistant_message", sequence=p.sequence, content=result.prefix_text
                    )
                )
                p.sequence += 1

            action_id = f"action_{p.iteration}_{p.sequence}"
            mtp_context = MTPExecutionContext(
                runtime_scope=frame.runtime_scope.with_action(action_id),
                execution_policy=frame.execution_policy,
                submit_operation=frame.submit_operation,
                language=frame.language,
            )
            mtp_result = await self._mtp_executor.intercept_and_execute(
                result.text,
                context=mtp_context,
            )
            if mtp_result is None:
                # 生成器只在文本含左定界符时报告 MTP 中断，执行器对这样的文本总会
                # 给出结果（解析失败也是 error 结果）；走到这里说明两者的约定被破坏。
                raise RuntimeError("MTP executor returned no result for an interrupted generation.")

            command = mtp_result.command
            verb_hint = "UNKNOWN"
            target_hint = ""
            args_hint: dict[str, Any] = {}
            raw_hint = result.mtp_fragment
            if command:
                verb_hint = command.verb.value
                target_hint, args_hint, raw_hint = self._extract_command_info(command, raw_hint)

            # 指令无法解析时动词与作用对象都为空，原文仍在 content 中，随后是 error 结果。
            p.turn_events.append(
                TurnEvent(
                    kind="tool_call",
                    sequence=p.sequence,
                    content=result.mtp_fragment,
                    action_id=action_id,
                    verb=command.verb.value if command else None,
                    targets=tuple(command.target.aliases) if command else (),
                    tool_args=freeze_mapping(command.args) if command and command.args else None,
                )
            )
            p.sequence += 1

            await output_sink.send(
                MTPStarted(
                    verb=verb_hint,
                    target=target_hint,
                    args=args_hint,
                    raw_text=raw_hint,
                    iteration=p.iteration,
                    action_id=action_id,
                )
            )

            if mtp_result.response_status == "suspend":
                # CALL 陷入：引擎把控制权交还编排，自己不 fork / resume / 组 CALL response。
                # 编排负责：append 已归一化的 CALL 文本 → fork子帧 → 跑子帧
                # → resume → harvest → 组 CALL response → append 回填 → 重入本帧。
                # 挂起时只有调用事件；结局在回填时作为结果事件写出。
                await output_sink.send(
                    MTPFinished(
                        verb=verb_hint,
                        target=target_hint,
                        args=args_hint,
                        raw_text=raw_hint,
                        status=mtp_result.response_status,
                        iteration=p.iteration,
                        action_id=action_id,
                    )
                )

                if mtp_result.call_request is None:
                    raise RuntimeError("CALL suspend response missing call_request.")
                return FrameExecutionResult(
                    status=FrameExecutionStatus.SUSPENDED,
                    call_request=mtp_result.call_request,
                    suspend_assistant_text=result.text,
                    suspend_action_id=action_id,
                )

            # 只记录已收到 ACK 的产物句柄；登记归 workspace，正式物化归 Patchouli。
            if mtp_result.response_status == "ack" and mtp_result.pending_alias:
                frame.add_harvested_alias(mtp_result.pending_alias)

            await output_sink.send(
                MTPFinished(
                    verb=verb_hint,
                    target=target_hint,
                    args=args_hint,
                    raw_text=raw_hint,
                    status=mtp_result.response_status,
                    iteration=p.iteration,
                    action_id=action_id,
                )
            )

            frame.working_history.append({"role": "assistant", "content": result.text})
            frame.working_history.append(
                {
                    "role": "user",
                    "content": mtp_result.formatted_response,
                }
            )
            # 回填文本带本地化标题，原样记录为 Alice 收到的返回。响应状态在事件
            # 构造时校验：warning 只作为 success 上的警告出现，不是状态。
            p.turn_events.append(
                TurnEvent(
                    kind="tool_result",
                    sequence=p.sequence,
                    content=mtp_result.formatted_response,
                    action_id=action_id,
                    status=cast(ToolResultStatus, mtp_result.response_status),
                    produced_aliases=(
                        (mtp_result.pending_alias,) if mtp_result.pending_alias else ()
                    ),
                )
            )
            p.sequence += 1

        return FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)

    @staticmethod
    def _seal_partial_text(frame: ExecutionFrame) -> None:
        """把已交付但尚未完成的文字同步封入进度，保持事件编号连续。

        未完成不记在事件上：本轮结局由封口后的交互记录表达。
        """
        progress = frame.progress
        text = "".join(progress.streaming_text)
        progress.streaming_text.clear()
        if text:
            progress.text_segments.append(text)
            progress.turn_events.append(
                TurnEvent(kind="assistant_message", sequence=progress.sequence, content=text)
            )
            progress.sequence += 1

    def _extract_command_info(self, command, raw_hint):
        """从解析出的 MTP 命令中提取 target 提示、args 与原文，用于流式交付的 MTP 事件。"""
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
