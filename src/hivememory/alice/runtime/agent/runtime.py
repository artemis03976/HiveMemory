from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.events import CallbackFrameEventSink, FrameEventSink
from hivememory.agent_runtime.loop_executor import AgentLoopExecutor
from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.products import FrameProducts, RuntimeProducts
from hivememory.agent_runtime.worker_agent import WorkerAgentService
from hivememory.core.models import TurnEvent
from hivememory.core.mtp import MTPCallResponse, MTPFormatter

if TYPE_CHECKING:
    from hivememory.agent_runtime.models import ExecutionFrame
    from hivememory.agent_runtime.mtp.mtp_executor import MTPExecutor
    from hivememory.system.config import AliceConfig
    from hivememory.system.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AgentRuntime:
    """单 Agent 运行时门面。

    封装执行引擎（loop_executor + worker_agent + mtp_executor）与
    PendingAtomRuntime，对外提供 "跑一个 frame" 和 "收割 Task" 的 API。
    类比 patchouli 的 LibrarianCore——把底层引擎组件收拢成一个聚合，
    编排层（AgentOrchestrator）只拿这个门面操作，不直接接触引擎细节。

    迭代上限由门面内部从 config.agent_runtime 消化，编排不再传 max_iterations。
    """

    def __init__(
        self,
        *,
        mtp_executor: MTPExecutor,
        alice_config: AliceConfig,
        pending_runtime: PendingAtomRuntime | None = None,
        loop_executor: AgentLoopExecutor | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._pending_runtime = pending_runtime or PendingAtomRuntime()
        # 模型注册表：在每个 frame 开始时根据 agent_profile.model_name 解析实际模型。
        # 若未提供，generation_options 中必须已包含 model，否则 WorkerAgentService 会报错。
        self._model_registry = model_registry
        if loop_executor is not None:
            self._loop_executor = loop_executor
        else:
            # WorkerAgentService 是无状态服务，不依赖任何实例级 LLM 配置。
            # 所有 LLM 参数均在运行时由 _resolve_model_for_frame 注入。
            worker_agent = WorkerAgentService()
            self._loop_executor = AgentLoopExecutor(
                worker_agent=worker_agent,
                mtp_executor=mtp_executor,
                config=alice_config.runtime,
            )

    @property
    def _max_iterations(self) -> int:
        return self._loop_executor.config.max_loop_iterations

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    async def run_frame(
        self,
        frame: ExecutionFrame,
        generation_options: dict[str, Any] | None = None,
        *,
        event_sink: FrameEventSink | None = None,
        cancel_event=None,
    ) -> FrameExecutionResult:
        """跑一个 frame 到自然收敛或命中 CALL（非流式）。"""
        generation_options = self._resolve_model_for_frame(frame, generation_options)
        max_iterations = frame.execution_policy.max_iterations or self._max_iterations
        return await self._loop_executor.execute_frame(
            frame=frame,
            max_iterations=max_iterations,
            generation_options=generation_options,
            event_sink=event_sink,
            cancel_event=cancel_event,
        )

    def run_frame_stream(
        self,
        frame: ExecutionFrame,
        generation_options: dict[str, Any] | None = None,
        cancel_event=None,
        on_suspend: Callable[[FrameExecutionResult], Awaitable[None]] | None = None,
        on_terminal: Callable[[FrameExecutionResult], Awaitable[None]] | None = None,
        event_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """跑一个 frame 并逐 token 流式输出；命中 CALL 时回调 on_suspend。"""

        async def _stream() -> AsyncGenerator[dict[str, Any], None]:
            queue: asyncio.Queue = asyncio.Queue()

            async def _emit(event: dict[str, Any]) -> None:
                await queue.put(event)

            sink = CallbackFrameEventSink(_emit, metadata=event_metadata)

            async def _runner() -> None:
                try:
                    while True:
                        result = await self.run_frame(
                            frame,
                            generation_options=generation_options,
                            event_sink=sink,
                            cancel_event=cancel_event,
                        )
                        if result.status == FrameExecutionStatus.SUSPENDED:
                            if on_suspend is not None:
                                await on_suspend(result)
                                continue
                            result = FrameExecutionResult(
                                status=FrameExecutionStatus.FAILED,
                                error=RuntimeError(
                                    "Frame suspended without an orchestration callback."
                                ),
                            )
                        if on_terminal is not None:
                            await on_terminal(result)
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
                await task

        return _stream()

    async def run_frame_emitting(
        self,
        frame: ExecutionFrame,
        generation_options: dict[str, Any] | None = None,
        stream_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        event_metadata: dict[str, Any] | None = None,
        cancel_event=None,
    ) -> FrameExecutionResult:
        """跑一个 frame，逐 token 推给 stream_emitter（供编排跑流式子帧）。"""
        event_sink = (
            CallbackFrameEventSink(stream_emitter, metadata=event_metadata)
            if stream_emitter is not None
            else None
        )
        return await self.run_frame(
            frame=frame,
            generation_options=generation_options,
            event_sink=event_sink,
            cancel_event=cancel_event,
        )

    def apply_call_response(
        self,
        frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        response: MTPCallResponse,
    ) -> None:
        """Apply one resolved CALL response to the suspended frame exactly once."""
        if suspension.status != FrameExecutionStatus.SUSPENDED:
            raise ValueError("CALL response requires a suspended frame result.")
        call_request = suspension.call_request
        action_id = suspension.suspend_action_id
        if call_request is None or not action_id:
            raise ValueError("CALL suspension is missing request or action id.")
        if response.agent_alias != call_request.target_alias:
            raise ValueError(
                "CALL response target does not match the suspended request: "
                f"expected={call_request.target_alias!r}, got={response.agent_alias!r}"
            )

        matching_calls = [
            event
            for event in frame.progress.turn_events
            if event.kind == "tool_call" and event.action_id == action_id
        ]
        if len(matching_calls) != 1:
            raise ValueError(
                f"CALL suspension has {len(matching_calls)} matching tool_call events: "
                f"action_id={action_id}"
            )
        if any(
            event.kind == "tool_result" and event.action_id == action_id
            for event in frame.progress.turn_events
        ):
            raise ValueError(f"CALL response was already applied: action_id={action_id}")

        formatted_response = MTPFormatter.format_call_response(
            response,
            getattr(frame.agent_profile, "language", None),
        )
        assistant_text = suspension.suspend_assistant_text or ""
        frame.working_history.append({"role": "assistant", "content": assistant_text})
        frame.working_history.append({"role": "user", "content": formatted_response})
        frame.progress.turn_events[frame.progress.turn_events.index(matching_calls[0])] = (
            matching_calls[0].model_copy(update={"status": response.status.value})
        )
        frame.progress.turn_events.append(
            TurnEvent(
                kind="tool_result",
                sequence=frame.progress.sequence,
                role="user",
                content=formatted_response,
                action_id=action_id,
                tool_kind="CALL",
                tool_name=call_request.target_alias,
                status=response.status.value,
                render_as="system_call_response",
            )
        )
        frame.progress.sequence += 1
        for alias in response.artifact_aliases:
            frame.add_harvested_alias(alias)

    def _resolve_model_for_frame(
        self,
        frame: ExecutionFrame,
        generation_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        根据 agent_profile 从注册表解析 LLM 参数，注入到 generation_options
        供 WorkerAgentService 使用，并把展示名写入 frame.progress.model_used。

        参数覆盖优先级（从高到低）：
            会话请求 (generation_options) > Agent Profile > 模型注册表定义

        - model：会话传入的注册表 ID 覆盖 profile.model_name（供侧边栏实时切换模型调试）
        - api_key / api_base：来自解析出的模型（Profile 仅通过 model_name 选择模型）
        - temperature / top_p：会话覆盖优先，其次 Profile，最后模型定义默认
        - max_tokens：会话覆盖优先，其次模型定义默认

        设计原则：
        - 注册表未注入（model_registry 为 None）时，由调用方通过 generation_options
          直接传 model；WorkerAgentService 缺 model 时会抛出明确错误。
        - 注册表解析失败不捕获，让错误向上传播——静默降级到"某个"模型不是用户意图。
        """
        if self._model_registry is None:
            return generation_options or {}

        profile = frame.agent_profile
        session_opts = generation_options or {}

        # 模型选择：会话传入的注册表 ID 优先，否则用 Agent Profile 的 model_name。
        # 二者语义一致（都是注册表 ID / 'default'），resolve 统一解析。
        session_model_id = session_opts.get("model")
        effective_model_name = session_model_id or profile.model_name

        # 覆盖值：会话请求优先于 Agent Profile；两者皆无则由 resolve 回落到模型定义
        temperature_override = session_opts.get("temperature")
        if temperature_override is None:
            temperature_override = profile.temperature

        top_p_override = session_opts.get("top_p")
        if top_p_override is None:
            top_p_override = profile.top_p

        max_tokens_override = session_opts.get("max_tokens")

        # 解析失败向上传播（model_name 不在注册表中属于配置错误，应立即暴露）
        llm_config, display_name = self._model_registry.resolve(
            effective_model_name,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
            top_p_override=top_p_override,
        )

        # 记录展示名，供 Orchestrator 组装 AgentRunResult 时读取
        frame.progress.model_used = display_name

        # 用解析结果覆盖 generation_options 中的对应键——
        # resolve 已完成三级优先级合并，这里以其结果为准（包括把会话传入的
        # 注册表 ID 替换为 litellm 模型标识符，供 WorkerAgentService 使用）。
        # api_key / api_base 为 None 是合法状态（litellm 从环境变量读取）。
        resolved: dict[str, Any] = {
            **session_opts,
            "model": llm_config.model,
            "api_key": llm_config.api_key,
            "api_base": llm_config.api_base,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "top_p": llm_config.top_p,
        }
        logger.debug(
            f"模型解析: effective={effective_model_name!r} "
            f"(session={session_model_id!r}, profile={profile.model_name!r}) → "
            f"litellm_model={llm_config.model!r}, display={display_name!r}, "
            f"temperature={llm_config.temperature}, top_p={llm_config.top_p}"
        )
        return resolved

    def mark_task_failed(self, pending_alias: str) -> None:
        """将 MATERIALIZING 的 atom 迁移到 FAILED（由 patchouli FAILED 事件触发）。"""
        self._pending_runtime.mark_failed(pending_alias)

    def mark_task_cancelled(self, pending_alias: str) -> None:
        """将 in-flight atom 迁移到 CANCELLED（由 patchouli CANCELLED 事件触发）。"""
        self._pending_runtime.cancel(pending_alias)

    def finalize_frame(
        self,
        frame: ExecutionFrame,
        result: FrameExecutionResult,
    ) -> FrameProducts:
        """Project successful frame artifacts or clean up an unsuccessful frame."""
        frame_id = frame.runtime_scope.frame_id
        if result.status != FrameExecutionStatus.COMPLETED:
            self._pending_runtime.cancel_frame(frame_id)
            return FrameProducts()

        aliases = list(frame.harvested_aliases)
        for alias in self._pending_runtime.aliases_by_frame(frame_id):
            if alias and alias not in aliases:
                aliases.append(alias)

        from hivememory.core.mtp.models import MTPVerb

        for event in frame.progress.turn_events:
            if event.kind == "tool_call" and event.tool_kind == MTPVerb.UPDATE.value:
                if event.target and event.target not in aliases:
                    aliases.append(event.target)
        for alias in aliases:
            frame.add_harvested_alias(alias)
        return FrameProducts(artifact_aliases=tuple(aliases))

    def finalize_run(
        self,
        run_id: str,
        result: FrameExecutionResult,
    ) -> RuntimeProducts:
        """Finalize one root run without interpreting frame topology."""
        if result.status != FrameExecutionStatus.COMPLETED:
            self._pending_runtime.cancel_run(run_id)
            return RuntimeProducts()

        aliases = self._pending_runtime.pending_aliases_by_run(run_id)
        tasks = self._pending_runtime.claim_for_materialization(aliases)
        self._pending_runtime.evict_by_run(run_id)
        return RuntimeProducts(materialize_tasks=tuple(tasks))

    def health(self) -> dict[str, Any]:
        return {"loop_executor": "ok", "worker_agent": "ok"}


__all__ = ["AgentRuntime"]
