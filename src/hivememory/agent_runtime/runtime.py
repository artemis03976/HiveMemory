from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus, FrameProducts
from hivememory.agent_runtime.loop import AgentLoopExecutor
from hivememory.agent_runtime.model.client import LLMClient
from hivememory.agent_runtime.model.resolution import ResolvedFrameModel, resolve_frame_model
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.output import FrameOutputSink
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models import AgentProfile, TurnEvent
from hivememory.core.mtp import MTPCallResponse, MTPFormatter

if TYPE_CHECKING:
    from hivememory.agent_runtime.frame import ExecutionFrame
    from hivememory.agent_runtime.model.resolution import ModelResolver
    from hivememory.agent_runtime.mtp.runtime import MTPExecutor
    from hivememory.config.alice import AgentRuntimeConfig, KoakumaConfig
    from hivememory.config.memory_compiler import MemoryCompilerConfig

logger = logging.getLogger(__name__)


class AgentRuntime:
    """单 Agent 运行时门面。

    封装执行引擎（loop_executor + llm_client + mtp_executor）与
    对外提供 "跑一个 frame" 与 "收割成功子帧产物" 的 API。
    它把底层引擎组件收拢成一个聚合，编排层（RunExecutor）
    只拿这个门面操作，不直接接触引擎细节。

    帧策略未指定迭代上限时，使用运行时配置中的 max_loop_iterations。
    """

    def __init__(
        self,
        *,
        mtp_executor: MTPExecutor,
        runtime_config: AgentRuntimeConfig,
        loop_executor: AgentLoopExecutor | None = None,
        model_registry: ModelResolver | None = None,
    ) -> None:
        # 模型注册表：在每个 frame 开始时根据 agent_profile.model_name 解析实际模型。
        # 若未提供，generation_options 中必须已包含 model，否则 LLMClient 会报错。
        self._model_registry = model_registry
        self._config = runtime_config
        if loop_executor is not None:
            self._loop_executor = loop_executor
        else:
            # LLMClient 是无状态服务，不依赖任何实例级 LLM 配置。
            # 所有 LLM 参数均在运行时由 resolve_model 注入。
            llm_client = LLMClient()
            self._loop_executor = AgentLoopExecutor(
                llm_client=llm_client,
                mtp_executor=mtp_executor,
                config=runtime_config,
            )

    @property
    def max_iterations(self) -> int:
        """帧未覆盖时采用的统一循环上限。"""
        return self._config.max_loop_iterations

    async def run_frame(
        self,
        frame: ExecutionFrame,
        *,
        generation_options: dict[str, Any] | None = None,
        output_sink: FrameOutputSink,
    ) -> FrameExecutionResult:
        """按输出端口生成，推进当前帧直到自然收敛、CALL 挂起或失败。"""
        try:
            if frame.resolved_model is None:
                frame.resolved_model = self.resolve_model(frame.agent_profile, generation_options)
            frame.progress.model_used = frame.resolved_model.display_name
            generation_options = frame.resolved_model.generation_options
        except ModelNotFoundError as error:
            logger.error("Frame model resolution failed: %s", error)
            return FrameExecutionResult(
                status=FrameExecutionStatus.FAILED,
                error=error,
            )
        max_iterations = frame.execution_policy.max_iterations or self.max_iterations
        return await self._loop_executor.execute_frame(
            frame=frame,
            max_iterations=max_iterations,
            generation_options=generation_options,
            output_sink=output_sink,
        )

    def apply_call_response(
        self,
        frame: ExecutionFrame,
        suspension: FrameExecutionResult,
        response: MTPCallResponse,
    ) -> None:
        """把一次已结算的 CALL response 恰好一次地回填到挂起的 caller frame。"""
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

    def resolve_model(
        self,
        profile: AgentProfile,
        generation_options: dict[str, Any] | None,
    ) -> ResolvedFrameModel:
        """解析帧模型与输入预算；解析结果随帧保留，恢复时不重新读取注册表。"""
        return resolve_frame_model(
            profile, generation_options, model_registry=self._model_registry, config=self._config
        )

    def finalize_frame(
        self,
        frame: ExecutionFrame,
        result: FrameExecutionResult,
    ) -> FrameProducts:
        """只回填成功子帧收到 ACK 的 alias；失败子帧的写入保持已提交。"""
        if result.status != FrameExecutionStatus.COMPLETED:
            return FrameProducts()
        return FrameProducts(artifact_aliases=tuple(frame.harvested_aliases))

    def health(self) -> dict[str, Any]:
        """返回运行时的静态健康状态（依赖的活跃性由业务层探测）。"""
        return {
            "agent_runtime": {"loop_executor": "ok", "worker_agent": "ok"},
            "koakuma_runtime": {"status": "ok"},
        }


def build_agent_runtime(
    *,
    runtime_config: AgentRuntimeConfig,
    koakuma_config: KoakumaConfig,
    memory_compiler_config: MemoryCompilerConfig,
    model_registry: ModelResolver | None = None,
) -> AgentRuntime:
    """装配单帧执行内核，编排层只取得运行时门面。

    Koakuma 与执行循环共享内核生命周期；Profile 与引用的派生缓存由
    workspace 持有，资源操作通过每帧的提交函数进入。
    """
    koakuma = KoakumaRuntime(
        config=koakuma_config,
        memory_compiler_config=memory_compiler_config,
    )
    runtime = AgentRuntime(
        mtp_executor=koakuma,
        runtime_config=runtime_config,
        model_registry=model_registry,
    )
    logger.info("AgentRuntime 初始化完成")
    return runtime


__all__ = ["AgentRuntime", "build_agent_runtime"]
