from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from hivememory.agent_runtime.loop_executor import AgentLoopExecutor
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.worker_agent import WorkerAgentService
from hivememory.core.models.pending import PendingAtomStatus

if TYPE_CHECKING:
    from hivememory.agent_runtime.models import ExecutionFrame, FrameExecutionResult
    from hivememory.agent_runtime.mtp.mtp_executor import MTPExecutor
    from hivememory.core.models.pending import PendingAtomMaterializeTask
    from hivememory.system.config import AliceConfig, SharedConfig
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
        mtp_executor: "MTPExecutor",
        alice_config: "AliceConfig",
        shared_config: "SharedConfig",
        pending_runtime: Optional[PendingAtomRuntime] = None,
        loop_executor: Optional[AgentLoopExecutor] = None,
        model_registry: Optional["ModelRegistry"] = None,
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

    async def run_frame(
        self,
        frame: "ExecutionFrame",
        generation_options: Optional[Dict[str, Any]] = None,
        cancel_event=None,
    ) -> "FrameExecutionResult":
        """跑一个 frame 到自然收敛或命中 CALL（非流式）。"""
        generation_options = self._resolve_model_for_frame(frame, generation_options)
        return await self._loop_executor.execute_frame(
            frame=frame,
            max_iterations=self._max_iterations,
            generation_options=generation_options,
            cancel_event=cancel_event,
        )

    def run_frame_stream(
        self,
        frame: "ExecutionFrame",
        generation_options: Optional[Dict[str, Any]] = None,
        cancel_event=None,
        on_suspend: Optional[Callable[["FrameExecutionResult"], Awaitable[None]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """跑一个 frame 并逐 token 流式输出；命中 CALL 时回调 on_suspend。"""
        generation_options = self._resolve_model_for_frame(frame, generation_options)
        return self._loop_executor.execute_frame_stream(
            frame=frame,
            max_iterations=self._max_iterations,
            generation_options=generation_options,
            cancel_event=cancel_event,
            on_suspend=on_suspend,
        )

    async def run_frame_emitting(
        self,
        frame: "ExecutionFrame",
        generation_options: Optional[Dict[str, Any]] = None,
        stream_emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        cancel_event=None,
    ) -> "FrameExecutionResult":
        """跑一个 frame，逐 token 推给 stream_emitter（供编排跑流式子帧）。"""
        generation_options = self._resolve_model_for_frame(frame, generation_options)
        return await self._loop_executor.execute_frame(
            frame=frame,
            max_iterations=self._max_iterations,
            generation_options=generation_options,
            stream_emitter=stream_emitter,
            use_stream_generation=stream_emitter is not None,
            cancel_event=cancel_event,
        )

    def _resolve_model_for_frame(
        self,
        frame: "ExecutionFrame",
        generation_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        根据 agent_profile.model_name 从注册表解析 LLM 参数，
        并注入到 generation_options 中供 WorkerAgentService 使用。

        同时将解析出的模型展示名写入 frame.progress.model_used，
        供 AgentOrchestrator 在组装 AgentRunResult 时读取。

        设计原则：
        - 若注册表未注入（model_registry 为 None），由调用方通过 generation_options
          直接传 model 参数；WorkerAgentService 缺少 model 时会抛出明确错误。
        - 注册表解析失败时不捕获异常，让错误向上传播——静默降级到"某个"模型不是用户意图。
        """
        if self._model_registry is None:
            return generation_options or {}

        model_name = frame.agent_profile.model_name
        # 会话温度覆盖：generation_options 中若有 temperature 则透传
        temperature_override = (generation_options or {}).get("temperature")

        # 注册表解析失败时向上传播，不捕获——
        # 解析失败通常意味着 model_name 不存在于注册表，属于配置错误，应立即暴露。
        llm_config, display_name = self._model_registry.resolve(
            model_name,
            temperature_override=temperature_override,
        )

        # 记录展示名，供 Orchestrator 组装 AgentRunResult 时读取
        frame.progress.model_used = display_name

        # 将注册表解析结果注入 generation_options
        # api_key / api_base 为 None 是合法状态（litellm 从环境变量读取）
        resolved: Dict[str, Any] = {
            **(generation_options or {}),
            "model": llm_config.model,
            "api_key": llm_config.api_key,
            "api_base": llm_config.api_base,
        }
        logger.debug(
            f"模型解析: model_name={model_name!r} → "
            f"litellm_model={llm_config.model!r}, display={display_name!r}"
        )
        return resolved

    def mark_task_failed(self, pending_alias: str) -> None:
        """将 MATERIALIZING 的 atom 迁移到 FAILED（由 patchouli FAILED 事件触发）。"""
        self._pending_runtime.mark_failed(pending_alias)

    def mark_task_cancelled(self, pending_alias: str) -> None:
        """将 in-flight atom 迁移到 CANCELLED（由 patchouli CANCELLED 事件触发）。"""
        self._pending_runtime.cancel(pending_alias)

    def cancel_tasks_by_run(self, run_id: str) -> List[str]:
        """取消本 run 仍在飞行中的 pending atom。"""
        return self._pending_runtime.cancel_run(run_id)

    def aliases_by_frame(self, frame_id: str) -> List[str]:
        """返回属于指定 frame 的全部 pending alias（不做状态过滤，供 harvest 使用）。"""
        return [
            atom.pending_alias
            for atom in self._pending_runtime.all_atoms()
            if atom.runtime_scope.frame_id == frame_id
        ]

    def collect_tasks_by_run(self, run_id: str) -> "List[PendingAtomMaterializeTask]":
        """收集本 run 的待物化 Task，并将状态从 PENDING 迁移到 MATERIALIZING（幂等）。
        同时回收上轮已结算的 atom（SETTLED/FAILED/CANCELLED → EXPIRED → 删除）。
        """
        aliases = [
            atom.pending_alias
            for atom in self._pending_runtime.all_atoms()
            if atom.runtime_scope.run_id == run_id
            and atom.status == PendingAtomStatus.PENDING
        ]
        tasks = self._pending_runtime.claim_for_materialization(aliases)
        self._pending_runtime.evict_by_run(run_id)
        return tasks

    def health(self) -> dict[str, Any]:
        return {"loop_executor": "ok", "worker_agent": "ok"}


__all__ = ["AgentRuntime"]
