"""Alice 对外 Agent run 用例。

AgentRunService 是 Alice 的公开 run 用例入口：接收任务进程组装的 CPU
输入清单，在内部转换为提示词组装使用的 ``AgentRunContext``；创建 root
frame、为每次 run 构造 run-local RunExecutor，并产出 CPU 中立的执行结果
（见 docs/alice/orchestration.md §1）。

统一入口 :meth:`AgentRunService.run_agent` 与 ``run_process`` 一样以
``stream`` 参数控制是否流式，内部只有一套执行骨架（会话、``agent.run.*``
事件、提示词组装、root frame 与 RunExecutor）：
``stream=True`` 返回交互事件的异步生成器（最后一项是 ``done``），
``stream=False`` 返回可 await 的 ``CPUExecutionResult``。
queue / runner task / stream sequence 与 RuntimeEvent envelope 实现均不
放在 application 层。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast, overload

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_executor import RunExecutor
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent.call_coordinator import CallCoordinator
from hivememory.alice.runtime.runtime_events import (
    AgentRunEventEmitter,
    AgentRunStats,
    BoundAgentRunEvents,
)
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IdentityScope,
)
from hivememory.core.protocol.models import (
    AgentRunContext,
    RetrievalResponse,
)
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUInputManifest,
    ProcessOperations,
)

logger = logging.getLogger(__name__)


class StreamExitReason(str, Enum):
    """流式 run 的结束原因，决定终态事件发布与收尾路径。"""

    RUNNING = "running"
    TERMINAL = "terminal"
    FAILED = "failed"
    CLOSED = "closed"
    MISSING_DONE = "missing_done"


def _agent_run_context_from_manifest(manifest: CPUInputManifest) -> AgentRunContext:
    """把 CPU 输入清单转换为提示词组装使用的内部运行上下文。

    ``interaction_id`` 取 ``process_id``（进程是本次 Interaction 的关联 ID
    事实来源）；``retrieval_result`` 由未编译的记忆原子构造，供预检索
    alias 登记等流程使用。``AgentRunContext`` 不出现在任何公开路由上。
    """
    return AgentRunContext(
        identity_scope=manifest.identity_scope,
        interaction_id=manifest.process_id,
        topic_id=manifest.topic_id,
        user_message=manifest.user_message,
        topic_context=manifest.topic_context,
        retrieval_result=RetrievalResponse.from_memories(manifest.memories),
        memory_context=manifest.memory_context,
        agent_profile=manifest.agent_profile,
        storage_available=manifest.storage_available,
        attachment_context=manifest.attachment_context,
    )


@dataclass(frozen=True, slots=True)
class _RunPreparation:
    """一次 run 的共享前置产物：内部上下文、会话与已 started 的事件绑定。"""

    context: AgentRunContext
    session: RunSession
    events: BoundAgentRunEvents


class AgentRunService:
    """Alice 对外 Agent run 用例的唯一入口。"""

    def __init__(
        self,
        *,
        agent_runtime: AgentRuntime,
        call_coordinator: CallCoordinator,
        frame_factory: FrameFactory,
        prompt_assembler: AgentPromptAssembler,
        stream_adapter: AgentRunStreamAdapter,
        agent_run_events: AgentRunEventEmitter,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_coordinator = call_coordinator
        self._frame_factory = frame_factory
        self._prompt_assembler = prompt_assembler
        self._stream_adapter = stream_adapter
        self._agent_run_events = agent_run_events

    # ========== 统一入口 ==========

    @overload
    def run_agent(
        self,
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None = None,
        *,
        operations: ProcessOperations,
        stream: Literal[True] = True,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    @overload
    def run_agent(
        self,
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None = None,
        *,
        operations: ProcessOperations,
        stream: Literal[False],
    ) -> Coroutine[Any, Any, CPUExecutionResult]: ...

    def run_agent(
        self,
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None = None,
        *,
        operations: ProcessOperations,
        stream: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None] | Coroutine[Any, Any, CPUExecutionResult]:
        """Alice Agent run 的统一入口（``stream`` 控制是否流式）。

        两种模式运行同一套执行骨架，终态事件语义一致。``stream=True``
        返回交互事件的异步生成器，最后一项是 ``done``（含运行元数据）；
        ``stream=False`` 返回可 await 的 ``CPUExecutionResult``。
        """
        producer = self._run_agent(
            input_manifest, generation_options, operations=operations, stream=stream
        )
        if stream:
            # 流式骨架只产出交互事件与 done；骨架的产出类型是两种形态的并集。
            return cast(AsyncGenerator[dict[str, Any], None], producer)
        return self._collect_execution_result(producer)

    async def _run_agent(
        self,
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None,
        *,
        operations: ProcessOperations,
        stream: bool,
    ) -> AsyncGenerator[dict[str, Any] | CPUExecutionResult, None]:
        """统一执行骨架：会话、事件、组装、执行与终态发布只有一份。

        ``stream`` 只决定执行是否接入 :class:`AgentRunStreamAdapter` 并产出
        交互事件：流式逐条产出事件后以 ``done`` 收尾；非流式直接 await
        RunExecutor 并产出唯一执行结果。
        """
        preparation = self._prepare_run(input_manifest)
        run_events = preparation.events

        exit_reason = StreamExitReason.RUNNING
        executor_stream: AsyncGenerator[dict[str, Any], None] | None = None

        try:
            executor = RunExecutor(
                agent_runtime=self._agent_runtime,
                session=preparation.session,
                call_coordinator=self._call_coordinator,
            )
            messages = self._prompt_assembler.build_main_agent_messages(preparation.context)
            if stream:
                agent_stream = self._stream_adapter.create(preparation.session)
                frame = self._create_root_frame(
                    messages=messages,
                    operations=operations,
                    identity_scope=preparation.context.identity_scope,
                    topic_id=preparation.context.topic_id,
                    session=preparation.session,
                    agent_profile=preparation.context.agent_profile,
                )
                event_metadata = self._event_metadata_for_frame(frame)
                executor_stream = agent_stream.events(
                    executor.run(
                        frame,
                        generation_options=generation_options,
                        run_output=agent_stream.output,
                    )
                )
                async for event in executor_stream:
                    yield event

                engine_result = executor.terminal_result
                if engine_result is None:
                    exit_reason = StreamExitReason.MISSING_DONE
                    run_events.failed(message="Agent stream ended without done event.")
                    raise RuntimeError("Agent stream ended without done event")
            else:
                frame = self._create_root_frame(
                    messages=messages,
                    operations=operations,
                    identity_scope=preparation.context.identity_scope,
                    topic_id=preparation.context.topic_id,
                    session=preparation.session,
                    agent_profile=preparation.context.agent_profile,
                )
                engine_result = await executor.run(frame, generation_options=generation_options)

            result = self._assemble_execution_result(frame, engine_result)
            self._publish_terminal(run_events, result, self._stats_for(frame))
            exit_reason = StreamExitReason.TERMINAL
            if stream:
                yield {
                    "event": "done",
                    "data": {
                        **result.model_dump(),
                        **event_metadata,
                        "stream_sequence": agent_stream.next_sequence,
                    },
                }
            else:
                yield result
        except Exception:
            if exit_reason not in (
                StreamExitReason.TERMINAL,
                StreamExitReason.MISSING_DONE,
            ):
                exit_reason = StreamExitReason.FAILED
                run_events.failed(message="Agent run failed.")
            raise
        finally:
            if exit_reason == StreamExitReason.RUNNING:
                # GeneratorExit（交付方断流）与 CancelledError（运行被取消）
                # 都按取消观测收口；流式附带 close_reason，非流式没有事件流。
                run_events.cancelled(
                    message=(
                        "Agent stream closed before terminal event."
                        if stream
                        else "Agent run cancelled before terminal event."
                    ),
                    close_reason="stream_closed" if stream else None,
                )
            if executor_stream is not None:
                try:
                    await executor_stream.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("关闭 Agent executor stream 失败", exc_info=True)

    async def _collect_execution_result(
        self,
        producer: AsyncGenerator[dict[str, Any] | CPUExecutionResult, None],
    ) -> CPUExecutionResult:
        """非流式交付：骨架只产出唯一的执行结果，直接取该项。"""
        result: CPUExecutionResult | None = None
        async for item in producer:
            if isinstance(item, CPUExecutionResult):
                result = item
        if result is None:
            raise RuntimeError("Agent run ended without execution result")
        return result

    # ========== 执行骨架的共享步骤 ==========

    def _prepare_run(self, input_manifest: CPUInputManifest) -> _RunPreparation:
        """构造 run 上下文与会话，绑定 ``agent.run.*`` 事件并发布 started。"""
        context = _agent_run_context_from_manifest(input_manifest)
        session = self._create_run_session(process_id=input_manifest.process_id)
        run_events = self._events_for_run(session, context)
        run_events.started()
        return _RunPreparation(context=context, session=session, events=run_events)

    def _create_root_frame(
        self,
        *,
        messages: list[dict[str, str]],
        operations: ProcessOperations,
        identity_scope: IdentityScope,
        topic_id: str,
        agent_profile: AgentProfile | None,
        session: RunSession,
    ) -> ExecutionFrame:
        """为当前 run 创建并登记唯一 root frame。"""
        profile = agent_profile or OMNI_DOLL_PROFILE
        policy = FrameExecutionPolicy.from_profile(
            profile,
            max_iterations=getattr(self._agent_runtime, "max_iterations", None),
        )
        frame = self._frame_factory.create(
            FrameSpec(
                runtime_scope=self._frame_factory.scope(
                    identity_scope=identity_scope,
                    run_id=session.agent_run_id,
                ),
                profile=profile,
                messages=messages,
                topic_id=topic_id or "",
                execution_policy=policy,
                operations=operations,
            )
        )
        session.register_root_frame(frame)
        return frame

    @staticmethod
    def _assemble_execution_result(
        frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
    ) -> CPUExecutionResult:
        """把执行层终态与轮次事件投影为 CPU 中立的执行结果。"""
        if engine_result.status == FrameExecutionStatus.CANCELLED:
            run_status = CPUExecutionStatus.CANCELLED
        elif engine_result.status == FrameExecutionStatus.COMPLETED:
            run_status = CPUExecutionStatus.COMPLETED
        else:
            run_status = CPUExecutionStatus.FAILED
        progress = frame.progress
        return CPUExecutionResult(
            status=run_status,
            final_text="".join(progress.text_segments),
            turn_events=progress.turn_events,
            model_used=progress.model_used,
        )

    @staticmethod
    def _stats_for(frame: ExecutionFrame) -> AgentRunStats:
        """从 frame 进度取得 ``agent.run.*`` 终态事件的观测统计。"""
        progress = frame.progress
        return AgentRunStats(
            mtp_iterations=max(0, progress.iteration - 1),
            total_iterations=progress.iteration,
        )

    @staticmethod
    def _event_metadata_for_frame(frame: ExecutionFrame) -> dict[str, Any]:
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "agent_run_id": frame.runtime_scope.run_id,
            "action_id": None,
            "scope": "main",
            "depth": 0,
            "agent_id": agent_id,
            "frame_id": frame.runtime_scope.frame_id,
        }

    @staticmethod
    def _create_run_session(
        *,
        process_id: str | None,
    ) -> RunSession:
        return RunSession(
            agent_run_id=f"agent_run_{uuid.uuid4().hex}",
            process_id=process_id,
        )

    @staticmethod
    def _publish_terminal(
        run_events: BoundAgentRunEvents,
        result: CPUExecutionResult,
        stats: AgentRunStats,
    ) -> None:
        if result.status == CPUExecutionStatus.CANCELLED.value:
            run_events.cancelled(stats)
        elif result.status == CPUExecutionStatus.FAILED.value:
            run_events.failed(stats)
        else:
            run_events.completed(stats)

    def _events_for_run(
        self,
        session: RunSession,
        agent_run_context: AgentRunContext,
    ) -> BoundAgentRunEvents:
        return self._agent_run_events.for_run(
            agent_run_id=session.agent_run_id,
            process_id=session.process_id,
            topic_id=agent_run_context.topic_id,
            agent_id=agent_run_context.identity_scope.actor_identity.agent_id,
            workspace_id=agent_run_context.identity_scope.workspace_identity.workspace_id,
        )


__all__ = ["AgentRunService"]
