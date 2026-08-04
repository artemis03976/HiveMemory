"""Alice 对外 Agent run 用例。

AgentRunService 是 Alice 的公开 run 用例入口：创建 root frame、为每次 run
构造 run-local RunExecutor、组装 ``AgentRunResult``，并在流式终态后发出
唯一 done（见 docs/alice/orchestration.md §1）。queue / runner task /
stream sequence 与 RuntimeEvent envelope 实现均不放在 application 层。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from hivememory.agent_runtime.aliases import KoakumaAtomCache
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.orchestration.call_coordinator import CallCoordinator
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_executor import RunExecutor
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter, BoundAgentRunEvents
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, Identity, MemoryAtom
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AgentRunStatus,
)
from hivememory.prompts.assembler import AgentPromptAssembler

logger = logging.getLogger(__name__)


class StreamExitReason(str, Enum):
    """流式 run 的结束原因，决定终态事件发布与收尾路径。"""

    RUNNING = "running"
    TERMINAL = "terminal"
    FAILED = "failed"
    CLOSED = "closed"
    MISSING_DONE = "missing_done"


class AgentRunService:
    """Alice 对外 Agent run 用例的唯一入口。"""

    def __init__(
        self,
        *,
        agent_runtime: AgentRuntime,
        call_coordinator: CallCoordinator,
        frame_factory: FrameFactory,
        prompt_assembler: AgentPromptAssembler,
        atom_cache: KoakumaAtomCache,
        stream_adapter: AgentRunStreamAdapter,
        agent_run_events: AgentRunEventEmitter,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_coordinator = call_coordinator
        self._frame_factory = frame_factory
        self._prompt_assembler = prompt_assembler
        self._atom_cache = atom_cache
        self._stream_adapter = stream_adapter
        self._agent_run_events = agent_run_events

    async def run_agent(
        self,
        agent_run_context: AgentRunContext,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        generation_id: str | None = None,
    ) -> AgentRunResult:
        session = self._create_run_session(
            generation_id=generation_id,
            cancel_event=cancel_event,
        )

        run_events = self._events_for_run(session, agent_run_context)
        run_events.started()

        try:
            self._register_preretrieval_aliases(agent_run_context.retrieval_result.memories)
            messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
            frame = self._create_root_frame(
                messages=messages,
                identity=agent_run_context.identity,
                topic_id=agent_run_context.topic_id,
                session=session,
                agent_profile=agent_run_context.agent_profile,
            )
            executor = RunExecutor(
                agent_runtime=self._agent_runtime,
                session=session,
                call_coordinator=self._call_coordinator,
            )
            engine_result = await executor.run(
                frame,
                generation_options=generation_options,
            )
            result = self._assemble_agent_run_result(
                frame,
                engine_result,
                executor.runtime_products or RuntimeProducts(),
            )
            self._publish_terminal(run_events, result)
            return result
        except Exception:
            run_events.failed(
                message="Agent run failed.",
            )
            raise

    async def run_agent_stream(
        self,
        agent_run_context: AgentRunContext,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        generation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        session = self._create_run_session(
            generation_id=generation_id,
            cancel_event=cancel_event,
        )

        run_events = self._events_for_run(session, agent_run_context)
        run_events.started()

        exit_reason = StreamExitReason.RUNNING
        executor_stream: AsyncGenerator[dict[str, Any], None] | None = None

        try:
            self._register_preretrieval_aliases(agent_run_context.retrieval_result.memories)
            messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
            agent_stream = self._stream_adapter.create(session)
            frame = self._create_root_frame(
                messages=messages,
                identity=agent_run_context.identity,
                topic_id=agent_run_context.topic_id,
                session=session,
                agent_profile=agent_run_context.agent_profile,
            )
            executor = RunExecutor(
                agent_runtime=self._agent_runtime,
                session=session,
                call_coordinator=self._call_coordinator,
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

            terminal_result = executor.terminal_result
            if terminal_result is None:
                exit_reason = StreamExitReason.MISSING_DONE
                run_events.failed(
                    message="Agent stream ended without done event.",
                )
                raise RuntimeError("Agent stream ended without done event")
            result = self._assemble_agent_run_result(
                frame,
                terminal_result,
                executor.runtime_products or RuntimeProducts(),
            )
            self._publish_terminal(run_events, result)
            exit_reason = StreamExitReason.TERMINAL
            yield {
                "event": "done",
                "data": {
                    **result.model_dump(),
                    **event_metadata,
                    "stream_sequence": agent_stream.next_sequence,
                },
            }
        except Exception:
            if exit_reason not in (
                StreamExitReason.TERMINAL,
                StreamExitReason.MISSING_DONE,
            ):
                exit_reason = StreamExitReason.FAILED
                run_events.failed(
                    message="Agent stream run failed.",
                )
            raise
        finally:
            if exit_reason == StreamExitReason.RUNNING:
                session.cancel_event.set()
                run_events.cancelled(
                    message="Agent stream closed before terminal event.",
                    close_reason="stream_closed",
                )
            if executor_stream is not None:
                await executor_stream.aclose()

    def _register_preretrieval_aliases(self, memories: list[MemoryAtom]) -> None:
        self._atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug("预检索记忆缓存完成: %s 条记忆已缓存到 Koakuma", len(memories))

    def _create_root_frame(
        self,
        *,
        messages: list[dict[str, str]],
        identity: Identity,
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
                runtime_scope=self._frame_factory.scope(run_id=session.agent_run_id),
                profile=profile,
                identity=identity,
                messages=messages,
                topic_id=topic_id or "",
                execution_policy=policy,
            )
        )
        session.register_root_frame(frame)
        return frame

    @staticmethod
    def _assemble_agent_run_result(
        frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        runtime_products: RuntimeProducts,
    ) -> AgentRunResult:
        """把执行层终态与产品投影为稳定的 Alice 公共结果。"""
        if engine_result.status == FrameExecutionStatus.CANCELLED:
            run_status = AgentRunStatus.CANCELLED
        elif engine_result.status == FrameExecutionStatus.COMPLETED:
            run_status = AgentRunStatus.COMPLETED
        else:
            run_status = AgentRunStatus.FAILED
        progress = frame.progress
        return AgentRunResult(
            status=run_status,
            final_text="".join(progress.text_segments),
            mtp_iterations=max(0, progress.iteration - 1),
            total_iterations=progress.iteration,
            turn_events=progress.turn_events,
            materialize_tasks=list(runtime_products.materialize_tasks),
            model_used=progress.model_used,
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
        generation_id: str | None,
        cancel_event: asyncio.Event | None,
    ) -> RunSession:
        return RunSession(
            agent_run_id=f"agent_run_{uuid.uuid4().hex}",
            generation_id=generation_id,
            cancel_event=cancel_event if cancel_event is not None else asyncio.Event(),
        )

    @staticmethod
    def _publish_terminal(
        run_events: BoundAgentRunEvents,
        result: AgentRunResult,
    ) -> None:
        if result.status == AgentRunStatus.CANCELLED.value:
            run_events.cancelled(result)
        elif result.status == AgentRunStatus.FAILED.value:
            run_events.failed(result)
        else:
            run_events.completed(result)

    def _events_for_run(
        self,
        session: RunSession,
        agent_run_context: AgentRunContext,
    ) -> BoundAgentRunEvents:
        return self._agent_run_events.for_run(
            agent_run_id=session.agent_run_id,
            generation_id=session.generation_id,
            topic_id=agent_run_context.topic_id,
            agent_id=agent_run_context.identity.agent_id,
        )


__all__ = ["AgentRunService"]
