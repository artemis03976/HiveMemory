"""Alice public facade for run-local agent orchestration."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.alice.runtime.agent.call_coordinator import CallCoordinator
from hivememory.alice.runtime.agent.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.runtime.agent.run_driver import RunDriver
from hivememory.alice.runtime.agent.run_session import RunSession
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.protocol.models import AgentRunResult, AgentRunStatus

if TYPE_CHECKING:
    from hivememory.agent_runtime.resolver import RuntimeAliasResolver
    from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
    from hivememory.alice.runtime.agent.runtime import AgentRuntime
    from hivememory.core.models import AgentProfile, Identity
    from hivememory.prompts.assembler import AgentPromptAssembler

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Adapt Alice public inputs/outputs to the run-local driver."""

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        agent_profile_resolver: AgentProfileResolver,
        alias_resolver: RuntimeAliasResolver,
        frame_factory: FrameFactory,
        prompt_assembler: AgentPromptAssembler,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._agent_profile_resolver = agent_profile_resolver
        self._alias_resolver = alias_resolver
        self._frame_factory = frame_factory
        self._prompt_assembler = prompt_assembler
        self._call_coordinator = CallCoordinator(
            agent_runtime,
            agent_profile_resolver,
            alias_resolver,
            frame_factory=frame_factory,
            prompt_assembler=prompt_assembler,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def run_agent(
        self,
        messages: list[dict[str, str]],
        identity: Identity,
        topic_id: str,
        *,
        session: RunSession,
        generation_options: dict[str, Any] | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> AgentRunResult:
        frame = self._create_root_frame(
            messages=messages,
            identity=identity,
            topic_id=topic_id,
            agent_profile=agent_profile,
            session=session,
        )
        driver = RunDriver(
            agent_runtime=self._agent_runtime,
            session=session,
            call_coordinator=self._call_coordinator,
        )
        result = await driver.run(
            frame,
            generation_options=generation_options,
        )

        return self._assemble_agent_run_result(
            frame,
            result,
            driver.runtime_products or RuntimeProducts(),
        )

    async def run_agent_stream(
        self,
        messages: list[dict[str, str]],
        identity: Identity,
        topic_id: str,
        *,
        session: RunSession,
        generation_options: dict[str, Any] | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        frame = self._create_root_frame(
            messages=messages,
            identity=identity,
            topic_id=topic_id,
            agent_profile=agent_profile,
            session=session,
        )
        driver = RunDriver(
            agent_runtime=self._agent_runtime,
            session=session,
            call_coordinator=self._call_coordinator,
        )
        event_metadata = self._event_metadata_for_frame(frame)

        async for event in driver.run_stream(
            frame,
            generation_options=generation_options,
            event_metadata=event_metadata,
        ):
            yield event

        terminal_result = driver.terminal_result
        if terminal_result is None:
            raise RuntimeError("Run driver ended without a terminal result.")
        result = self._assemble_agent_run_result(
            frame,
            terminal_result,
            driver.runtime_products or RuntimeProducts(),
        )

        yield {
            "event": "done",
            "data": {
                **result.model_dump(),
                **event_metadata,
                "stream_sequence": driver.next_stream_sequence,
            },
        }

    def _create_root_frame(
        self,
        *,
        messages: list[dict[str, str]],
        identity: Identity,
        topic_id: str,
        agent_profile: AgentProfile | None,
        session: RunSession,
    ) -> ExecutionFrame:
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


__all__ = ["AgentOrchestrator"]
