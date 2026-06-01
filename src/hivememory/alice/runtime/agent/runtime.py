from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from hivememory.core.models import AgentProfile
from hivememory.core.protocol.models import ChatResult

from hivememory.alice.runtime.agent.loop_executor import KernelLoopExecutor
from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
from hivememory.alice.runtime.agent.worker_agent import WorkerAgentService
from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler
from hivememory.alice.runtime.orchestrator import AgentOrchestrator

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus
    from hivememory.alice.runtime.agent.mtp_executor import MTPExecutor
    from hivememory.alice.runtime.resolver import RuntimeAliasResolver
    from hivememory.prompts.assembler import AgentPromptAssembler
    from hivememory.system.config import HiveMemoryConfig

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Alice 执行态 runtime，持有编排驱动器与执行引擎。"""

    def __init__(
        self,
        *,
        local_bus: "AliceBus",
        prompt_assembler: "AgentPromptAssembler",
        mtp_executor: "MTPExecutor",
        config: "HiveMemoryConfig",
        alias_resolver: "RuntimeAliasResolver",
    ) -> None:
        agent_profile_resolver = AgentProfileResolver(local_bus=local_bus)
        frame_scheduler = FrameScheduler(prompt_assembler=prompt_assembler)
        worker_agent = WorkerAgentService(config=config.llm.worker)
        loop_executor = KernelLoopExecutor(
            worker_agent=worker_agent,
            mtp_executor=mtp_executor,
            config=config.agent_runtime,
        )
        self._orchestrator = AgentOrchestrator(
            loop_executor=loop_executor,
            frame_scheduler=frame_scheduler,
            agent_profile_resolver=agent_profile_resolver,
            alias_resolver=alias_resolver,
        )
        self._agent_profile_resolver = agent_profile_resolver

    async def get_agent_profile(self, agent_alias: str) -> AgentProfile:
        return await self._agent_profile_resolver.resolve(agent_alias)

    async def run_agent(
        self,
        messages: list[dict[str, str]],
        identity,
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        cancel_event=None,
    ) -> ChatResult:
        return await self._orchestrator.run_agent(
            messages=messages,
            identity=identity,
            topic_id=topic_id,
            generation_options=generation_options,
            agent_profile=agent_profile,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        messages: list[dict[str, str]],
        identity,
        topic_id: str,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        cancel_event=None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._orchestrator.run_agent_stream(
            messages=messages,
            identity=identity,
            topic_id=topic_id,
            generation_options=generation_options,
            agent_profile=agent_profile,
            cancel_event=cancel_event,
        ):
            yield event

    def health(self) -> dict[str, Any]:
        return {
            "loop_executor": "ok",
            "frame_scheduler": "ok",
            "worker_agent": "ok",
        }


__all__ = ["AgentRuntime"]
