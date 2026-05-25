from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from hivememory.core.models import AgentProfile
from hivememory.core.protocol.models import ChatResult

from hivememory.alice.runtime.agent.loop_executor import KernelLoopExecutor
from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
from hivememory.alice.runtime.agent.worker_agent import WorkerAgentService
from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus
    from hivememory.alice.runtime.agent.mtp_executor import MTPExecutor
    from hivememory.alice.runtime.resolver import RuntimeAliasResolver
    from hivememory.prompts.assembler import AgentPromptAssembler
    from hivememory.system.config import HiveMemoryConfig

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Alice 执行态 runtime，持有循环、帧调度与生成引擎。"""

    def __init__(
        self,
        *,
        local_bus: "AliceBus",
        prompt_assembler: "AgentPromptAssembler",
        mtp_executor: "MTPExecutor",
        config: "HiveMemoryConfig",
        alias_resolver: "RuntimeAliasResolver",
    ) -> None:
        self._agent_profile_resolver = AgentProfileResolver(local_bus=local_bus)

        self._frame_scheduler = FrameScheduler(
            prompt_assembler=prompt_assembler,
        )
        self._worker_agent = WorkerAgentService(config=config.llm.worker)
        self._mtp_executor = mtp_executor
        self._loop_executor = KernelLoopExecutor(
            worker_agent=self._worker_agent,
            frame_scheduler=self._frame_scheduler,
            local_bus=local_bus,
            agent_profile_resolver=self._agent_profile_resolver,
            mtp_executor=self._mtp_executor,
            config=config.agent_runtime,
            alias_resolver=alias_resolver,
        )

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
        return await self._loop_executor.execute_main_frame(
            messages=messages,
            generation_options=generation_options,
            agent_profile=agent_profile,
            topic_id=topic_id,
            identity=identity,
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
        async for event in self._loop_executor.execute_main_frame_stream(
            messages=messages,
            generation_options=generation_options,
            agent_profile=agent_profile,
            topic_id=topic_id,
            identity=identity,
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
