from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from hivememory.core.models import AgentProfile, OMNI_DOLL_PROFILE
from hivememory.core.protocol.models import ChatResult

from hivememory.alice.runtime.cache import AgentProfileCache
from hivememory.alice.runtime.loop_executor import KernelLoopExecutor
from hivememory.alice.runtime.worker_agent import WorkerAgentService
from hivememory.alice.runtime.frame_scheduler import FrameScheduler
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.alice.runtime.core import AliceRuntime

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Alice 执行态 runtime，持有循环、帧调度与生成引擎。"""

    def __init__(self, runtime: "AliceRuntime") -> None:
        self._runtime = runtime

        self._agent_profile_cache = AgentProfileCache()
        
        self._frame_scheduler = FrameScheduler(
            runtime=runtime,
            prompt_assembler=runtime.prompt_assembler,
        )
        self._worker_agent = WorkerAgentService(config=runtime.config.llm.worker)
        self._loop_executor = KernelLoopExecutor(
            runtime=runtime,
            worker_agent=self._worker_agent,
            config=runtime.config.agent_runtime,
        )

    @property
    def frame_scheduler(self) -> "FrameScheduler":
        return self._frame_scheduler

    @property
    def loop_executor(self) -> KernelLoopExecutor:
        return self._loop_executor

    @property
    def worker_agent(self) -> WorkerAgentService:
        return self._worker_agent

    async def get_agent_profile(self, agent_alias: str) -> AgentProfile:
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        cached = self._agent_profile_cache.get(agent_alias)
        if cached is not None:
            return cached

        global_bus = self._runtime.global_bus
        if global_bus is not None:
            try:
                profile = await global_bus.request(
                    PatchouliRoutes.GET_AGENT_PROFILE,
                    agent_alias,
                )
            except Exception as e:
                logger.warning(f"Failed to load agent profile '{agent_alias}' via bus: {e}")
                profile = None

            if profile is not None:
                logger.info(f"Agent profile '{agent_alias}' loaded and cached.")
                self._agent_profile_cache.store(
                    agent_alias,
                    None,
                    profile,
                )
                return profile

        logger.info(
            f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE."
        )
        return OMNI_DOLL_PROFILE

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
