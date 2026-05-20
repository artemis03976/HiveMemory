from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional

from hivememory.core.models import MemoryAtom
from hivememory.core.protocol.models import AgentRunContext, ChatResult

from hivememory.alice.contracts.local_routes import AliceLocalRoutes
from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.alice.runtime.agent.mtp_executor import KoakumaMTPExecutor
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import HiveMemoryConfig

logger = logging.getLogger(__name__)


class AliceRuntime:
    """Alice 子系统的 runtime 聚合根。"""

    def __init__(
        self,
        config: HiveMemoryConfig,
    ) -> None:
        self._config = config
        self._local_bus = AliceBus()
        self._local_routes_registered = False

        self._koakuma = KoakumaRuntime(
            bus=self._local_bus,
            config=config.koakuma,
        )
        self._prompt_assembler = AgentPromptAssembler(config.koakuma)
        self._mtp_executor = KoakumaMTPExecutor(self._koakuma)
        self._agent_runtime = AgentRuntime(
            local_bus=self._local_bus,
            prompt_assembler=self._prompt_assembler,
            mtp_executor=self._mtp_executor,
            config=config,
        )

        logger.info("AliceRuntime 初始化完成")

    def register_preretrieval_aliases(self, memories: list[MemoryAtom]) -> None:
        self._koakuma.atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug(
                f"预检索记忆缓存完成: {len(memories)} 条记忆已缓存到 Koakuma"
            )

    def mount_local_routes(self) -> None:
        if self._local_routes_registered:
            return

        self._local_bus.register(
            AliceLocalRoutes.RUN_AGENT,
            self.run_agent,
        )
        self._local_bus.register(
            AliceLocalRoutes.RUN_AGENT_STREAM,
            self.run_agent_stream,
        )
        self._local_bus.register(
            AliceLocalRoutes.REGISTER_PRERETRIEVAL_ALIASES,
            self.register_preretrieval_aliases,
        )
        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        if not self._local_routes_registered:
            return

        for route in AliceLocalRoutes.ALL:
            self._local_bus.unregister(route)
        self._local_routes_registered = False

    def health(self) -> dict[str, Any]:
        return {
            "local_routes_registered": self._local_routes_registered,
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {
                "status": "ok",
            },
            "profile_cache": {
                "status": "ok",
            },
        }

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    @property
    def local_bus(self) -> AliceBus:
        return self._local_bus

    @property
    def local_routes_registered(self) -> bool:
        return self._local_routes_registered

    async def run_agent(
        self,
        agent_run_context: AgentRunContext,
        generation_options: Optional[dict[str, Any]] = None,
        cancel_event=None,
    ) -> ChatResult:
        messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
        return await self._agent_runtime.run_agent(
            messages=messages,
            identity=agent_run_context.identity,
            topic_id=agent_run_context.topic_id,
            generation_options=generation_options,
            agent_profile=agent_run_context.agent_profile,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        agent_run_context: AgentRunContext,
        generation_options: Optional[dict[str, Any]] = None,
        cancel_event=None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
        async for event in self._agent_runtime.run_agent_stream(
            messages=messages,
            identity=agent_run_context.identity,
            topic_id=agent_run_context.topic_id,
            generation_options=generation_options,
            agent_profile=agent_run_context.agent_profile,
            cancel_event=cancel_event,
        ):
            yield event


__all__ = ["AliceRuntime"]
