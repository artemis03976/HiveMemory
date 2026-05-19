from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from hivememory.core.models import AgentProfile, MemoryAtom, OMNI_DOLL_PROFILE
from hivememory.core.protocol.models import ChatResult

from hivememory.alice.contracts.local_routes import AliceLocalRoutes
from hivememory.alice.runtime.agent_runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.cache import AgentProfileCache
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.system.config import HiveMemoryConfig

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)


class AliceRuntime:
    """Alice 子系统的 runtime 聚合根。"""

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: Optional["GlobalSystemBus"] = None,
    ) -> None:
        self._config = config
        self._global_bus = global_bus
        self._local_bus = AliceBus()
        self._local_routes_registered = False

        self._koakuma = KoakumaRuntime(
            bus=self._local_bus,
            config=config.koakuma,
        )
        self._agent_profile_cache = AgentProfileCache()
        self._agent_runtime = AgentRuntime(runtime=self)

        logger.info("AliceRuntime 初始化完成")

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    @property
    def global_bus(self) -> Optional["GlobalSystemBus"]:
        return self._global_bus

    @property
    def local_bus(self) -> AliceBus:
        return self._local_bus

    @property
    def local_routes_registered(self) -> bool:
        return self._local_routes_registered

    @property
    def koakuma(self) -> KoakumaRuntime:
        return self._koakuma

    @property
    def agent_runtime(self) -> AgentRuntime:
        return self._agent_runtime

    @property
    def frame_scheduler(self):
        return self._agent_runtime.frame_scheduler

    @property
    def loop_executor(self):
        return self._agent_runtime.loop_executor

    @property
    def worker_agent(self):
        return self._agent_runtime.worker_agent

    async def run_agent(
        self,
        messages: list[dict[str, str]],
        identity,
        agent_id: str,
        topic_id: str,
        generation_options: Optional[dict[str, Any]] = None,
        agent_profile=None,
        cancel_event=None,
    ) -> ChatResult:
        return await self._agent_runtime.run_agent(
            messages=messages,
            identity=identity,
            agent_id=agent_id,
            topic_id=topic_id,
            generation_options=generation_options,
            agent_profile=agent_profile,
            cancel_event=cancel_event,
        )

    async def run_agent_stream(
        self,
        messages: list[dict[str, str]],
        identity,
        agent_id: str,
        topic_id: str,
        generation_options: Optional[dict[str, Any]] = None,
        agent_profile=None,
        cancel_event=None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._agent_runtime.run_agent_stream(
            messages=messages,
            identity=identity,
            agent_id=agent_id,
            topic_id=topic_id,
            generation_options=generation_options,
            agent_profile=agent_profile,
            cancel_event=cancel_event,
        ):
            yield event

    async def get_agent_profile(self, agent_alias: str) -> AgentProfile:
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        cached = self._agent_profile_cache.get(agent_alias)
        if cached is not None:
            return cached

        if self._global_bus is not None:
            try:
                profile = await self._global_bus.request(
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

        logger.info(f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE.")
        return OMNI_DOLL_PROFILE

    def get_mtp_prompt(self, profile: Optional[AgentProfile] = None) -> str:
        if not self._config.koakuma.enabled:
            return ""

        prompt_config = self._config.koakuma.mtp_prompt
        if not prompt_config.enabled:
            return ""

        from hivememory.prompts.mtp import MTPPromptBuilder

        allowed_verbs = None
        allowed_runtime_tools = None
        if profile and profile.allowed_mtp_verbs is not None:
            allowed_verbs = profile.allowed_mtp_verbs
        if profile and profile.allowed_sys_tools is not None:
            allowed_runtime_tools = profile.allowed_sys_tools

        builder = MTPPromptBuilder(
            language=prompt_config.language,
            include_demo=prompt_config.include_demo,
            include_error_handling=prompt_config.include_error_handling,
            allowed_verbs=allowed_verbs,
            allowed_runtime_tools=allowed_runtime_tools,
        )
        return builder.build()

    def register_preretrieval_aliases(self, memories: list[MemoryAtom]) -> None:
        self._koakuma.atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug(
                f"预检索记忆缓存完成: {len(memories)} 条记忆已缓存到 Koakuma"
            )

    def export_interaction_state(self) -> dict[str, Any]:
        return {
            "mtp_traces": self._koakuma.get_interaction_traces(),
            "write_focus": self._koakuma.get_write_focus(),
            "update_focus": self._koakuma.get_update_focus(),
        }

    def mount_local_routes(self) -> None:
        if self._local_routes_registered:
            return

        self._local_bus.register(
            AliceLocalRoutes.RUN_AGENT,
            self.run_agent,
        )
        self._local_bus.register(
            AliceLocalRoutes.RUN_AGENT_STREAM,
            self._run_agent_stream_route,
        )
        self._local_bus.register(
            AliceLocalRoutes.REGISTER_PRERETRIEVAL_ALIASES,
            self._register_preretrieval_aliases_route,
        )
        self._local_bus.register(
            AliceLocalRoutes.GET_INTERACTION_STATE,
            self._get_interaction_state_route,
        )
        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        if not self._local_routes_registered:
            return

        for route in AliceLocalRoutes.ALL:
            self._local_bus.unregister(route)
        self._local_routes_registered = False

    async def _run_agent_stream_route(self, *args: Any, **kwargs: Any) -> Any:
        return self.run_agent_stream(*args, **kwargs)

    async def _register_preretrieval_aliases_route(self, *args: Any, **kwargs: Any) -> None:
        self.register_preretrieval_aliases(*args, **kwargs)

    async def _get_interaction_state_route(self) -> dict[str, Any]:
        return self.export_interaction_state()

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


__all__ = ["AliceRuntime"]
