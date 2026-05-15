from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

from hivememory.patchouli.config import HiveMemoryConfig
from hivememory.patchouli.protocol.models import ChatResult
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.patchouli_subsystem import PatchouliSubsystemAdapter
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import (
    GlobalMaintenanceScheduler,
)


class HiveMemorySystem:
    """
    HiveMemory 顶层系统门面 (Phase A)

    薄门面 + 宿主容器。将所有业务逻辑委托给 Patchouli 子系统，
    同时建立多子系统架构的结构基础。
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        patchouli: PatchouliSystem,
        global_bus: GlobalSystemBus,
        scheduler: GlobalMaintenanceScheduler,
        patchouli_subsystem: PatchouliSubsystemAdapter,
        chat_service: ChatApplicationService,
        ingress_service: PassiveIngressService,
    ) -> None:
        self._config = config
        self._patchouli = patchouli
        self._global_bus = global_bus
        self._scheduler = scheduler
        self._patchouli_subsystem = patchouli_subsystem
        self._chat_service = chat_service
        self._ingress_service = ingress_service
        self._started = False
        self._scheduler_stopped = False

    @classmethod
    def build(
        cls,
        config: Optional[HiveMemoryConfig] = None,
    ) -> "HiveMemorySystem":
        from hivememory.patchouli.config import load_app_config
        from hivememory.patchouli.runtime.bridge import PatchouliBridge
        from hivememory.patchouli.runtime.bus import PatchouliBus

        config = config or load_app_config()

        global_bus = GlobalSystemBus()
        scheduler = GlobalMaintenanceScheduler(
            tick_seconds=config.scheduler.tick_seconds,
            shutdown_wait_seconds=config.scheduler.shutdown_wait_seconds,
        )

        patchouli = PatchouliSystem(config=config)
        patchouli_bus = PatchouliBus()
        patchouli_bridge = PatchouliBridge(
            local_bus=patchouli_bus,
            global_bus=global_bus,
        )

        patchouli_subsystem = PatchouliSubsystemAdapter(
            patchouli=patchouli,
            local_bus=patchouli_bus,
            bridge=patchouli_bridge,
            scheduler=scheduler,
        )

        chat_service = ChatApplicationService(patchouli=patchouli)
        ingress_service = PassiveIngressService(
            bus=global_bus,
            config=config,
            scheduler=scheduler,
        )

        return cls(
            config=config,
            patchouli=patchouli,
            global_bus=global_bus,
            scheduler=scheduler,
            patchouli_subsystem=patchouli_subsystem,
            chat_service=chat_service,
            ingress_service=ingress_service,
        )

    # ========== 生命周期 ==========

    async def start(self) -> None:
        if self._started:
            return
        await self._patchouli_subsystem.start()
        self._scheduler.start()
        self._started = True
        self._scheduler_stopped = False
        await self._ingress_service.start()

    async def stop(self) -> None:
        await self._stop_scheduler()
        await self._ingress_service.shutdown_drain()
        if not self._started:
            return
        await self._patchouli_subsystem.stop()
        self._started = False
        self._scheduler_stopped = False

    async def _stop_scheduler(self) -> None:
        if not self._started or self._scheduler_stopped:
            return
        await self._scheduler.stop()
        self._scheduler_stopped = True

    async def health(self) -> dict[str, Any]:
        subsystem_health = {
            self._patchouli_subsystem.name: await self._patchouli_subsystem.health()
        }
        return {
            "status": "ok" if self._started else "stopped",
            "subsystems": subsystem_health,
            "models_ready": self._patchouli.kernel.is_models_ready(),
        }

    # ========== 聊天 ==========

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        return await self._chat_service.chat(
            user_message=user_message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        )

    async def chat_stream(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in self._chat_service.chat_stream(
            user_message=user_message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        ):
            yield event

    # ========== 被动接入 ==========

    async def ingest_event(
        self,
        event: Any,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._ingress_service.ingest_event(
            event=event,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    async def flush_observer_session(
        self,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> bool:
        return await self._ingress_service.flush_observer_session(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    # ========== 生成控制 ==========

    def cancel_generation(self, generation_id: str) -> bool:
        return self._chat_service.cancel_generation(generation_id)

    # ========== 兼容性访问器 ==========

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    @config.setter
    def config(self, value: HiveMemoryConfig) -> None:
        self._config = value
        self._patchouli.config = value

    @property
    def patchouli(self) -> PatchouliSystem:
        return self._patchouli

    @property
    def kernel(self):
        return self._patchouli.kernel

    @property
    def storage(self):
        return self._patchouli.storage

    async def manual_trigger(self, topic_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._patchouli.manual_trigger(topic_id=topic_id)
