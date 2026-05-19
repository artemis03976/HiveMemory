from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from hivememory.alice.system import AliceSystem
from hivememory.core.protocol.models import ChatResult
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.config import HiveMemoryConfig
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
        alice: AliceSystem,
        global_bus: GlobalSystemBus,
        scheduler: GlobalMaintenanceScheduler,
        chat_service: ChatApplicationService,
        ingress_service: PassiveIngressService,
    ) -> None:
        self._config = config

        self._global_bus = global_bus
        self._scheduler = scheduler

        self._patchouli = patchouli
        self._alice = alice

        self._chat_service = chat_service
        self._ingress_service = ingress_service

        self._started = False
        self._scheduler_stopped = False

    @classmethod
    def build(
        cls,
        config: HiveMemoryConfig | None = None,
    ) -> HiveMemorySystem:
        from hivememory.system.config import load_app_config

        config = config or load_app_config()

        global_bus = GlobalSystemBus()
        scheduler = GlobalMaintenanceScheduler(
            tick_seconds=config.scheduler.tick_seconds,
            shutdown_wait_seconds=config.scheduler.shutdown_wait_seconds,
        )

        # 1. Patchouli 先创建（提供 bus 和 storage，并通过 global bus 调用 Alice）
        patchouli = PatchouliSystem(
            config=config,
            global_bus=global_bus,
            scheduler=scheduler,
        )

        # 2. Alice 创建（使用自有 AliceBus，通过全局总线访问 Patchouli 记忆能力）
        alice = AliceSystem(
            config=config,
            global_bus=global_bus,
        )

        chat_service = ChatApplicationService(global_bus=global_bus)
        ingress_service = PassiveIngressService(
            bus=global_bus,
            config=config,
            scheduler=scheduler,
        )

        return cls(
            config=config,
            patchouli=patchouli,
            alice=alice,
            global_bus=global_bus,
            scheduler=scheduler,
            chat_service=chat_service,
            ingress_service=ingress_service,
        )

    # ========== 生命周期 ==========

    async def start(self) -> None:
        if self._started:
            return
        await self._patchouli.start()
        await self._alice.start()
        self._scheduler.start()
        self._started = True
        self._scheduler_stopped = False
        await self._ingress_service.start()

    async def stop(self) -> None:
        await self._stop_scheduler()
        await self._ingress_service.shutdown_drain()
        if not self._started:
            return
        await self._alice.stop()
        await self._patchouli.stop()
        self._started = False
        self._scheduler_stopped = False

    async def _stop_scheduler(self) -> None:
        if not self._started or self._scheduler_stopped:
            return
        await self._scheduler.stop()
        self._scheduler_stopped = True

    async def health(self) -> dict[str, Any]:
        subsystem_health = {
            self._patchouli.name: await self._patchouli.health(),
            self._alice.name: await self._alice.health(),
        }
        return {
            "status": "ok" if self._started else "stopped",
            "subsystems": subsystem_health,
            "models_ready": self._patchouli.runtime.is_models_ready(),
        }

    # ========== 聊天 ==========

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: str | None = None,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
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
        session_id: str | None = None,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
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
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._ingress_service.ingest_event(
            event=event,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    async def flush_ingressor(
        self,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: str | None = None,
    ) -> bool:
        return await self._ingress_service.flush_ingressor(
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
    def alice(self) -> AliceSystem:
        return self._alice

    @property
    def runtime(self):
        return self._patchouli.runtime

    @property
    def storage(self):
        return self._patchouli.storage

    async def manual_archive_topic(self, topic_id: str | None = None) -> dict[str, Any]:
        return await self._global_bus.request(
            "patchouli.public.manual_archive_topic",
            topic_id=topic_id,
        )
