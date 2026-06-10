from __future__ import annotations

from typing import Any

from hivememory.alice.system import AliceSystem
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.config import RuntimeEventsConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventBus, RuntimeEventSink
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
        memory_service: MemoryApplicationService,
        memory_task_service: MemoryTaskApplicationService,
        agent_service: AgentApplicationService,
        topic_service: TopicApplicationService,
        readiness_service: SystemReadinessService,
        runtime_events: RuntimeEventBus | None = None,
        runtime_event_sink: RuntimeEventSink | None = None,
    ) -> None:
        self._config = config

        self._global_bus = global_bus
        self._scheduler = scheduler
        self._runtime_events = runtime_events
        self._runtime_event_sink = runtime_event_sink or NullRuntimeEventSink()

        self._patchouli = patchouli
        self._alice = alice

        self._chat_service = chat_service
        self._ingress_service = ingress_service
        self._memory_service = memory_service
        self._memory_task_service = memory_task_service
        self._agent_service = agent_service
        self._topic_service = topic_service
        self._readiness_service = readiness_service

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
        runtime_events_config = getattr(config, "runtime_events", None)
        if not isinstance(runtime_events_config, RuntimeEventsConfig):
            runtime_events_config = RuntimeEventsConfig()

        runtime_events = (
            RuntimeEventBus(
                buffer_size=runtime_events_config.buffer_size,
                subscriber_queue_size=runtime_events_config.subscriber_queue_size,
            )
            if runtime_events_config.enabled
            else None
        )
        runtime_event_sink: RuntimeEventSink = runtime_events or NullRuntimeEventSink()

        # 1. Patchouli 先创建（提供 bus 和 storage，并通过 global bus 调用 Alice）
        patchouli = PatchouliSystem(
            config=config,
            global_bus=global_bus,
            scheduler=scheduler,
            runtime_events=runtime_event_sink.scoped("patchouli"),
        )

        # 2. Alice 创建（使用自有 AliceBus，通过全局总线访问 Patchouli 记忆能力）
        alice = AliceSystem(
            config=config,
            global_bus=global_bus,
            runtime_events=runtime_event_sink.scoped("alice"),
        )

        chat_service = ChatApplicationService(
            global_bus=global_bus,
            runtime_events=runtime_event_sink.scoped(
                "system",
                component="chat_application_service",
            ),
        )
        ingress_service = PassiveIngressService(
            bus=global_bus,
            config=config,
            scheduler=scheduler,
        )
        memory_service = MemoryApplicationService(
            global_bus=global_bus,
            config=config,
        )
        memory_task_service = MemoryTaskApplicationService(
            global_bus=global_bus,
        )
        agent_service = AgentApplicationService(
            global_bus=global_bus,
            config=config,
        )
        topic_service = TopicApplicationService(
            global_bus=global_bus,
            config=config,
        )
        readiness_service = SystemReadinessService(
            global_bus=global_bus,
        )

        return cls(
            config=config,
            patchouli=patchouli,
            alice=alice,
            global_bus=global_bus,
            scheduler=scheduler,
            chat_service=chat_service,
            ingress_service=ingress_service,
            memory_service=memory_service,
            memory_task_service=memory_task_service,
            agent_service=agent_service,
            topic_service=topic_service,
            readiness_service=readiness_service,
            runtime_events=runtime_events,
            runtime_event_sink=runtime_event_sink,
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

    # ========== 应用服务入口 ==========

    @property
    def chat_service(self) -> ChatApplicationService:
        return self._chat_service

    @property
    def ingress_service(self) -> PassiveIngressService:
        return self._ingress_service

    @property
    def memory_service(self) -> MemoryApplicationService:
        return self._memory_service

    @property
    def memory_task_service(self) -> MemoryTaskApplicationService:
        return self._memory_task_service

    @property
    def agent_service(self) -> AgentApplicationService:
        return self._agent_service

    @property
    def topic_service(self) -> TopicApplicationService:
        return self._topic_service

    @property
    def readiness_service(self) -> SystemReadinessService:
        return self._readiness_service

    @property
    def runtime_events(self) -> RuntimeEventBus | None:
        return self._runtime_events

    @property
    def runtime_event_sink(self) -> RuntimeEventSink:
        return self._runtime_event_sink

    # ========== 配置管理 ==========

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    @config.setter
    def config(self, value: HiveMemoryConfig) -> None:
        self._config = value
        self._patchouli.config = value
