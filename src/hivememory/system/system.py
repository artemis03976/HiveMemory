from __future__ import annotations

from time import monotonic
from typing import Any, Literal

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
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import (
    NullRuntimeEventSink,
    RuntimeEventBus,
    RuntimeEventSink,
    safe_runtime_event_value,
)
from hivememory.system.runtime.scheduler.global_scheduler import (
    GlobalMaintenanceScheduler,
)


class HiveMemorySystem:
    """
    HiveMemory 顶层系统门面

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
        model_registry: ModelRegistry,
        provider_registry: ProviderRegistry,
        runtime_events: RuntimeEventBus | None = None,
        runtime_event_sink: RuntimeEventSink | None = None,
        gateway: Any | None = None,
    ) -> None:
        self._config = config

        self._global_bus = global_bus
        self._scheduler = scheduler
        self._runtime_events = runtime_events
        self._runtime_event_sink = runtime_event_sink or NullRuntimeEventSink()

        self._patchouli = patchouli
        self._alice = alice
        self._gateway = gateway

        self._chat_service = chat_service
        self._ingress_service = ingress_service
        self._memory_service = memory_service
        self._memory_task_service = memory_task_service
        self._agent_service = agent_service
        self._topic_service = topic_service
        self._readiness_service = readiness_service

        # 注册表：全局单例，供 API 层（deps.py）注入到路由
        self._model_registry = model_registry
        self._provider_registry = provider_registry

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
        scheduler = GlobalMaintenanceScheduler(
            tick_seconds=config.scheduler.tick_seconds,
            shutdown_wait_seconds=config.scheduler.shutdown_wait_seconds,
            runtime_events=runtime_event_sink.scoped(
                "system",
                component="maintenance_scheduler",
            ),
        )

        # 模型注册表 & 提供商凭证注册表：提前初始化（在 Patchouli/Alice 之前）
        # ProviderRegistry 合并 env 层（来自 .env）与 yaml 层（providers.secrets.yaml）
        provider_registry = ProviderRegistry(env_providers=config.shared.providers)

        # ModelRegistry 注入 ProviderRegistry 引用（动态查询，修改立即生效）
        model_registry = ModelRegistry(provider_registry=provider_registry)

        # 预解析 gateway / librarian 的 LLM 配置：
        # model_id 引用注册表，凭证由 provider 表补齐，temperature/max_tokens 保留组件值。
        # 预解析后 config.shared.llm.* 中的 model/api_key/api_base 已被填充，
        # PatchouliSystem 构造时无需感知注册表。
        config.shared.llm.gateway = model_registry.resolve_for_llm_config(
            config.shared.llm.gateway
        )
        config.shared.llm.librarian = model_registry.resolve_for_llm_config(
            config.shared.llm.librarian
        )

        # 1. System Gateway 创建（使用已解析的 gateway LLM 配置）
        from hivememory.system.gateway import build_system_gateway
        from hivememory.system.gateway.commands import (
            SystemCommandDispatcher,
            create_builtin_command_registry,
        )
        command_config = config.gateway.commands
        command_registry = (
            create_builtin_command_registry(command_config.builtin)
            if command_config.enabled
            else None
        )
        system_gateway = build_system_gateway(
            config=config.gateway,
            llm_config=config.shared.llm.gateway,
            command_registry=command_registry,
        )
        command_dispatcher = (
            SystemCommandDispatcher(
                command_registry,
                global_bus=global_bus,
                debug_enabled=command_config.enable_debug_commands,
                expose_listing=command_config.expose_listing,
            )
            if command_registry is not None
            else None
        )

        # 2. Patchouli 创建（使用已解析的 librarian LLM 配置；Gateway 由 System 注入）
        patchouli = PatchouliSystem(
            config=config,
            gateway_gaze=system_gateway.gaze,
            global_bus=global_bus,
            scheduler=scheduler,
            runtime_events=runtime_event_sink.scoped("patchouli"),
        )

        # 3. Alice 创建（worker 模型在运行时逐帧由注册表解析）
        alice = AliceSystem(
            config=config,
            global_bus=global_bus,
            runtime_events=runtime_event_sink.scoped("alice"),
            model_registry=model_registry,
        )

        chat_service = ChatApplicationService(
            global_bus=global_bus,
            runtime_events=runtime_event_sink.scoped(
                "system",
                component="chat_application_service",
            ),
            command_gaze=system_gateway.gaze if command_config.enabled else None,
            command_dispatcher=command_dispatcher,
            command_config=command_config,
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
            model_registry=model_registry,
            provider_registry=provider_registry,
            runtime_events=runtime_events,
            runtime_event_sink=runtime_event_sink,
            gateway=system_gateway,
        )

    # ========== 生命周期 ==========

    async def start(self) -> None:
        start_time = monotonic()
        completed_steps: list[str] = []
        if self._started:
            self._emit_lifecycle_event(
                RuntimeEventType.SYSTEM_STARTING,
                status="starting",
                data={
                    "already_started": True,
                    "steps": [],
                    "completed_steps": completed_steps,
                },
            )
            self._emit_lifecycle_event(
                RuntimeEventType.SYSTEM_READY,
                status="ready",
                duration_ms=(monotonic() - start_time) * 1000,
                data={
                    "already_started": True,
                    "steps": [],
                    "completed_steps": completed_steps,
                },
            )
            return

        steps = [
            "patchouli.start",
            "alice.start",
            "scheduler.start",
            "passive_ingress.start",
        ]
        self._emit_lifecycle_event(
            RuntimeEventType.SYSTEM_STARTING,
            status="starting",
            data={
                "already_started": False,
                "steps": steps,
                "completed_steps": completed_steps,
            },
        )
        try:
            await self._patchouli.start()
            completed_steps.append("patchouli.start")
            await self._alice.start()
            completed_steps.append("alice.start")
            self._scheduler.start()
            completed_steps.append("scheduler.start")
            await self._ingress_service.start()
            completed_steps.append("passive_ingress.start")
            self._started = True
            self._scheduler_stopped = False
        except Exception as exc:
            self._emit_lifecycle_event(
                RuntimeEventType.SYSTEM_START_FAILED,
                status="failed",
                severity="error",
                reason=str(exc),
                duration_ms=(monotonic() - start_time) * 1000,
                data={
                    "already_started": False,
                    "steps": steps,
                    "completed_steps": completed_steps,
                    "failed_step": self._first_unfinished_step(
                        steps,
                        completed_steps,
                    ),
                    "error": str(exc),
                },
            )
            raise

        self._emit_lifecycle_event(
            RuntimeEventType.SYSTEM_READY,
            status="ready",
            duration_ms=(monotonic() - start_time) * 1000,
            data={
                "already_started": False,
                "steps": steps,
                "completed_steps": completed_steps,
            },
        )

    async def stop(self) -> None:
        start_time = monotonic()
        was_started = self._started
        completed_steps: list[str] = []
        steps = [
            "scheduler.stop",
            "passive_ingress.shutdown_drain",
            "alice.stop",
            "patchouli.stop",
        ]
        self._emit_lifecycle_event(
            RuntimeEventType.SYSTEM_SHUTTING_DOWN,
            status="shutting_down",
            data={
                "already_stopped": not was_started,
                "steps": steps,
                "completed_steps": completed_steps,
            },
        )

        passive_shutdown_drain: Any = None
        scheduler_stopped = self._scheduler_stopped
        try:
            await self._stop_scheduler()
            scheduler_stopped = self._scheduler_stopped
            completed_steps.append("scheduler.stop")
            passive_shutdown_drain = await self._ingress_service.shutdown_drain()
            completed_steps.append("passive_ingress.shutdown_drain")
            if not was_started:
                self._emit_lifecycle_event(
                    RuntimeEventType.SYSTEM_STOPPED,
                    status="stopped",
                    duration_ms=(monotonic() - start_time) * 1000,
                    data={
                        "already_stopped": True,
                        "steps": steps,
                        "completed_steps": completed_steps,
                        "scheduler_stopped": scheduler_stopped,
                        "passive_shutdown_drain": passive_shutdown_drain,
                    },
                )
                return

            await self._alice.stop()
            completed_steps.append("alice.stop")
            await self._patchouli.stop()
            completed_steps.append("patchouli.stop")
            self._started = False
        except Exception as exc:
            self._emit_lifecycle_event(
                RuntimeEventType.SYSTEM_STOP_FAILED,
                status="failed",
                severity="error",
                reason=str(exc),
                duration_ms=(monotonic() - start_time) * 1000,
                data={
                    "already_stopped": not was_started,
                    "steps": steps,
                    "completed_steps": completed_steps,
                    "failed_step": self._first_unfinished_step(
                        steps,
                        completed_steps,
                    ),
                    "scheduler_stopped": scheduler_stopped,
                    "passive_shutdown_drain": passive_shutdown_drain,
                    "error": str(exc),
                },
            )
            raise

        self._scheduler_stopped = False
        self._emit_lifecycle_event(
            RuntimeEventType.SYSTEM_STOPPED,
            status="stopped",
            duration_ms=(monotonic() - start_time) * 1000,
            data={
                "already_stopped": False,
                "steps": steps,
                "completed_steps": completed_steps,
                "scheduler_stopped": scheduler_stopped,
                "passive_shutdown_drain": passive_shutdown_drain,
            },
        )

    async def _stop_scheduler(self) -> None:
        scheduler_running = getattr(self._scheduler, "is_running", False)
        if self._scheduler_stopped or (not self._started and not scheduler_running):
            return
        await self._scheduler.stop()
        self._scheduler_stopped = True

    def _emit_lifecycle_event(
        self,
        event_type: RuntimeEventType,
        *,
        status: str,
        severity: Literal["debug", "info", "warning", "error"] = "info",
        reason: str | None = None,
        duration_ms: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        # 系统生命周期事件只用于外部观测，不改变 start/stop 的业务语义。
        payload = {
            "duration_ms": duration_ms,
            **(data or {}),
        }
        self._runtime_event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                source="system",
                subsystem="system",
                component="hivememory_system",
                severity=severity,
                status=status,
                reason=reason,
                data=safe_runtime_event_value(payload),
            )
        )

    @staticmethod
    def _first_unfinished_step(
        steps: list[str],
        completed_steps: list[str],
    ) -> str | None:
        completed = set(completed_steps)
        for step in steps:
            if step not in completed:
                return step
        return None

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
    def gateway(self) -> Any | None:
        return self._gateway

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

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry
