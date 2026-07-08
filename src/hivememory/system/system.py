from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from hivememory.system.assembler import (
    SystemAssembler,
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.system.runtime.events import (
    RuntimeEventBus,
    RuntimeEventSink,
    safe_runtime_event_value,
)

if TYPE_CHECKING:
    from hivememory.gateway import GatewaySystem
    from hivememory.system.application.agent_service import AgentApplicationService
    from hivememory.system.application.chat_service import ChatApplicationService
    from hivememory.system.application.memory_service import MemoryApplicationService
    from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
    from hivememory.system.application.passive_ingress_service import PassiveIngressService
    from hivememory.system.application.readiness_service import SystemReadinessService
    from hivememory.system.application.topic_service import TopicApplicationService


class HiveMemorySystem:
    """
    HiveMemory 顶层系统门面

    薄门面 + 宿主容器。装配逻辑委托给 SystemAssembler，
    本类只负责生命周期管理与应用服务入口。
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        runtime: _RuntimeBundle,
        registries: _RegistriesBundle,
        subsystems: _SubsystemBundle,
        services: _ServicesBundle,
    ) -> None:
        self._config = config

        # 运行时基础设施
        self._global_bus = runtime.global_bus
        self._scheduler = runtime.scheduler
        self._runtime_events = runtime.event_bus
        self._runtime_event_sink = runtime.event_sink

        # 子系统
        self._gateway = subsystems.gateway
        self._patchouli = subsystems.patchouli
        self._alice = subsystems.alice

        # 应用服务
        self._chat_service = services.chat
        self._ingress_service = services.ingress
        self._memory_service = services.memory
        self._memory_task_service = services.memory_task
        self._agent_service = services.agent
        self._topic_service = services.topic
        self._readiness_service = services.readiness

        # 注册表：全局单例，供 API 层（deps.py）注入到路由
        self._model_registry = registries.model_registry
        self._provider_registry = registries.provider_registry

        self._started = False
        self._scheduler_stopped = False

    @classmethod
    def build(
        cls,
        config: HiveMemoryConfig | None = None,
    ) -> HiveMemorySystem:
        from hivememory.system.config import load_app_config

        config = config or load_app_config()
        return SystemAssembler(config).assemble()

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
            "gateway.start",
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
            await self._gateway.start()
            completed_steps.append("gateway.start")
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
            "gateway.stop",
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
            await self._gateway.stop()
            completed_steps.append("gateway.stop")
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
            self._gateway.name: await self._gateway.health(),
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
    def gateway(self) -> GatewaySystem:
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
