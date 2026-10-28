from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from hivememory.components.events.bus import (
    RuntimeEventBus,
    RuntimeEventSink,
    safe_runtime_event_value,
)
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.runtime_events import (
    RuntimeEvent,
    RuntimeEventType,
)
from hivememory.system.assembler import (
    SystemAssembler,
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry

if TYPE_CHECKING:
    from hivememory.gateway import GatewaySystem
    from hivememory.system.application.import_bus_service import ImportBusService
    from hivememory.system.application.readiness_service import SystemReadinessService
    from hivememory.workspace.authentication import ActorAuthenticationGateway
    from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
    from hivememory.workspace.capability.root.assets import RootWorkspaceAssetService
    from hivememory.workspace.capability.root.memory import RootMemoryService
    from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
    from hivememory.workspace.capability.root.session import RootSessionService
    from hivememory.workspace.capability.root.topic import RootTopicService
    from hivememory.workspace.process.service import TaskProcessService
    from hivememory.workspace.read_view import WorkspaceReadView


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
        access_gateway: ActorAuthenticationGateway,
    ) -> None:
        self._config = config

        # 运行时基础设施
        self._global_bus = runtime.global_bus
        self._scheduler = runtime.scheduler
        self._workspace_asset_store = runtime.workspace_asset_store
        self._workspace_read_view = runtime.workspace_read_view
        self._sessions = runtime.sessions
        self._runtime_events = runtime.event_bus
        self._runtime_event_sink = runtime.event_sink

        # 子系统
        self._gateway = subsystems.gateway
        self._patchouli = subsystems.patchouli
        self._alice = subsystems.alice

        # 应用服务
        self._process_service = services.process
        self._import_bus_service = services.import_bus
        self._root_memory_service = services.root_memory
        self._root_memory_task_service = services.root_memory_task
        self._root_agent_service = services.root_agent
        self._root_topic_service = services.root_topic
        self._root_session_service = services.root_session
        self._readiness_service = services.readiness
        self._root_workspace_asset_service = services.root_workspace_assets

        # 注册表：全局单例，供 API 层（deps.py）注入到路由
        self._model_registry = registries.model_registry
        self._provider_registry = registries.provider_registry

        # 访问控制：统一认证网关是唯一对外认证入口，生产 HTTP 入口经它
        # 认证。签发与撤销归认证一侧（经网关转交），操作授权者无状态、
        # 不需要关闭，门面只持有网关以在停止时先关闭、最后撤销。
        self._access_gateway = access_gateway

        self._started = False
        self._scheduler_stopped = False

    @classmethod
    def build(
        cls,
        config: HiveMemoryConfig | None = None,
    ) -> HiveMemorySystem:
        from hivememory.config.app import load_app_config

        config = config or load_app_config()
        return SystemAssembler(config).assemble()

    # ========== 生命周期 ==========

    async def start(self) -> None:
        if self._workspace_asset_store.is_closed:
            # stop 后的 Store 不提供 reopen；重启必须重新装配新的 System。
            raise RuntimeError("HiveMemorySystem 已停止，不能重新启动已关闭的 AssetStore")
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
            "access_gateway.close",
            "scheduler.stop",
            "import_bus.shutdown",
            "process_service.shutdown",
            "sessions.close_and_clear",
            "alice.stop",
            "patchouli.stop",
            "gateway.stop",
            "workspace_read_view.close",
            "workspace_asset_store.close_and_clear",
            "workspace_access_contexts.revoke",
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

        import_bus_shutdown: Any = None
        scheduler_stopped = self._scheduler_stopped
        try:
            # 先关闭统一认证网关、拒绝新认证：已签发
            # context 由各自所有者在收尾时撤销；任务进程收尾后经网关撤销全部。
            self._access_gateway.close()
            completed_steps.append("access_gateway.close")
            await self._stop_scheduler()
            scheduler_stopped = self._scheduler_stopped
            completed_steps.append("scheduler.stop")
            # 认证网关已关闭，新的导入请求以 503 拒绝；这里只等待在途导入，
            # 已接纳的交互由 Patchouli 停止时排空提交队列。
            import_bus_shutdown = await self._import_bus_service.shutdown()
            completed_steps.append("import_bus.shutdown")
            await self._process_service.shutdown()
            completed_steps.append("process_service.shutdown")
            self._sessions.close_and_clear()
            completed_steps.append("sessions.close_and_clear")
            if not was_started:
                self._workspace_read_view.close()
                completed_steps.append("workspace_read_view.close")
                self._workspace_asset_store.close_and_clear()
                completed_steps.append("workspace_asset_store.close_and_clear")
                self._access_gateway.revoke_all_contexts()
                completed_steps.append("workspace_access_contexts.revoke")
                self._emit_lifecycle_event(
                    RuntimeEventType.SYSTEM_STOPPED,
                    status="stopped",
                    duration_ms=(monotonic() - start_time) * 1000,
                    data={
                        "already_stopped": True,
                        "steps": steps,
                        "completed_steps": completed_steps,
                        "scheduler_stopped": scheduler_stopped,
                        "import_bus_shutdown": import_bus_shutdown,
                    },
                )
                return

            await self._alice.stop()
            completed_steps.append("alice.stop")
            await self._patchouli.stop()
            completed_steps.append("patchouli.stop")
            await self._gateway.stop()
            completed_steps.append("gateway.stop")
            # 最终清理必须晚于 Patchouli Topic settlement/generation drain 与剩余消费者；
            # Alice 已卸载运行入口；workspace 继续保持
            # 结算/失效订阅直到 Patchouli drain 完成，随后停止新读、取消订阅
            # 并清理派生缓存（不触碰 canonical），最后收尾 AssetStore。
            self._workspace_read_view.close()
            completed_steps.append("workspace_read_view.close")
            self._workspace_asset_store.close_and_clear()
            completed_steps.append("workspace_asset_store.close_and_clear")
            # 最后经认证网关撤销全部已签发 context：在途任务进程与请求收尾
            # 仍需用各自持有的 context 通过授权；此后它们不再能通过授权
            # （操作授权者无状态，不需要关闭）。
            self._access_gateway.revoke_all_contexts()
            completed_steps.append("workspace_access_contexts.revoke")
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
                    "import_bus_shutdown": import_bus_shutdown,
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
                "import_bus_shutdown": import_bus_shutdown,
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
    def process_service(self) -> TaskProcessService:
        return self._process_service

    @property
    def import_bus_service(self) -> ImportBusService:
        """接收外部 harness 一轮完整交互记录的 Import Bus 服务。"""
        return self._import_bus_service

    @property
    def root_memory_service(self) -> RootMemoryService:
        """只供 root 调用的记忆管理 API，actor 操作经操作入口提交。"""
        return self._root_memory_service

    @property
    def root_memory_task_service(self) -> RootMemoryTaskService:
        """只供 root 调用的记忆任务管理 API。"""
        return self._root_memory_task_service

    @property
    def root_agent_service(self) -> RootAgentProfileService:
        """只供 root 调用的 Agent Profile 管理 API。"""
        return self._root_agent_service

    @property
    def root_session_service(self) -> RootSessionService:
        """会话管理 root API。"""
        return self._root_session_service

    @property
    def root_topic_service(self) -> RootTopicService:
        """只供 root 调用的话题管理 API。"""
        return self._root_topic_service

    @property
    def readiness_service(self) -> SystemReadinessService:
        return self._readiness_service

    @property
    def root_workspace_asset_service(self) -> RootWorkspaceAssetService:
        """只供 root 调用的 WorkspaceAsset 上传 API。"""
        return self._root_workspace_asset_service

    @property
    def workspace_read_view(self) -> WorkspaceReadView:
        """workspace 读取能力与派生缓存的运行时聚合。"""
        return self._workspace_read_view

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
        # Patchouli 只持有自己的配置段（组合根按段注入），同步更新该段引用。
        self._patchouli.config = value.patchouli

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry

    @property
    def access_gateway(self) -> ActorAuthenticationGateway:
        """统一 Actor Authentication 网关。"""
        return self._access_gateway
