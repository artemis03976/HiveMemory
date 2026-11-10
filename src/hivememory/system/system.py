from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from hivememory.components.events.bus import RuntimeEventBus
from hivememory.components.events.publisher import Severity
from hivememory.config.app import HiveMemoryConfig
from hivememory.config.files import write_atomically
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.patchouli.contracts import PatchouliRoutes
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
    from hivememory.core.models import WorkspaceIdentity
    from hivememory.gateway import GatewaySystem
    from hivememory.system.access import SystemActorAccessRegistry
    from hivememory.system.application.import_bus_service import ImportBusService
    from hivememory.system.application.readiness_service import SystemReadinessService
    from hivememory.workspace.authentication import ActorAuthenticationGateway
    from hivememory.workspace.capability.operations import WorkspaceOperationEntry
    from hivememory.workspace.capability.root.access import RootAccessService
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
        principal_registry: SystemActorAccessRegistry,
    ) -> None:
        self._config = config

        # 运行时基础设施
        self._global_bus = runtime.global_bus
        self._scheduler = runtime.scheduler
        self._workspace_asset_store = runtime.workspace_asset_store
        self._workspace_read_view = runtime.workspace_read_view
        self._sessions = runtime.sessions
        self._runtime_events = runtime.event_bus
        self._lifecycle_events = runtime.event_publisher.scoped(
            subsystem="system", source="system", component="hivememory_system"
        )

        # 子系统
        self._gateway = subsystems.gateway
        self._patchouli = subsystems.patchouli
        self._alice = subsystems.alice

        # 应用服务
        self._process_service = services.process
        self._import_bus_service = services.import_bus
        self._operation_entry = services.operation_entry
        self._root_memory_service = services.root_memory
        self._root_memory_task_service = services.root_memory_task
        self._root_agent_service = services.root_agent
        self._root_topic_service = services.root_topic
        self._root_session_service = services.root_session
        self._root_access_service = services.root_access
        self._readiness_service = services.readiness
        self._root_workspace_asset_service = services.root_workspace_assets

        # 注册表：全局单例，供 API 层（deps.py）注入到路由
        self._model_registry = registries.model_registry
        self._provider_registry = registries.provider_registry

        # 访问控制：统一认证网关是唯一对外认证入口，生产 HTTP 入口经它
        # 认证，得到工作 actor。操作授权者无状态、不需要关闭，门面只持有
        # 网关以在停止时先关闭、拒绝新认证。
        self._access_gateway = access_gateway
        # 接入登记属于 System：管理接口与提供商、模型接口一样直接读写注册表。
        self._principal_registry = principal_registry

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
                reason="start_step_failed",
                duration_ms=(monotonic() - start_time) * 1000,
                data={
                    "already_started": False,
                    "steps": steps,
                    "completed_steps": completed_steps,
                    "failed_step": self._first_unfinished_step(
                        steps,
                        completed_steps,
                    ),
                    "error_class": type(exc).__name__,
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
            # 先关闭统一认证网关、拒绝新认证：新的请求与导入不再取得工作 actor，
            # 已有进程照常收尾；在途请求由随后各组件的关闭拒绝。
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
            self._started = False
        except Exception as exc:
            self._emit_lifecycle_event(
                RuntimeEventType.SYSTEM_STOP_FAILED,
                status="failed",
                severity="error",
                reason="stop_step_failed",
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
                    "error_class": type(exc).__name__,
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
        severity: Severity = "info",
        reason: str | None = None,
        duration_ms: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        # 系统生命周期事件只用于外部观测，不改变 start/stop 的业务语义。
        payload = {
            "duration_ms": duration_ms,
            **(data or {}),
        }
        self._lifecycle_events.emit(
            event_type, status=status, severity=severity, reason=reason, data=payload
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

    async def seed_preset_agent_profiles(self, workspace: WorkspaceIdentity) -> list[str]:
        """初次部署时向 Workspace 写入预设 Agent，只做一次，返回写入的 alias。

        数据根中有标记文件时什么也不做；写入成功后留下标记，用户之后删掉预设 Agent
        也不会在重启时写回。写入需要嵌入模型，调用方在模型预热之后调用。
        """
        marker = Path(self._config.system.preset_agent_profiles_marker)
        if marker.exists():
            return []
        created: list[str] = await self._global_bus.request(
            PatchouliRoutes.AGENT_PROFILE_SEED_PRESETS, workspace
        )
        write_atomically(marker, b"")
        return created

    # ========== 应用服务入口 ==========

    @property
    def process_service(self) -> TaskProcessService:
        return self._process_service

    @property
    def import_bus_service(self) -> ImportBusService:
        """Import Bus 服务：接收外部 harness 一轮完整交互记录，也为写入中的一轮确定话题。

        导入路由使用前者；MCP 入口在写入与更新前使用后者。
        """
        return self._import_bus_service

    @property
    def operation_entry(self) -> WorkspaceOperationEntry:
        """actor API 唯一的调用方：与 Alice 共用同一实例，门面只供 MCP 入口使用。

        MCP 入口以每次调用认证得到的工作 actor 经 ``execute_request`` 提交检索、
        读取、写入与更新；门面不交出 actor API 本身。
        """
        return self._operation_entry

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
    def root_access_service(self) -> RootAccessService:
        """root 访问记录管理 API（准入与 actor 操作白名单）。"""
        return self._root_access_service

    @property
    def principal_registry(self) -> SystemActorAccessRegistry:
        """System 接入注册表：外部来源的运行时登记、修改、凭据轮换与删除。"""
        return self._principal_registry

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

    # ========== 配置管理 ==========

    @property
    def config(self) -> HiveMemoryConfig:
        """运行配置：启动时装配的配置对象树，组件持有其中的配置段，不整体替换。"""
        return self._config

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
