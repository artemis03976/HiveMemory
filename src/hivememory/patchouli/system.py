"""
帕秋莉子系统 (The Patchouli System / The Facility)

定位：记忆存储、检索与感知子系统 (v4 Phase B/C)
职责：
    - 持有 TheEye (Ingress Gateway): 意图识别、查询重写、话题路由
    - 持有 PatchouliRuntime: 记忆运行时，管理感知 (Perception)、检索 (Retrieval)、生成 (Generation) 与生命周期 (Lifecycle)
    - 提供记忆能力公开路由 (memory.retrieve, memory.retrieve_by_aliases)
    - 实现 SubsystemProtocol 契约

数据流:
    Active: ChatService -> prepare_agent_run (Patchouli) -> run_agent (Alice) -> finalize_agent_run (Patchouli)
    Passive: PassiveIngressService -> ingest_event -> Patchouli (submit_interaction)

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye (Gateway) ──→ PatchouliRuntime  │
    │                         ├── Perception  │
    │                         ├── Retrieval   │
    │                         ├── Generation  │
    │                         └── Lifecycle   │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 4.0
"""

import logging

from typing import TYPE_CHECKING, Any, Optional

from hivememory.patchouli.runtime.bridge import PatchouliBridge
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    MemoryManagementService,
    MemoryTaskManagementService,
    ModelReadinessService,
    TopicManagementService,
)
from hivememory.patchouli.runtime import PatchouliRuntime
from hivememory.patchouli.service import PatchouliService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec

if TYPE_CHECKING:
    from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler

logger = logging.getLogger(__name__)


class PatchouliSystem(SubsystemProtocol):
    """
    帕秋莉体系 (The Facility) - HiveMemory 的完整封装 v4.0

    外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliRuntime (运行时宿主)。
    TheEye 独立于 Runtime 之外，做第一道拦截与信息重整，
    处理完后将标准化请求传入 Runtime / Service 进行后续处理。

    架构:
        - TheEye (真理之眼): Ingress Gateway，流量入口、意图判断、查询重写
        - PatchouliRuntime (帕秋莉运行时): 记忆域运行时宿主
            - RetrievalFamiliar (检索使魔): 上下文检索 (Hot)
            - LibrarianCore (馆长本体): 后台记忆维护 (Cold)
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: Optional[GlobalSystemBus] = None,
        scheduler: Optional["AsyncMaintenanceScheduler"] = None,
        runtime_events: RuntimeEventSink | None = None,
    ):
        self.config = config
        self._global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        # 1. 初始化 Runtime（运行时负责组装 Retrieval + Librarian 组件图）
        self.runtime = PatchouliRuntime(
            patchouli_config=self.config.patchouli,
            shared_config=self.config.shared,
            runtime_events=self._runtime_events,
        )

        # 2. 初始化 Gateway
        self._init_gateway()

        # 3. 构建 TheEye (Phase 4.5 Agentic Dispatcher — 仅保留 gaze 职责)
        self.eye = TheEye(
            engine=self._gateway_engine,
        )

        # 4. Patchouli 对外能力门面
        self._service = PatchouliService(
            runtime=self.runtime,
            eye=self.eye,
            global_bus=global_bus,
        )
        self._memory_management_service = MemoryManagementService(
            storage=self.runtime.storage,
            lifecycle_familiar=self.runtime.lifecycle_familiar,
        )
        self._memory_task_management_service = MemoryTaskManagementService(
            task_controller=self.runtime._task_controller,
        )
        self._agent_profile_management_service = AgentProfileManagementService(
            storage=self.runtime.storage,
        )
        self._topic_management_service = TopicManagementService(
            perception_familiar=self.runtime.perception_familiar,
            retrieval_familiar=self.runtime.retrieval_familiar,
        )
        self._model_readiness_service = ModelReadinessService(
            runtime=self.runtime,
        )
        self._bridge = PatchouliBridge(
            runtime=self.runtime,
            service=self._service,
            memory_management_service=self._memory_management_service,
            memory_task_management_service=self._memory_task_management_service,
            agent_profile_management_service=self._agent_profile_management_service,
            topic_management_service=self._topic_management_service,
            model_readiness_service=self._model_readiness_service,
            global_bus=global_bus,
        )

        self._scheduler = scheduler
        self._maintenance_registered = False

        logger.info("PatchouliSystem 帕秋莉系统初始化完成")

    def _init_gateway(self) -> None:
        """
        初始化 Gateway 相关基础设施

        Gateway LLM 和 Gateway Engine 属于 TheEye 的依赖，
        独立于 Runtime 管理。
        """
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self._gateway_llm_service = get_gateway_llm_service(
            config=self.config.shared.llm.gateway
        )

        from hivememory.engines.gateway import (
            BaseInterceptor,
            BaseSemanticAnalyzer,
            GatewayEngine,
            create_interceptor,
            create_semantic_analyzer,
        )

        config = self.config.patchouli.gateway

        interceptor: BaseInterceptor = create_interceptor(config.interceptor)

        semantic_analyzer: BaseSemanticAnalyzer = create_semantic_analyzer(
            config.analyzer,
            self._gateway_llm_service,
        )

        self._gateway_engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

    @property
    def service(self) -> PatchouliService:
        """访问 Patchouli 对外能力门面。"""
        return self._service

    @property
    def name(self) -> str:
        return "patchouli"

    # ========== 维护任务注册 ==========

    _MAINTENANCE_OWNER = "patchouli"

    def register_maintenance_tasks(self, scheduler) -> bool:
        """向全局维护器注册 Patchouli 子系统的维护任务。"""
        if not self.config.scheduler.enabled:            return False
        tasks_config = self.config.scheduler.tasks
        scheduler.register(
            MaintenanceTaskSpec(
                owner=self._MAINTENANCE_OWNER,
                name="perception_idle_flush",
                interval_seconds=tasks_config.perception_idle_flush_interval_seconds,
                enabled=tasks_config.enable_perception_idle_flush,
            ),
            self.runtime.perception_familiar.scan_idle_buffers_once,
        )
        scheduler.register(
            MaintenanceTaskSpec(
                owner=self._MAINTENANCE_OWNER,
                name="memory_gardening",
                interval_seconds=tasks_config.lifecycle_gc_interval_hours * 3600,
                enabled=tasks_config.enable_lifecycle_gc,
            ),
            self.runtime.lifecycle_familiar.run_gardening_once,
        )
        return True

    def unregister_maintenance_tasks(self, scheduler) -> int:
        """从全局维护器卸载 Patchouli 子系统的维护任务。"""
        return scheduler.unregister_owner(self._MAINTENANCE_OWNER)

    async def start(self) -> None:
        await self.runtime.ensure_storage_ready()

        if not self.runtime.local_routes_registered:
            self.runtime.mount_local_routes(self.service)

        self._bridge.mount()

        if self._scheduler and not self._maintenance_registered:
            self._maintenance_registered = self.register_maintenance_tasks(
                self._scheduler
            )

    async def stop(self) -> None:
        if self._scheduler and self._maintenance_registered:
            self.unregister_maintenance_tasks(self._scheduler)
            self._maintenance_registered = False

        await self.runtime.shutdown_drain()

        self._bridge.unmount()

        self.runtime.unmount_local_routes()

    async def health(self) -> dict[str, Any]:
        models_ready = self.runtime.is_models_ready()
        return {
            "status": "ok" if models_ready else "warming_up",
            "models_ready": models_ready,
        }

__all__ = [
    "PatchouliSystem",
]
