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

from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    MemoryManagementService,
    ModelReadinessService,
    TopicManagementService,
)
from hivememory.patchouli.runtime import PatchouliRuntime
from hivememory.patchouli.service import PatchouliService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
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
    ):
        self.config = config
        self._global_bus = global_bus

        # 1. 初始化 Runtime（运行时负责组装 Retrieval + Librarian 组件图）
        self.runtime = PatchouliRuntime(config=self.config)

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
            lifecycle_engine=self.runtime.librarian_core.lifecycle_engine,
        )
        self._agent_profile_management_service = AgentProfileManagementService(
            storage=self.runtime.storage,
        )
        self._topic_management_service = TopicManagementService(
            librarian_core=self.runtime.librarian_core,
        )
        self._model_readiness_service = ModelReadinessService(
            runtime=self.runtime,
        )

        self._scheduler = scheduler
        self._public_routes_registered = False
        self._maintenance_registered = False
        self._local_events_registered = False

        logger.info("PatchouliSystem 帕秋莉系统初始化完成")

    def _init_gateway(self) -> None:
        """
        初始化 Gateway 相关基础设施

        Gateway LLM 和 Gateway Engine 属于 TheEye 的依赖，
        独立于 Runtime 管理。
        """
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self._gateway_llm_service = get_gateway_llm_service(
            config=self.config.llm.gateway
        )

        from hivememory.engines.gateway import (
            BaseInterceptor,
            BaseSemanticAnalyzer,
            GatewayEngine,
            create_interceptor,
            create_semantic_analyzer,
        )

        config = self.config.gateway

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
        if not self.config.scheduler.enabled:
            return False
        tasks_config = self.config.scheduler.tasks
        scheduler.register(
            MaintenanceTaskSpec(
                owner=self._MAINTENANCE_OWNER,
                name="perception_idle_flush",
                interval_seconds=tasks_config.perception_idle_flush_interval_seconds,
                enabled=tasks_config.enable_perception_idle_flush,
            ),
            self.runtime.librarian_core.perception_layer.scan_idle_buffers_once,
        )
        scheduler.register(
            MaintenanceTaskSpec(
                owner=self._MAINTENANCE_OWNER,
                name="memory_gardening",
                interval_seconds=tasks_config.lifecycle_gc_interval_hours * 3600,
                enabled=tasks_config.enable_lifecycle_gc,
            ),
            self.runtime.librarian_core.run_gardening_once,
        )
        return True

    def unregister_maintenance_tasks(self, scheduler) -> int:
        """从全局维护器卸载 Patchouli 子系统的维护任务。"""
        return scheduler.unregister_owner(self._MAINTENANCE_OWNER)

    async def start(self) -> None:
        if not self.runtime.local_routes_registered:
            self.runtime.mount_local_routes(self.service)

        if self._global_bus and not self._local_events_registered:
            self._register_local_event_bridges()
            self._local_events_registered = True

        if self._global_bus and not self._public_routes_registered:
            self._register_public_routes()
            self._public_routes_registered = True

        if self._scheduler and not self._maintenance_registered:
            self._maintenance_registered = self.register_maintenance_tasks(
                self._scheduler
            )

    async def stop(self) -> None:
        if self._scheduler and self._maintenance_registered:
            self.unregister_maintenance_tasks(self._scheduler)
            self._maintenance_registered = False

        await self.runtime.shutdown_drain()

        if self._global_bus and self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False

        if self._global_bus and self._local_events_registered:
            self._unregister_local_event_bridges()
            self._local_events_registered = False

        self.runtime.unmount_local_routes()

    async def health(self) -> dict[str, Any]:
        models_ready = self.runtime.is_models_ready()
        return {
            "status": "ok" if models_ready else "warming_up",
            "models_ready": models_ready,
        }

    def _register_public_routes(self) -> None:
        self._global_bus.register(
            PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
            self.service.analyze_and_retrieve,
        )
        self._global_bus.register(
            PatchouliRoutes.SUBMIT_INTERACTION,
            self.runtime.librarian_core.submit_interaction,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_CREATE,
            self._memory_management_service.create_memory,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_LIST,
            self._memory_management_service.list_memories,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_GET,
            self._memory_management_service.get_memory,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_UPDATE,
            self._memory_management_service.update_memory,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_DELETE,
            self._memory_management_service.delete_memory,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_RECORD_FEEDBACK,
            self._memory_management_service.record_feedback,
        )
        self._global_bus.register(
            PatchouliRoutes.AGENT_PROFILE_CREATE,
            self._agent_profile_management_service.create_agent_profile,
        )
        self._global_bus.register(
            PatchouliRoutes.AGENT_PROFILE_LIST,
            self._agent_profile_management_service.list_agent_profiles,
        )
        self._global_bus.register(
            PatchouliRoutes.TOPIC_LIST_ACTIVE,
            self._topic_management_service.list_active_topics,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_RETRIEVE,
            self.runtime.retrieval_familiar.retrieve_async,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            self.runtime.retrieval_familiar.retrieve_by_aliases_async,
        )
        self._global_bus.register(
            PatchouliRoutes.GET_AGENT_PROFILE,
            self.runtime._get_agent_profile,
        )
        self._global_bus.register(
            PatchouliRoutes.PREPARE_AGENT_RUN,
            self.service.prepare_agent_run,
        )
        self._global_bus.register(
            PatchouliRoutes.FINALIZE_AGENT_RUN,
            self.service.finalize_agent_run,
        )
        self._global_bus.register(
            PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN,
            self.service.cleanup_prepared_agent_run,
        )
        self._global_bus.register(
            PatchouliRoutes.MANUAL_ARCHIVE_TOPIC,
            self._topic_management_service.archive_topic,
        )
        self._global_bus.register(
            PatchouliRoutes.EVICT_TOPIC,
            self._topic_management_service.evict_topic,
        )
        self._global_bus.register(
            PatchouliRoutes.RECORD_MEMORY_CITATION,
            self.service.record_memory_citation,
        )
        self._global_bus.register(
            PatchouliRoutes.WARMUP_MODELS,
            self._model_readiness_service.warmup_models,
        )
        self._global_bus.register(
            PatchouliRoutes.MODELS_READY,
            self._model_readiness_service.is_models_ready,
        )

    def _unregister_public_routes(self) -> None:
        self._global_bus.unregister(PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE)
        self._global_bus.unregister(PatchouliRoutes.SUBMIT_INTERACTION)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_CREATE)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_LIST)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_GET)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_UPDATE)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_DELETE)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_RECORD_FEEDBACK)
        self._global_bus.unregister(PatchouliRoutes.AGENT_PROFILE_CREATE)
        self._global_bus.unregister(PatchouliRoutes.AGENT_PROFILE_LIST)
        self._global_bus.unregister(PatchouliRoutes.TOPIC_LIST_ACTIVE)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_RETRIEVE)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES)
        self._global_bus.unregister(PatchouliRoutes.GET_AGENT_PROFILE)
        self._global_bus.unregister(PatchouliRoutes.PREPARE_AGENT_RUN)
        self._global_bus.unregister(PatchouliRoutes.FINALIZE_AGENT_RUN)
        self._global_bus.unregister(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN)
        self._global_bus.unregister(PatchouliRoutes.MANUAL_ARCHIVE_TOPIC)
        self._global_bus.unregister(PatchouliRoutes.EVICT_TOPIC)
        self._global_bus.unregister(PatchouliRoutes.RECORD_MEMORY_CITATION)
        self._global_bus.unregister(PatchouliRoutes.WARMUP_MODELS)
        self._global_bus.unregister(PatchouliRoutes.MODELS_READY)

    def _register_local_event_bridges(self) -> None:
        self.runtime.local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            self._forward_pending_atom_settled,
        )
        self.runtime.local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            self._forward_pending_atom_failed,
        )

    def _unregister_local_event_bridges(self) -> None:
        self.runtime.local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            self._forward_pending_atom_settled,
        )
        self.runtime.local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            self._forward_pending_atom_failed,
        )

    async def _forward_pending_atom_settled(self, *, settlement) -> None:
        if self._global_bus is None:
            return
        await self._global_bus.publish(
            GlobalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )

    async def _forward_pending_atom_failed(self, *, pending_alias: str) -> None:
        if self._global_bus is None:
            return
        await self._global_bus.publish(
            GlobalEvents.PENDING_ATOM_FAILED,
            pending_alias=pending_alias,
        )


__all__ = [
    "PatchouliSystem",
]
