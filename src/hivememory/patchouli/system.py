"""
帕秋莉子系统 (The Patchouli System / The Facility)

定位：记忆存储、检索与感知子系统 (v4 Phase B/C)
职责：
    - 持有 PatchouliRuntime: 记忆运行时，管理感知 (Perception)、检索 (Retrieval)、生成 (Generation) 与生命周期 (Lifecycle)
    - 消费 System Gateway 注入的 gaze 能力
    - 提供记忆能力公开路由 (memory.retrieve, memory.retrieve_by_aliases)
    - 实现 SubsystemProtocol 契约

数据流:
    Active: ChatService -> prepare_agent_run (Patchouli) -> run_agent (Alice) -> finalize_agent_run (Patchouli)
    Passive: PassiveIngressService -> ingest_event -> Patchouli (submit_interaction)

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  System Gateway gaze ─→ PatchouliRuntime│
    │                         ├── Perception  │
    │                         ├── Retrieval   │
    │                         ├── Generation  │
    │                         └── Lifecycle   │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 4.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hivememory.patchouli.application import (
    AgentProfileManagementService,
    MemoryManagementService,
    MemoryTaskManagementService,
    ModelReadinessService,
    TopicManagementService,
)
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.runtime import PatchouliRuntime
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
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

    外层容器，持有 PatchouliRuntime (运行时宿主)，并消费 System Gateway
    注入的 gaze callable。入口分析完成后，标准化请求传入 Runtime / Service
    进行后续记忆处理。

    架构:
        - PatchouliRuntime (帕秋莉运行时): 记忆域运行时宿主
            - PerceptionFamiliar (感知使魔): 话题缓冲与结算触发
            - RetrievalFamiliar (检索使魔): 上下文检索 (Hot)
            - MemoryGenerationFamiliar / Coordinator: 记忆生成执行与编排
            - LifecycleFamiliar (生命周期使魔): 活力维护与园艺任务
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: GlobalSystemBus | None = None,
        scheduler: AsyncMaintenanceScheduler | None = None,
        runtime_events: RuntimeEventSink | None = None,
    ):
        self.config = config
        self._global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        # 1. 初始化 Runtime（运行时负责组装感知、检索、生成、生命周期组件图）
        self.runtime = PatchouliRuntime(
            patchouli_config=self.config.patchouli,
            shared_config=self.config.shared,
            runtime_events=self._runtime_events,
        )

        self._interaction_submission_queue = InteractionSubmissionQueue(
            self.runtime.perception_familiar.submit_interaction,
            runtime_events=self._runtime_events.scoped(
                "patchouli",
                component="interaction_submission_queue",
            ),
        )

        # 2. Patchouli 对外能力门面。Active/Passive 共用同一条 interaction lane。
        self._service = PatchouliService(
            bus=self.runtime.local_bus,
            interaction_queue=self._interaction_submission_queue,
            memory_compiler_config=self.config.memory_compiler,
            pending_atom_settler=self.runtime.pending_atom_settler,
        )
        self._memory_management_service = MemoryManagementService(
            bus=self.runtime.local_bus,
        )
        self._memory_task_management_service = MemoryTaskManagementService(
            bus=self.runtime.local_bus,
        )
        self._agent_profile_management_service = AgentProfileManagementService(
            bus=self.runtime.local_bus,
        )
        self._topic_management_service = TopicManagementService(
            bus=self.runtime.local_bus,
        )
        self._model_readiness_service = ModelReadinessService(
            bus=self.runtime.local_bus,
        )
        self._public_api = PatchouliPublicApi(
            chat=self._service,
            memory=self._memory_management_service,
            memory_tasks=self._memory_task_management_service,
            agent_profiles=self._agent_profile_management_service,
            topics=self._topic_management_service,
            readiness=self._model_readiness_service,
        )
        self._bridge = PatchouliBridge(
            local_bus=self.runtime.local_bus,
            global_bus=global_bus,
            public_api=self._public_api,
        )
        self._scheduler = scheduler
        self._maintenance_registered = False

        logger.info("PatchouliSystem 帕秋莉系统初始化完成")

    @property
    def service(self) -> PatchouliService:
        """访问 Patchouli 对外能力门面。"""
        return self._service

    @property
    def interaction_submission_queue(self) -> InteractionSubmissionQueue:
        """访问 active/passive 共用的 interaction submission queue。"""
        return self._interaction_submission_queue

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
        await self._interaction_submission_queue.start()
        await self.runtime.start_memory_generation_queue()

        if self._scheduler and not self._maintenance_registered:
            self._maintenance_registered = self.register_maintenance_tasks(self._scheduler)

    async def stop(self) -> None:
        if self._scheduler and self._maintenance_registered:
            self.unregister_maintenance_tasks(self._scheduler)
            self._maintenance_registered = False

        # interaction work 不可取消；先等待所有已接纳 work 和 Active continuation
        # 完成，再进入 perception flush，避免 handler 晚于 shutdown settlement 写入。
        await self._interaction_submission_queue.drain_all(timeout=None)
        await self.service.drain_active_finalizations()
        await self._interaction_submission_queue.stop()
        await self.runtime.shutdown_drain()
        await self.runtime.stop_memory_generation_queue()

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
