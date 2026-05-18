"""
帕秋莉子系统 (The Patchouli System / The Facility)

定位：记忆存储、检索与感知子系统 (v4 Phase B/C)
职责：
    - 持有 TheEye (Ingress Gateway): 意图识别、查询重写、话题路由
    - 持有 PatchouliKernel: 记忆内核，管理感知 (Perception)、检索 (Retrieval)、生成 (Generation) 与生命周期 (Lifecycle)
    - 提供记忆能力公开路由 (memory.retrieve, memory.get_memory_by_alias)
    - 实现 SubsystemProtocol 契约

数据流:
    Active: ChatService -> prepare_agent_run (Patchouli) -> run_agent (Alice) -> finalize_agent_run (Patchouli)
    Passive: PassiveIngressService -> ingest_event -> Patchouli (ingest_interaction)

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye (Gateway) ──→ PatchouliKernel   │
    │                        ├── Perception   │
    │                        ├── Retrieval    │
    │                        ├── Generation   │
    │                        └── Lifecycle    │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 4.0
"""

import logging
import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Optional

from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel import PatchouliKernel
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec

if TYPE_CHECKING:
    from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler

logger = logging.getLogger(__name__)


class PatchouliSystem(SubsystemProtocol):
    """
    帕秋莉体系 (The Facility) - HiveMemory 的完整封装 v3.0

    外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliKernel (内核调度器)。
    TheEye 独立于 Kernel 之外，做第一道拦截与信息重整，
    处理完后将标准化请求传入 Kernel 进行调度。

    架构:
        - TheEye (真理之眼): Ingress Gateway，流量入口、意图判断、查询重写
        - PatchouliKernel (帕秋莉内核): 中心调度器
            - RetrievalFamiliar (检索使魔): 上下文检索 (Hot)
            - LibrarianCore (馆长本体): 后台记忆维护 (Cold)

    使用示例:
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>>
        >>> from hivememory.system.config import load_app_config
        >>> config = load_app_config()
        >>> system = PatchouliSystem(config=config)
        >>>
        >>> result = system.process_interaction(
        ...     role="user",
        ...     content="帮我写贪吃蛇游戏",
        ...     context=[],
        ...     user_id="user123"
        ... )
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: Optional[GlobalSystemBus] = None,
        scheduler: Optional["AsyncMaintenanceScheduler"] = None,
    ):
        """
        初始化帕秋莉系统

        Args:
            config: 由上层装配并注入的 HiveMemory 配置

        Examples:
            >>> from hivememory.system.config import load_app_config
            >>> config = load_app_config()
            >>> system = PatchouliSystem(config=config)
        """
        self.config = config

        # 0. 子系统运行时挂载点
        self._local_bus = PatchouliBus()
        self._global_bus = global_bus

        # 1. 初始化 Kernel（内核负责组装 Retrieval + Librarian 运行时）
        self.kernel = PatchouliKernel(config=self.config)

        # 2. 初始化 Gateway
        self._init_gateway()

        # 3. 构建 TheEye (Phase 4.5 Agentic Dispatcher — 仅保留 gaze 职责)
        self.eye = TheEye(
            engine=self._gateway_engine,
        )

        # 4. Patchouli 对外能力门面
        self._service = PatchouliService(
            kernel=self.kernel,
            eye=self.eye,
            global_bus=global_bus,
            local_bus=self._local_bus,
        )

        self._scheduler = scheduler
        self._local_routes_registered = False
        self._public_routes_registered = False
        self._maintenance_registered = False
        self._shutdown_drain_started = False

        logger.info("PatchouliSystem 帕秋莉系统初始化完成")

    def _init_gateway(self) -> None:
        """
        初始化 Gateway 相关基础设施

        Gateway LLM 和 Gateway Engine 属于 TheEye 的依赖，
        独立于 Kernel 管理。
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
            self._gateway_llm_service
        )

        self._gateway_engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

    # ========== 向后兼容属性 ==========

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔"""
        return self.kernel.retrieval_familiar

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问帕秋莉本体"""
        return self.kernel.librarian_core

    @property
    def storage(self):
        """访问存储层（代理到 Kernel）"""
        return self.kernel.storage

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
            self.kernel.librarian_core.perception_layer.scan_idle_buffers_once,
        )
        return True

    def unregister_maintenance_tasks(self, scheduler) -> int:
        """从全局维护器卸载 Patchouli 子系统的维护任务。"""
        return scheduler.unregister_owner(self._MAINTENANCE_OWNER)

    async def start(self) -> None:
        if self._local_bus and not self._local_routes_registered:
            self._register_local_routes()
            self._local_routes_registered = True

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

        await self.shutdown_drain()

        if self._global_bus and self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False

        if self._local_bus and self._local_routes_registered:
            self._unregister_local_routes()
            self._local_routes_registered = False

    async def health(self) -> dict[str, Any]:
        models_ready = self.kernel.is_models_ready()
        return {
            "status": "ok" if models_ready else "warming_up",
            "models_ready": models_ready,
        }

    async def shutdown_drain(self) -> dict[str, Any]:
        """服务关闭前强制归档 perception 中的活跃话题。"""
        if self._shutdown_drain_started:
            logger.info("shutdown drain 已执行，跳过重复调用")
            return {
                "success": True,
                "observer_payloads_submitted": 0,
                "perception": {
                    "success": True,
                    "trigger_reason": "shutdown",
                    "flushed_topics": [],
                    "skipped_topics": [],
                    "archived_blocks": 0,
                },
                "reentrant": True,
            }

        self._shutdown_drain_started = True
        logger.info("开始执行 shutdown drain")

        perception_result = await self.kernel.librarian_core.perception_layer.flush_all_for_shutdown()
        result = {
            "success": True,
            "observer_payloads_submitted": 0,
            "perception": perception_result,
            "reentrant": False,
        }
        logger.info(
            f"shutdown drain 完成: observer_payloads=0, "
            f"flushed_topics={len(perception_result['flushed_topics'])}"
        )
        return result

    def _register_local_routes(self) -> None:
        self._local_bus.register(
            "librarian.submit_interaction",
            self.kernel.librarian_core.submit_interaction,
        )
        self._local_bus.register(
            "passive.analyze_and_retrieve",
            self.service.analyze_and_retrieve,
        )
        self._local_bus.register(
            "memory.retrieve",
            self._retrieve_memories,
        )
        self._local_bus.register(
            "memory.get_memory_by_alias",
            self._get_memory_by_alias,
        )
        self._local_bus.register(
            "librarian.prepare_topic",
            self.kernel.librarian_core.prepare_topic,
        )
        self._local_bus.register(
            "librarian.get_active_topics_snapshots",
            self._get_active_topics_snapshots,
        )
        self._local_bus.register(
            "service.prepare_agent_run",
            self.service.prepare_agent_run,
        )
        self._local_bus.register(
            "service.finalize_agent_run",
            self.service.finalize_agent_run,
        )
        self._local_bus.register(
            "service.cleanup_prepared_agent_run",
            self.service.cleanup_prepared_agent_run,
        )
        self._local_bus.register(
            "librarian.manual_archive_topic",
            self.kernel.librarian_core.manual_archive_topic,
        )

    def _unregister_local_routes(self) -> None:
        self._local_bus.unregister("librarian.submit_interaction")
        self._local_bus.unregister("passive.analyze_and_retrieve")
        self._local_bus.unregister("memory.retrieve")
        self._local_bus.unregister("memory.get_memory_by_alias")
        self._local_bus.unregister("librarian.prepare_topic")
        self._local_bus.unregister("librarian.get_active_topics_snapshots")
        self._local_bus.unregister("service.prepare_agent_run")
        self._local_bus.unregister("service.finalize_agent_run")
        self._local_bus.unregister("service.cleanup_prepared_agent_run")
        self._local_bus.unregister("librarian.manual_archive_topic")

    def _register_public_routes(self) -> None:
        self._global_bus.register(
            PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
            self.service.analyze_and_retrieve,
        )
        self._global_bus.register(
            PatchouliRoutes.SUBMIT_INTERACTION,
            self.kernel.librarian_core.submit_interaction,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_RETRIEVE,
            self._retrieve_memories,
        )
        self._global_bus.register(
            PatchouliRoutes.MEMORY_GET_BY_ALIAS,
            self._get_memory_by_alias,
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
            self.service.manual_archive_topic,
        )

    def _unregister_public_routes(self) -> None:
        self._global_bus.unregister(PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE)
        self._global_bus.unregister(PatchouliRoutes.SUBMIT_INTERACTION)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_RETRIEVE)
        self._global_bus.unregister(PatchouliRoutes.MEMORY_GET_BY_ALIAS)
        self._global_bus.unregister(PatchouliRoutes.PREPARE_AGENT_RUN)
        self._global_bus.unregister(PatchouliRoutes.FINALIZE_AGENT_RUN)
        self._global_bus.unregister(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN)
        self._global_bus.unregister(PatchouliRoutes.MANUAL_ARCHIVE_TOPIC)

    async def _retrieve_memories(self, request, mode: str = "active"):
        return await asyncio.to_thread(
            self.kernel.retrieval_familiar.retrieve,
            request,
            mode,
        )

    async def _get_memory_by_alias(
        self,
        alias: str,
        user_id: str | None = None,
    ):
        result = self.kernel.storage.get_memory_by_alias(alias, user_id)
        if inspect.isawaitable(result):
            return await result
        return result


__all__ = [
    "PatchouliSystem",
]
