"""
帕秋莉运行时 (Patchouli Runtime) - v4.0

定位：记忆域 Runtime 宿主 与 State Manager (状态管理器)
职责：
    - 管理 Perception/Retrieval/Generation/Lifecycle 使魔与生成协调器
    - 基础设施初始化（存储、Librarian LLM、Reranker）
    - 引擎构建（Perception、Generation、Lifecycle、Retrieval）
    - 持有 Patchouli local bus 与内部能力路由
    - 提供 warmup / health / shutdown drain 等运行时能力

架构定位：
    PatchouliRuntime 是记忆域的能力运行环境宿主，
    负责与存储层 (Qdrant) 的直接交互。
    Gateway / TheEye 由顶层 System Gateway 托管，Patchouli 只消费其输出。
    计算与智能编排职责已在 Phase C 迁移至 Alice 子系统。

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  PatchouliRuntime                       │
    │      ├── PerceptionFamiliar             │
    │      ├── RetrievalFamiliar              │
    │      ├── GenerationFamiliar             │
    │      ├── GenerationCoordinator          │
    │      └── LifecycleFamiliar              │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 4.0
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from hivememory.core.models import Identity
from hivememory.core.protocol.models import (
    EyeGazeResult,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTaskWaitSummary
from hivememory.patchouli.runtime.route_bindings import build_patchouli_route_bindings
from hivememory.patchouli.runtime.shutdown_drain import (
    shutdown_drain_completed_severity,
    shutdown_drain_completed_status,
    summarize_shutdown_drain_failure,
    summarize_shutdown_drain_result,
)
from hivememory.system.config import PatchouliConfig, SharedConfig
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink
from hivememory.system.runtime.operations import RuntimeOperationObserver

if TYPE_CHECKING:
    from hivememory.patchouli.service import PatchouliService
    from hivememory.patchouli.services.lifecycle import LifecycleFamiliar
    from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
    from hivememory.patchouli.control.memory_generation_coordinator import MemoryGenerationCoordinator
    from hivememory.patchouli.services.perception import PerceptionFamiliar
    from hivememory.patchouli.services.retrieval import RetrievalFamiliar

logger = logging.getLogger(__name__)


class PatchouliRuntime:
    """
    帕秋莉运行时 (Patchouli Runtime) - v4.0

    记忆域运行时装配根，管理感知、检索、生成、生命周期等内部服务。
    不持有 TheEye (Gateway)，TheEye 独立于 Runtime 之外运行。

    职责:
        - 基础设施初始化 (storage, LLM, embedding, reranker)
        - 引擎构建 (perception, generation, lifecycle, retrieval)
        - 服务注册与持有
        - 持有 Patchouli local bus 与内部能力路由
        - 处理 Eye 传入的热路径请求
        - 承担 shutdown drain 运行时行为

    使用示例:
        >>> # 推荐：通过 PatchouliSystem 使用（自动组装 Eye + Runtime）
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>> from hivememory.system.config import load_app_config
        >>> config = load_app_config()
        >>> system = PatchouliSystem(config=config)
        >>> runtime = system.runtime
        >>>
        >>> # 高级：直接使用 Runtime（需自行管理 Gateway）
        >>> from hivememory.patchouli.runtime import PatchouliRuntime
        >>> runtime = PatchouliRuntime()
    """

    def __init__(
        self,
        patchouli_config: PatchouliConfig,
        shared_config: SharedConfig,
        runtime_events: RuntimeEventSink | None = None,
    ):
        self._patchouli_config = patchouli_config
        self._shared_config = shared_config
        self._runtime_events = runtime_events or NullRuntimeEventSink()
        self._local_bus = PatchouliBus()
        self._local_routes_registered = False
        self._shutdown_drain_started = False

        # 1. 初始化基础设施
        self._init_infrastructure()

        # 2. 构建引擎
        self._engines: Dict[str, Any] = self._build_engines()

        # 3. 注册微服务
        self._services: Dict[str, Any] = {}
        self._register_services()

        logger.info("PatchouliRuntime 帕秋莉运行时初始化完成")

        # 模型预热状态
        self._models_ready = False

    @property
    def local_bus(self) -> PatchouliBus:
        return self._local_bus

    @property
    def local_routes_registered(self) -> bool:
        return self._local_routes_registered

    def mount_local_routes(self, service: "PatchouliService") -> None:
        if self._local_routes_registered:
            return

        for route, handler in build_patchouli_route_bindings(self, service):
            self._local_bus.register(route, handler)
        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        if not self._local_routes_registered:
            return

        for route in PatchouliLocalRoutes.ALL:
            self._local_bus.unregister(route)
        self._local_routes_registered = False

    def list_local_routes(self) -> list[str]:
        return self._local_bus.list_routes()

    # ========== 模型预热 ==========

    async def warmup_models(self) -> None:
        """
        后台预热所有推理模型 (Embedding + Reranker)

        在后台线程中触发模型加载，避免首次请求时的延迟。
        服务启动后由 lifespan 异步调用，不阻塞 HTTP 服务可用性。
        """
        import time
        start = time.time()
        logger.info("开始后台预热推理模型...")

        try:
            # 存储层 Embedding (BGE-M3, 用于写入时的向量化)
            await asyncio.to_thread(self.storage.embedding_service.warmup)

            # Reranker
            if self.reranker_service is not None:
                await asyncio.to_thread(self.reranker_service.warmup)

            self._models_ready = True
            elapsed = (time.time() - start) * 1000
            logger.info(f"推理模型预热完成 ({elapsed:.0f}ms)")

        except Exception as e:
            logger.error(f"模型预热失败: {e}", exc_info=True)
            # 预热失败不阻塞服务，退化为首次请求时懒加载
            self._models_ready = False

    def is_models_ready(self) -> bool:
        """
        检查所有推理模型是否已加载就绪

        Returns:
            True 如果所有模型已加载，否则 False
        """
        if self._models_ready:
            return True

        # 动态检查（覆盖懒加载已完成但 warmup 未调用的情况）
        storage_ready = self.storage.embedding_service.is_loaded()
        reranker_ready = (
            self.reranker_service is None
            or self.reranker_service.is_loaded()
        )
        return storage_ready and reranker_ready

    async def shutdown_drain(self) -> dict[str, Any]:
        """
        服务关闭前强制归档活跃话题并等待后台记忆生成任务完成。
        """
        observer = RuntimeOperationObserver(
            self._runtime_events,
            subsystem="patchouli",
            component="patchouli_runtime",
            operation_key="patchouli.shutdown_drain",
            operation_name="shutdown_drain",
            operation_kind="shutdown",
        )
        return await observer.observe(
            self._run_shutdown_drain,
            started_data={"reentrant": self._shutdown_drain_started},
            summarize=summarize_shutdown_drain_result,
            completed_status=shutdown_drain_completed_status,
            completed_severity=shutdown_drain_completed_severity,
            failed_data=summarize_shutdown_drain_failure,
        )

    async def _run_shutdown_drain(self) -> dict[str, Any]:
        if self._shutdown_drain_started:
            logger.info("shutdown drain 已执行，跳过重复调用")
            generation_summary = MemoryGenerationTaskWaitSummary.from_results([])
            return {
                "success": True,
                "perception": {
                    "success": True,
                    "trigger_reason": "shutdown",
                    "flushed_topics": [],
                    "skipped_topics": [],
                    "archived_blocks": 0,
                },
                "generation": generation_summary,
                "reentrant": True,
            }

        self._shutdown_drain_started = True
        logger.info("开始执行 shutdown drain")

        perception_result = await self.perception_familiar.flush_all_for_shutdown()
        generation_result = await self._task_controller.wait_all(
            timeout=(
                self._patchouli_config.shutdown
                .generation_wait_timeout_seconds
            ),
        )
        timed_out_task_ids = [
            result.task_id
            for result in generation_result.results
            if result.timed_out and result.found
        ]
        cancelled_after_timeout = 0
        if timed_out_task_ids:
            cancelled_after_timeout = await self._task_controller.cancel_many(
                timed_out_task_ids,
                reason="shutdown_timeout",
            )
        result = {
            "success": generation_result.timed_out == 0,
            "perception": perception_result,
            "generation": generation_result,
            "generation_cancelled_after_timeout": cancelled_after_timeout,
            "reentrant": False,
        }
        logger.info(
            f"shutdown drain 完成: observer_payloads=0, "
            f"flushed_topics={len(perception_result.flushed_topics)}, "
            f"generation_timed_out={generation_result.timed_out}"
        )
        return result

    # ========== 基础设施初始化 ==========

    def _init_infrastructure(self) -> None:
        """
        初始化运行时基础设施组件（单例服务）

        包含：存储层、Librarian LLM、Reranker、MemoryLibrary
        """
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self._patchouli_config.storage,
            embedding_config=self._shared_config.embedding.default,
        )

        from hivememory.infrastructure.llm import get_librarian_llm_service
        self.librarian_llm_service = get_librarian_llm_service(
            config=self._shared_config.llm.librarian
        )

        from hivememory.infrastructure.rerank import get_fast_embed_reranker_service
        reranker_config = self._patchouli_config.retrieval.retriever.reranker
        if reranker_config.enabled:
            self.reranker_service = get_fast_embed_reranker_service(
                config=reranker_config
            )
        else:
            self.reranker_service = None

        # MemoryLibrary：三层存储协调层，在引擎构建前初始化以便注入
        self.memory_library = self._build_memory_library()

    def _build_memory_library(self):
        """构建 MemoryLibrary（三层存储协调层）"""
        from hivememory.patchouli.memory_library import (
            MemoryLibrary,
            ShortTermMemoryStore,
            MidTermMemoryStore,
            LongTermMemoryStore,
            QdrantStorageAdapter,
            FileBasedStorageAdapter,
        )

        perception_config = self._patchouli_config.perception.engine
        max_resident = getattr(perception_config, "max_resident_topics", 5)

        short_term = ShortTermMemoryStore(max_resident_topics=max_resident)
        mid_term = MidTermMemoryStore(primary=QdrantStorageAdapter(self.storage))

        archiver_config = self._patchouli_config.lifecycle.archiver
        long_term = LongTermMemoryStore(
            port=FileBasedStorageAdapter(
                archive_dir=archiver_config.archive_dir,
                compress=archiver_config.compression,
            )
        )

        artifact_config = self._patchouli_config.artifacts
        artifact_store = None
        if artifact_config.enabled:
            from hivememory.patchouli.memory_library.adapters.artifact import FilesystemArtifactStorageAdapter
            from hivememory.patchouli.memory_library.stores import ArtifactStore
            artifact_store = ArtifactStore(
                FilesystemArtifactStorageAdapter(
                    root_dir=artifact_config.root_dir,
                    max_inline_summary_chars=artifact_config.max_inline_summary_chars,
                )
            )

        return MemoryLibrary(
            short_term=short_term,
            mid_term=mid_term,
            long_term=long_term,
            artifact_store=artifact_store,
        )

    # ========== 引擎构建 ==========

    def _build_engines(self) -> Dict[str, Any]:
        """
        构建所有引擎，返回字典统一管理

        包含：perception, generation, lifecycle, retrieval, artifact
        不包含：gateway（属于 TheEye 的依赖）
        """
        return {
            "perception": self._build_perception_layer(),
            "generation": self._build_generation_engine(),
            "lifecycle": self._build_lifecycle_engine(),
            "retrieval": self._build_retrieval_engine(),
            "artifact": self._build_artifact_engine(),
        }

    def _build_retrieval_engine(self):
        """[私有构建器] 构建 Retrieval 引擎"""
        from hivememory.engines.retrieval import RetrievalEngine, BaseMemoryRetriever, create_retriever

        config = self._patchouli_config.retrieval

        retriever: BaseMemoryRetriever = create_retriever(
            self.memory_library.mid_term,
            config.retriever,
            self.reranker_service
        )

        return RetrievalEngine(retriever=retriever)

    def _build_perception_layer(self):
        """[私有构建器] 组装 Perception 层，注入 MemoryLibrary.short_term"""
        from hivememory.engines.perception import create_perception_layer

        return create_perception_layer(
            config=self._patchouli_config.perception,
            llm_service=self.librarian_llm_service,
            short_term_store=self.memory_library.short_term,
        )

    def _build_generation_engine(self):
        """[私有构建器] 组装 Generation 引擎"""
        from hivememory.engines.generation import (
            MemoryGenerationEngine,
            BaseMemoryExtractor, create_extractor,
            BaseDeduplicator, create_deduplicator,
        )

        config = self._patchouli_config.generation

        extractor: BaseMemoryExtractor = create_extractor(
            config.extractor,
            self.librarian_llm_service,
        )

        deduplicator: BaseDeduplicator = create_deduplicator(
            config.deduplicator
        )

        return MemoryGenerationEngine(
            mid_term=self.memory_library.mid_term,
            extractor=extractor,
            deduplicator=deduplicator,
        )

    def _build_artifact_engine(self):
        """[私有构建器] 组装 ArtifactEngine — store 由 MemoryLibrary 统一持有"""
        from hivememory.engines.artifacts import (
            ArtifactEngine,
            create_document_builder,
            create_interaction_builder,
            create_memory_builder,
        )

        config = self._patchouli_config.artifacts
        store = self.memory_library.artifact_store
        if not config.enabled:
            store = None

        return ArtifactEngine(
            config=config,
            interaction=create_interaction_builder(config.interaction, store),
            document=create_document_builder(config.document, store),
            memory=create_memory_builder(config.memory, store),
        )

    def _build_lifecycle_engine(self):
        """[私有构建器] 组装 Lifecycle 模块"""
        from hivememory.engines.lifecycle import (
            MemoryLifecycleEngine,
            VitalityCalculator,
            DynamicReinforcementEngine,
            BaseGarbageCollector, create_garbage_collector,
        )

        vitality_calculator = VitalityCalculator(
            self._patchouli_config.lifecycle.vitality_calculator
        )

        reinforcement_engine = DynamicReinforcementEngine(
            mid_term=self.memory_library.mid_term,
            config=self._patchouli_config.lifecycle.reinforcement,
            vitality_calculator=vitality_calculator
        )

        garbage_collector: BaseGarbageCollector = create_garbage_collector(
            self.memory_library,
            self._patchouli_config.lifecycle.garbage_collector
        )

        return MemoryLifecycleEngine(
            mid_term=self.memory_library.mid_term,
            vitality_calculator=vitality_calculator,
            reinforcement_engine=reinforcement_engine,
            garbage_collector=garbage_collector,
        )

    # ========== 服务注册 ==========

    def _register_services(self) -> None:
        """
        注册微服务到运行时

        当前注册：perception、retrieval、generation、generation_coordinator、lifecycle。
        MemoryGenerationTaskController 通过 local bus 请求生成执行，不再注入馆长本体。
        """
        from hivememory.patchouli.services.lifecycle import LifecycleFamiliar
        from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
        from hivememory.patchouli.control.memory_generation_coordinator import MemoryGenerationCoordinator
        from hivememory.patchouli.control.memory_generation_tasks import MemoryGenerationTaskController
        from hivememory.patchouli.services.perception import PerceptionFamiliar
        from hivememory.patchouli.services.retrieval import RetrievalFamiliar

        self._services["retrieval"] = RetrievalFamiliar(
            engine=self._engines["retrieval"],
            memory_library=self.memory_library,
            local_bus=self._local_bus,
        )

        self._services["generation"] = MemoryGenerationFamiliar(
            generation_engine=self._engines["generation"],
            memory_library=self.memory_library,
            artifact_engine=self._engines["artifact"],
        )

        self._services["generation_coordinator"] = MemoryGenerationCoordinator(
            bus=self._local_bus,
        )

        self._task_controller = MemoryGenerationTaskController(
            bus=self._local_bus,
            runtime_events=self._runtime_events.scoped(
                "patchouli",
                component="memory_generation_task_controller",
            ),
        )

        self._services["perception"] = PerceptionFamiliar(
            perception_layer=self._engines["perception"],
            bus=self._local_bus,
            config=self._patchouli_config.perception,
            memory_library=self.memory_library,
        )

        self._services["lifecycle"] = LifecycleFamiliar(
            lifecycle_engine=self._engines["lifecycle"],
            memory_library=self.memory_library,
        )

    @property
    def perception_familiar(self) -> PerceptionFamiliar:
        """访问感知使魔服务。"""
        return self._services["perception"]

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔服务"""
        return self._services["retrieval"]

    @property
    def memory_generation_familiar(self) -> MemoryGenerationFamiliar:
        """访问记忆生成使魔服务。"""
        return self._services["generation"]

    @property
    def memory_generation_coordinator(self) -> MemoryGenerationCoordinator:
        """访问记忆生成协调器。"""
        return self._services["generation_coordinator"]

    @property
    def lifecycle_familiar(self) -> LifecycleFamiliar:
        """访问生命周期使魔服务。"""
        return self._services["lifecycle"]

    # ========== 健康检查 ==========

    async def check_storage_health(self) -> bool:
        """
        存储层健康检查

        用于系统级降级判断：如果 Qdrant 不可达，
        在 System Prompt 中注入降级通知，阻止 Agent 发出 MTP 指令。

        Returns:
            bool: True 表示存储可用，False 表示离线
        """
        report = await self.memory_library.check_storage_health()
        if not report.healthy:
            unhealthy = [
                component
                for component in report.components
                if component.required and not component.healthy
            ]
            detail = {component.name: component.detail for component in unhealthy}
            logger.warning(f"Storage health check failed: {detail}")
        return report.healthy

    async def ensure_storage_ready(self) -> None:
        await self.storage.ensure_ready()

__all__ = [
    "PatchouliRuntime",
]
