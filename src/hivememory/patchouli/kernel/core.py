"""
帕秋莉运行时 (Patchouli Runtime) - v4.0

定位：记忆域 Runtime 宿主 与 State Manager (状态管理器)
职责：
    - 管理 RetrievalFamiliar (检索) 和 LibrarianCore (感知/生成/生命周期)
    - 基础设施初始化（存储、Librarian LLM、Reranker）
    - 引擎构建（Perception、Generation、Lifecycle、Retrieval）
    - 持有 Patchouli local bus 与内部能力路由
    - 提供 warmup / health / shutdown drain 等运行时能力

架构定位：
    PatchouliRuntime 是记忆域的能力运行环境宿主，
    负责与存储层 (Qdrant) 的直接交互。
    计算与智能编排职责已在 Phase C 迁移至 Alice 子系统。

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye ──→ PatchouliRuntime            │
    │               ├── RetrievalFamiliar     │
    │               └── LibrarianCore         │
    │                    ├── Perception       │
    │                    ├── Generation       │
    │                    └── Lifecycle        │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 4.0
"""

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from hivememory.core.models import Identity
from hivememory.core.protocol.models import (
    EyeGazeResult,
)
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.system.config import HiveMemoryConfig, load_app_config

if TYPE_CHECKING:
    from hivememory.patchouli.service import PatchouliService

logger = logging.getLogger(__name__)

_LOCAL_ROUTE_NAMES = (
    "librarian.submit_interaction",
    "passive.analyze_and_retrieve",
    "memory.retrieve",
    "memory.get_memory_by_alias",
    "memory.get_agent_profile",
    "librarian.prepare_topic",
    "librarian.get_active_topics_snapshots",
    "service.prepare_agent_run",
    "service.finalize_agent_run",
    "service.cleanup_prepared_agent_run",
    "librarian.manual_archive_topic",
)


class PatchouliRuntime:
    """
    帕秋莉运行时 (Patchouli Runtime) - v4.0

    记忆域运行时装配根，管理 RetrievalFamiliar 与 LibrarianCore 两个核心组件。
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
        >>> from hivememory.patchouli.kernel import PatchouliRuntime
        >>> runtime = PatchouliRuntime()
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
    ):
        """
        初始化帕秋莉运行时

        Args:
            config: 完整的 HiveMemory 配置（可选）
        """
        self.config = config or load_app_config()
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

        self._local_bus.register(
            "librarian.submit_interaction",
            self.librarian_core.submit_interaction,
        )
        self._local_bus.register(
            "passive.analyze_and_retrieve",
            service.analyze_and_retrieve,
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
            "memory.get_agent_profile",
            self._get_agent_profile,
        )
        self._local_bus.register(
            "librarian.prepare_topic",
            self.librarian_core.prepare_topic,
        )
        self._local_bus.register(
            "librarian.get_active_topics_snapshots",
            self.librarian_core.get_active_topics_snapshots,
        )
        self._local_bus.register(
            "service.prepare_agent_run",
            service.prepare_agent_run,
        )
        self._local_bus.register(
            "service.finalize_agent_run",
            service.finalize_agent_run,
        )
        self._local_bus.register(
            "service.cleanup_prepared_agent_run",
            service.cleanup_prepared_agent_run,
        )
        self._local_bus.register(
            "librarian.manual_archive_topic",
            self.librarian_core.manual_archive_topic,
        )
        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        if not self._local_routes_registered:
            return

        for route in _LOCAL_ROUTE_NAMES:
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

        perception_result = await self.librarian_core.perception_layer.flush_all_for_shutdown()
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

    # ========== 基础设施初始化 ==========

    def _init_infrastructure(self) -> None:
        """
        初始化运行时基础设施组件（单例服务）

        包含：存储层、Librarian LLM、Reranker
        不包含：Gateway LLM（属于 TheEye 的依赖，由 PatchouliSystem 管理）
        不包含：感知层 Embedding（感知层已不再使用 Embedding）
        """
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self.config.qdrant,
            embedding_config=self.config.embedding.default,
        )

        from hivememory.infrastructure.llm import get_librarian_llm_service
        self.librarian_llm_service = get_librarian_llm_service(
            config=self.config.llm.librarian
        )

        from hivememory.infrastructure.rerank import get_fast_embed_reranker_service
        reranker_config = self.config.retrieval.retriever.reranker
        if reranker_config.enabled:
            self.reranker_service = get_fast_embed_reranker_service(
                config=reranker_config
            )
        else:
            self.reranker_service = None

    # ========== 引擎构建 ==========

    def _build_engines(self) -> Dict[str, Any]:
        """
        构建所有引擎，返回字典统一管理

        包含：perception, generation, lifecycle, retrieval
        不包含：gateway（属于 TheEye 的依赖）
        """
        return {
            "perception": self._build_perception_layer(),
            "generation": self._build_generation_engine(),
            "lifecycle": self._build_lifecycle_engine(),
            "retrieval": self._build_retrieval_engine(),
        }

    def _build_retrieval_engine(self):
        """[私有构建器] 构建 Retrieval 引擎"""
        from hivememory.engines.retrieval import (
            RetrievalEngine,
            BaseMemoryRetriever, create_retriever,
            BaseContextRenderer, create_renderer,
        )

        config = self.config.retrieval

        retriever: BaseMemoryRetriever = create_retriever(
            self.storage,
            config.retriever,
            self.reranker_service
        )

        renderer: BaseContextRenderer = create_renderer(config.renderer)

        return RetrievalEngine(
            retriever=retriever,
            renderer=renderer,
        )

    def _build_perception_layer(self):
        """[私有构建器] 组装 Perception 层"""
        from hivememory.engines.perception import create_perception_layer

        return create_perception_layer(
            config=self.config.perception,
            llm_service=self.librarian_llm_service,
        )

    def _build_generation_engine(self):
        """[私有构建器] 组装 Generation 引擎"""
        from hivememory.engines.generation import (
            MemoryGenerationEngine,
            BaseMemoryExtractor, create_extractor,
            BaseDeduplicator, create_deduplicator,
        )

        config = self.config.generation

        extractor: BaseMemoryExtractor = create_extractor(
            config.extractor,
            self.librarian_llm_service
        )

        deduplicator: BaseDeduplicator = create_deduplicator(
            self.storage,
            config.deduplicator
        )

        return MemoryGenerationEngine(
            storage=self.storage,
            extractor=extractor,
            deduplicator=deduplicator,
        )

    def _build_lifecycle_engine(self):
        """[私有构建器] 组装 Lifecycle 模块"""
        from hivememory.engines.lifecycle import (
            MemoryLifecycleEngine,
            VitalityCalculator,
            DynamicReinforcementEngine,
            BaseMemoryArchiver, create_archiver,
            BaseGarbageCollector, create_garbage_collector,
        )

        vitality_calculator = VitalityCalculator(
            self.config.lifecycle.vitality_calculator
        )

        reinforcement_engine = DynamicReinforcementEngine(
            storage=self.storage,
            config=self.config.lifecycle.reinforcement,
            vitality_calculator=vitality_calculator
        )

        archiver: BaseMemoryArchiver = create_archiver(
            self.storage,
            self.config.lifecycle.archiver
        )

        garbage_collector: BaseGarbageCollector = create_garbage_collector(
            self.storage,
            archiver,
            vitality_calculator,
            self.config.lifecycle.garbage_collector
        )

        return MemoryLifecycleEngine(
            storage=self.storage,
            vitality_calculator=vitality_calculator,
            reinforcement_engine=reinforcement_engine,
            archiver=archiver,
            garbage_collector=garbage_collector,
        )

    # ========== 服务注册 ==========

    def _register_services(self) -> None:
        """
        注册微服务到运行时

        当前注册：retrieval (RetrievalFamiliar), librarian (LibrarianCore)
        """
        # 构建被动模式渲染器 (Passive.md §5.2)
        from hivememory.engines.retrieval.renderer import FullContextRenderer
        from hivememory.system.config import FullRendererConfig
        passive_renderer = FullContextRenderer(FullRendererConfig())

        self._services["retrieval"] = RetrievalFamiliar(
            storage=self.storage,
            engine=self._engines["retrieval"],
            passive_renderer=passive_renderer,
        )

        self._services["librarian"] = LibrarianCore(
            storage=self.storage,
            lifecycle_engine=self._engines["lifecycle"],
            perception_layer=self._engines["perception"],
            generation_engine=self._engines["generation"],
        )

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔服务"""
        return self._services["retrieval"]

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问馆长本体服务"""
        return self._services["librarian"]

    # ========== 健康检查 ==========

    def check_storage_health(self) -> bool:
        """
        存储层健康检查

        用于系统级降级判断：如果 Qdrant 不可达，
        在 System Prompt 中注入降级通知，阻止 Agent 发出 MTP 指令。

        Returns:
            bool: True 表示存储可用，False 表示离线
        """
        try:
            self.storage.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")
            return False

    async def _retrieve_memories(self, request, mode: str = "active"):
        return await asyncio.to_thread(
            self.retrieval_familiar.retrieve,
            request,
            mode,
        )

    async def _get_memory_by_alias(
        self,
        alias: str,
        user_id: str | None = None,
    ):
        result = self.storage.get_memory_by_alias(alias, user_id)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _get_agent_profile(self, agent_alias: str):
        result = self.storage.get_agent_profile(agent_alias)
        if inspect.isawaitable(result):
            return await result
        return result


__all__ = [
    "PatchouliRuntime",
]
