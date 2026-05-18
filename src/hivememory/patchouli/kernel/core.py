"""
帕秋莉内核 (Patchouli Kernel) - v4.0

定位：记忆域 Orchestrator (编排器) 与 State Manager (状态管理器)
职责：
    - 管理 RetrievalFamiliar (检索) 和 LibrarianCore (感知/生成/生命周期)
    - 基础设施初始化（存储、Librarian LLM、Reranker）
    - 引擎构建（Perception、Generation、Lifecycle、Retrieval）

架构定位：
    PatchouliKernel 是记忆域的中心节点，负责与存储层 (Qdrant) 的直接交互。
    计算与智能编排职责已在 Phase C 迁移至 Alice 子系统。

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye ──→ PatchouliKernel             │
    │              ├── RetrievalFamiliar      │
    │              └── LibrarianCore          │
    │                   ├── Perception        │
    │                   ├── Generation        │
    │                   └── Lifecycle         │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 4.0
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from hivememory.core.models import Identity, AgentProfile, OMNI_DOLL_PROFILE
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.core.protocol.models import (
    EyeGazeResult,
    KernelHotResult,
    RetrievalRequest,
)
from hivememory.system.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore

logger = logging.getLogger(__name__)


class PatchouliKernel:
    """
    帕秋莉内核 (Patchouli Kernel) - v3.0

    记忆域运行时装配根，管理 RetrievalFamiliar 与 LibrarianCore 两个核心组件。
    不持有 TheEye (Gateway)，TheEye 独立于 Kernel 之外运行。

    职责:
        - 基础设施初始化 (storage, LLM, embedding, reranker)
        - 引擎构建 (perception, generation, lifecycle, retrieval)
        - 服务注册与持有
        - 处理 Eye 传入的热路径请求

    使用示例:
        >>> # 推荐：通过 PatchouliSystem 使用（自动组装 Eye + Kernel）
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>> from hivememory.system.config import load_app_config
        >>> config = load_app_config()
        >>> system = PatchouliSystem(config=config)
        >>> kernel = system.kernel
        >>>
        >>> # 高级：直接使用 Kernel（需自行管理 Gateway）
        >>> from hivememory.patchouli.kernel import PatchouliKernel
        >>> kernel = PatchouliKernel()
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
    ):
        """
        初始化帕秋莉内核

        Args:
            config: 完整的 HiveMemory 配置（可选）
        """
        self.config = config or load_app_config()

        # 1. 初始化基础设施（单例服务）
        self._init_infrastructure()

        # 2. 构建引擎
        self._engines: Dict[str, Any] = self._build_engines()

        # 3. 注册微服务
        self._services: Dict[str, Any] = {}
        self._register_services()

        logger.info("PatchouliKernel 帕秋莉内核初始化完成")

        # 模型预热状态
        self._models_ready = False

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

    # ========== 基础设施初始化 ==========

    def _init_infrastructure(self) -> None:
        """
        初始化内核基础设施组件（单例服务）

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
        注册微服务到内核

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

    # ========== 智能体配置加载 (供 prepare_agent_run 使用) ==========

    def load_agent_profile(self, agent_alias: str) -> AgentProfile:
        """
        加载人偶图纸配置：storage 冷查询 → omni_doll 兜底

        注意：正式的 Agent 运行时缓存由 Alice 子系统管理。
        此处仅为 prepare_agent_run 提供必要的配置加载能力。

        Args:
            agent_alias: 人偶别名 (如 "coder_doll")

        Returns:
            AgentProfile: 人偶配置（永不返回 None）
        """
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        try:
            atom = self.storage.get_memory_by_alias(agent_alias)
            if atom:
                profile = AgentProfile.from_atom(atom)
                if profile:
                    return profile
        except Exception as e:
            logger.warning(f"Failed to load agent profile '{agent_alias}' from storage: {e}")

        logger.info(f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE.")
        return OMNI_DOLL_PROFILE

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

    # ========== 公开 API ==========

    async def handle_hot(
        self,
        gaze_result: EyeGazeResult,
        enable_retrieval: bool = True,
        mode: str = "active",
    ) -> KernelHotResult:
        retrieved_context = None
        retrieved_memories = []

        if enable_retrieval:
            retrieval_request = self.build_retrieval_request(gaze_result)
            if retrieval_request:
                retrieved_result = await asyncio.to_thread(
                    self.retrieval_familiar.retrieve,
                    retrieval_request,
                    mode,
                )
                if not retrieved_result.is_empty():
                    retrieved_context = retrieved_result.rendered_context
                    retrieved_memories = retrieved_result.memories

        return KernelHotResult(
            intent=gaze_result.intent.value,
            rewritten=gaze_result.rewritten_query,
            keywords=gaze_result.search_keywords,
            worth_saving=gaze_result.worth_saving,
            rendered_memory_context=retrieved_context,
            retrieved_memories=retrieved_memories,
        )

    def build_retrieval_request(
        self, gaze_result: EyeGazeResult
    ) -> Optional[RetrievalRequest]:
        """
        从 EyeGazeResult 构建 RetrievalRequest 协议消息

        只有 RAG 意图才构建检索请求。

        Args:
            gaze_result: TheEye 的统一输出

        Returns:
            RetrievalRequest 如果 intent 是 RAG，否则返回 None
        """
        if gaze_result.intent != GatewayIntent.RAG:
            return None

        return RetrievalRequest(
            semantic_query=gaze_result.rewritten_query,
            keywords=gaze_result.search_keywords,
            identity=gaze_result.identity,
        )

__all__ = [
    "PatchouliKernel",
]
