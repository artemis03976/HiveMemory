"""
帕秋莉内核 (Patchouli Kernel)

定位：系统的 Orchestrator (编排器) 与 State Manager (状态管理器)
职责：
    - 管理 Retrieval Familiar 和 Librarian Core 两个微服务
    - 维护服务注册表与调度总线
    - 基础设施初始化（存储、LLM、Reranker）
    - 引擎构建（Perception、Generation、Lifecycle、Retrieval）

架构定位：
    PatchouliKernel 是星形拓扑的中心节点，独立于 TheEye (Gateway) 之外。
    TheEye 作为 Ingress Controller 在 Kernel 外部运行，处理完请求后
    通过标准化接口将 JobRequest 传入 Kernel。

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye ──→ PatchouliKernel             │
    │              ├── RetrievalFamiliar      │
    │              ├── LibrarianCore          │
    │              └── Koakuma                │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 3.0
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from hivememory.core.models import Identity, MemoryAtom, AgentProfile, OMNI_DOLL_PROFILE
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.protocol.models import (
    EyeGazeResult,
    InteractionPayload,
    KernelHotResult,
    MTPExecutionResult,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.system.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.kernel.runtime.cache import AgentProfileCache

if TYPE_CHECKING:
    from hivememory.infrastructure.system_bus import SystemBus

logger = logging.getLogger(__name__)


class PatchouliKernel:
    """
    帕秋莉内核 (Patchouli Kernel) - v3.0

    星形拓扑的中心调度器，管理 RetrievalFamiliar, LibrarianCore, KoakumaRuntime 三个微服务。
    不持有 TheEye (Gateway)，TheEye 独立于 Kernel 之外运行。

    职责:
        - 基础设施初始化 (storage, LLM, embedding, reranker)
        - 引擎构建 (perception, generation, lifecycle, retrieval)
        - 服务注册与调度
        - 处理 Eye 传入的热路径/冷路径请求

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
        bus: Optional["SystemBus"] = None,
    ):
        """
        初始化帕秋莉内核

        Args:
            config: 完整的 HiveMemory 配置（可选）
            bus: SystemBus 实例（可选），用于模块间通信路由
        """
        self.config = config or load_app_config()
        self._bus = bus

        # 1. 初始化基础设施（单例服务）
        self._init_infrastructure()

        # 2. 构建引擎
        self._engines: Dict[str, Any] = self._build_engines()

        # 3. 注册微服务
        self._services: Dict[str, Any] = {}
        self._register_services()

        # 4. 注册总线路由（如果有 bus）
        if self._bus:
            self._register_bus_routes()

        # 5. 人偶图纸缓存 (多智能体系统)
        self._agent_profile_cache = AgentProfileCache()

        # 6. 帧调度器 (Phase 2 多智能体子代理调用)
        from hivememory.patchouli.kernel.runtime.frame_scheduler import FrameScheduler
        self._frame_scheduler = FrameScheduler(self)

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

        当前注册：retrieval (RetrievalFamiliar), librarian (LibrarianCore), koakuma (KoakumaRuntime)

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
            bus=self._bus,
            lifecycle_engine=self._engines["lifecycle"],
            perception_layer=self._engines["perception"],
            generation_engine=self._engines["generation"],
        )

        # Koakuma (MTP Runtime Service)
        self._services["koakuma"] = KoakumaRuntime(
            bus=self._bus,
            config=self.config.koakuma,
        )

    # ========== 服务访问器 ==========

    def _register_bus_routes(self) -> None:
        """
        在 SystemBus 上注册外部 RPC 路由

        路由命名规范: {service}.{method}
        仅注册外部/分身通信路由，内部模块调用已改为直接调用。

        """
        bus = self._bus
        retrieval_svc = self._services["retrieval"]
        librarian_svc = self._services["librarian"]
        koakuma_svc = self._services["koakuma"]

        # --- Librarian 服务路由（包含感知层代理接口）---
        bus.register("librarian.ingest_interaction", librarian_svc.ingest_interaction)
        bus.register("librarian.manual_trigger", librarian_svc.manual_trigger)
        bus.register("librarian.prepare_topic", librarian_svc.prepare_topic)
        bus.register("librarian.get_active_topics_snapshots", librarian_svc.get_active_topics_snapshots)

        # --- Retrieval 服务路由 ---
        bus.register("retrieval.retrieve", retrieval_svc.retrieve)

        # --- Storage 服务路由 ---
        bus.register("storage.get_memory", self.storage.get_memory)
        bus.register("storage.get_memory_by_alias", self.storage.get_memory_by_alias)

        # --- Koakuma 服务路由 ---
        bus.register("koakuma.intercept_and_execute", koakuma_svc.intercept_and_execute)

        logger.info(
            f"SystemBus 路由注册完成: {len(bus.list_routes())} 条路由"
        )

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔服务"""
        return self._services["retrieval"]

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问馆长本体服务"""
        return self._services["librarian"]

    @property
    def koakuma(self) -> KoakumaRuntime:
        """访问小恶魔 MTP 运行时服务"""
        return self._services["koakuma"]

    # ========== 多智能体调度 (Phase 1) ==========

    @property
    def agent_profile_cache(self) -> AgentProfileCache:
        """访问人偶图纸缓存"""
        return self._agent_profile_cache

    @property
    def frame_scheduler(self) -> "FrameScheduler":
        """访问帧调度器 (Phase 2 多智能体子代理调用)"""
        return self._frame_scheduler

    def load_agent_profile(self, agent_alias: str) -> AgentProfile:
        """
        加载人偶图纸配置：缓存优先 → storage 冷查询 → omni_doll 兜底

        Args:
            agent_alias: 人偶别名 (如 "coder_doll")

        Returns:
            AgentProfile: 人偶配置（永不返回 None）
        """
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        profile = self._agent_profile_cache.load(agent_alias, self.storage)
        if profile is not None:
            return profile

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
                if self._bus:
                    retrieved_result = await self._bus.async_request(
                        "retrieval.retrieve", retrieval_request, mode=mode,
                    )
                else:
                    retrieved_result = await asyncio.to_thread(
                        self.retrieval_familiar.retrieve,
                        retrieval_request,
                        mode,
                    )
                if not retrieved_result.is_empty():
                    retrieved_context = retrieved_result.rendered_context
                    retrieved_memories = retrieved_result.memories

                    # 将预检索记忆的别名注册到 Koakuma 的 L1 热映射，
                    # 使 Agent 在看到渲染结果后可直接用别名发起 MTP READ
                    self._register_preretrieval_aliases(retrieved_result.memories)

        return KernelHotResult(
            intent=gaze_result.intent.value,
            rewritten=gaze_result.rewritten_query,
            keywords=gaze_result.search_keywords,
            worth_saving=gaze_result.worth_saving,
            rendered_memory_context=retrieved_context,
            retrieved_memories=retrieved_memories,
        )
 
    async def handle_mtp(
        self,
        assistant_text: str,
    ) -> Optional[MTPExecutionResult]:
        if self._bus:
            return await self._bus.async_request(
                "koakuma.intercept_and_execute", assistant_text
            )
        return await asyncio.to_thread(
            self.koakuma.intercept_and_execute,
            assistant_text,
        )

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        target_topic: str = "NEW_TOPIC",
    ) -> None:
        """
        提交交互载荷到 Librarian (阻塞等待)

        并发范式 (参考 Concurrent.md):
            这是冷链路入口，必须阻塞等待完成。
            确保 token 溢出压缩等操作完成后再处理下一波请求。

        Args:
            payload: Kernel → Perception 的原子传输包
            target_topic: 路由目标话题 ID 或 "NEW_TOPIC" (由 TheEye 决定)
        """
        try:
            if self._bus:
                await self._bus.async_request("librarian.ingest_interaction", payload, target_topic)
            else:
                await self.librarian_core.ingest_interaction(payload, target_topic)
        except Exception as e:
            logger.warning(f"Interaction submission failed: {e}")

    def get_mtp_prompt(
        self,
        profile: Optional[AgentProfile] = None,
    ) -> str:
        """
        获取 MTP 协议教学 System Prompt 片段

        仅包含 MTP 协议语法教学，不包含角色设定（persona）。
        当传入 AgentProfile 时，根据权限白名单动态过滤可用指令和工具。

        Args:
            profile: 人偶图纸配置（可选），用于权限过滤

        Returns:
            str: MTP prompt 片段。MTP 未启用时返回空字符串。
        """
        if not self.config.koakuma.enabled:
            return ""

        prompt_config = self.config.koakuma.mtp_prompt
        if not prompt_config.enabled:
            return ""

        from hivememory.prompts.mtp import MTPPromptBuilder

        # 从 profile 提取权限过滤参数
        # None = 全部允许, [] = 禁止所有, [...] = 白名单
        allowed_verbs = None
        allowed_kernel_tools = None
        if profile and profile.allowed_mtp_verbs is not None:
            allowed_verbs = profile.allowed_mtp_verbs
        if profile and profile.allowed_sys_tools is not None:
            allowed_kernel_tools = profile.allowed_sys_tools

        builder = MTPPromptBuilder(
            language=prompt_config.language,
            include_demo=prompt_config.include_demo,
            include_error_handling=prompt_config.include_error_handling,
            allowed_verbs=allowed_verbs,
            allowed_kernel_tools=allowed_kernel_tools,
        )
        return builder.build()

    def _register_preretrieval_aliases(self, memories: List[MemoryAtom]) -> None:
        """
        将预检索记忆的完整原子注册到 Koakuma 缓存

        预检索注入的记忆上下文使用别名作为 id，Agent 看到后会直接
        用别名发起 MTP READ。此方法确保这些别名在 Koakuma 中可解析。

        Args:
            memories: 预检索返回的 MemoryAtom 列表
        """
        self.koakuma.atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug(
                f"预检索记忆缓存完成: {len(memories)} 条记忆已缓存到 Koakuma"
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

    # ========== 总线方法代理 ==========

    async def manual_trigger(self, topic_id: Optional[str] = None) -> Dict[str, Any]:
        """
        手动触发话题结算 (Archive + Compact)

        用户主动保存当前对话状态。语义为"立即归档 + 生成摘要并保留内存"。
        话题不会被驱逐，可以继续接收新的交互。

        Args:
            topic_id: 目标话题 ID。如果为 None，使用最后活跃的话题。

        Returns:
            Dict: 包含 success, topic_id, message, blocks_archived 的结果字典
        """
        if self._bus:
            return await self._bus.async_request("librarian.manual_trigger", topic_id)
        return await self.librarian_core.manual_trigger(topic_id)

    async def get_topic_snapshots(self, identity: "Identity") -> List:
        """
        获取活跃话题快照列表

        从感知层获取所有活跃话题的快照，包含每个话题的最后一轮对话。

        Args:
            identity: 用户身份标识

        Returns:
            List[TopicSnapshot]: 话题快照列表
        """
        if self._bus:
            return await self._bus.async_request(
                "librarian.get_active_topics_snapshots",
                identity=identity,
            )
        return self.librarian_core.get_active_topics_snapshots(identity)

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: "Identity",
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        预创建/刷新话题，同时获取话题上下文

        Returns:
            (real_topic_id, pool_snapshot, topic_context)
        """
        if self._bus:
            return await self._bus.async_request(
                "librarian.prepare_topic",
                target_topic_id, new_topic_title, new_topic_summary, identity,
            )
        return await self.librarian_core.prepare_topic(
            target_topic_id, new_topic_title, new_topic_summary, identity
        )


__all__ = [
    "PatchouliKernel",
]
