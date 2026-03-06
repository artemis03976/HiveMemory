"""
帕秋莉内核 (Patchouli Kernel)

定位：系统的 Orchestrator (编排器) 与 State Manager (状态管理器)
职责：
    - 管理 Retrieval Familiar 和 Librarian Core 两个微服务
    - 维护服务注册表与调度总线
    - 基础设施初始化（存储、LLM、Embedding、Reranker）
    - 引擎构建（Perception、Generation、Lifecycle、Retrieval）

架构定位：
    PatchouliKernel 是星形拓扑的中心节点，独立于 TheEye (Gateway) 之外。
    TheEye 作为 Ingress Controller 在 Kernel 外部运行，处理完请求后
    通过标准化接口将 JobRequest 传入 Kernel。

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye ──→ PatchouliKernel             │
    │              ├── RetrievalFamiliar       │
    │              ├── LibrarianCore           │
    │              └── (Koakuma - 预留)        │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 3.0
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.engines.perception.models import InteractionPayload
from hivememory.patchouli.protocol.models import (
    EyeGazeResult,
    KernelHotResult,
    MTPExecutionResult,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime

if TYPE_CHECKING:
    from hivememory.infrastructure.system_bus import SystemBus

logger = logging.getLogger(__name__)


class PatchouliKernel:
    """
    帕秋莉内核 (Patchouli Kernel) - v3.0

    星形拓扑的中心调度器，管理 Retrieval 和 Librarian 两个微服务。
    不持有 TheEye (Gateway)，TheEye 独立于 Kernel 之外运行。

    职责:
        - 基础设施初始化 (storage, LLM, embedding, reranker)
        - 引擎构建 (perception, generation, lifecycle, retrieval)
        - 服务注册与调度
        - 处理 Eye 传入的热路径/冷路径请求

    使用示例:
        >>> # 推荐：通过 PatchouliSystem 使用（自动组装 Eye + Kernel）
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>> system = PatchouliSystem()
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

        logger.info("PatchouliKernel 帕秋莉内核初始化完成")

    # ========== 基础设施初始化 ==========

    def _init_infrastructure(self) -> None:
        """
        初始化内核基础设施组件（单例服务）

        包含：存储层、感知层 Embedding、Librarian LLM、Reranker
        不包含：Gateway LLM（属于 TheEye 的依赖，由 PatchouliSystem 管理）
        """
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self.config.qdrant,
            embedding_config=self.config.embedding.default,
        )

        from hivememory.infrastructure.embedding import get_perception_embedding_service
        self.perception_embedding_service = get_perception_embedding_service(
            config=self.config.embedding.perception
        )

        from hivememory.infrastructure.llm import get_librarian_llm_service
        self.librarian_llm_service = get_librarian_llm_service(
            config=self.config.llm.librarian
        )

        from hivememory.infrastructure.rerank import get_flag_reranker_service
        reranker_config = self.config.retrieval.retriever.reranker
        if reranker_config.enabled:
            self.reranker_service = get_flag_reranker_service(
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
            embedding_service=self.perception_embedding_service,
            reranker_service=self.reranker_service,
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
        from hivememory.patchouli.config import FullRendererConfig
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
        bus.register("librarian.add_flush_observer", librarian_svc.add_flush_observer)
        bus.register("librarian.get_active_topics_menu", librarian_svc.get_active_topics_menu)

        # --- Retrieval 服务路由 ---
        bus.register("retrieval.retrieve", retrieval_svc.retrieve)

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

    # ========== 公开 API ==========

    def handle_hot(
        self,
        gaze_result: EyeGazeResult,
        enable_retrieval: bool = True,
        mode: str = "active",
    ) -> KernelHotResult:
        """
        处理 Eye 传入的热路径预检索请求

        Args:
            gaze_result: TheEye 的统一输出
            enable_retrieval: 是否执行预检索 (False 时跳过，即使 intent 为 RAG)
            mode: 运行模式 ("active" | "passive")，透传给 RetrievalFamiliar

        Returns:
            KernelHotResult: 热路径处理结果
        """
        # 构建 RetrievalRequest 并调度 Retrieval（热路径）
        retrieved_context = None
        if enable_retrieval:
            retrieval_request = self.build_retrieval_request(gaze_result)
            if retrieval_request:
                if self._bus:
                    retrieved_result = self._bus.request(
                        "retrieval.retrieve", retrieval_request, mode=mode,
                    )
                else:
                    retrieved_result = self.retrieval_familiar.retrieve(
                        retrieval_request, mode=mode,
                    )
                if not retrieved_result.is_empty():
                    retrieved_context = retrieved_result.rendered_context

        return KernelHotResult(
            intent=gaze_result.intent.value,
            rewritten=gaze_result.rewritten_query,
            keywords=gaze_result.search_keywords,
            worth_saving=gaze_result.worth_saving,
            memory=retrieved_context,
        )

    def handle_mtp(
        self,
        assistant_text: str,
    ) -> Optional[MTPExecutionResult]:
        """
        MTP 拦截与执行入口 (Section 3.1)

        供外部 Kernel Loop 调用。当 LLM API 因 stop=["⟫"] 停止时，
        将 assistant 输出传入此方法进行 MTP 指令检测与执行。

        流程 (Section 7.4):
        Phase B (Decision): 捕获 MTP 信号 → 提取指令 Buffer
        Phase C (Execution): 发送给 Koakuma 解析执行
        Phase D (Resume): 返回 XML 结果供 Kernel 追加到 History

        Args:
            assistant_text: LLM 生成的文本 (在 ⟫ 处被截断)

        Returns:
            MTPExecutionResult 如果检测到 MTP 指令，否则 None
        """
        if self._bus:
            return self._bus.request("koakuma.intercept_and_execute", assistant_text)
        return self.koakuma.intercept_and_execute(assistant_text)

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

    def get_mtp_prompt(self) -> str:
        """
        获取 MTP System Prompt 片段 (Section 5.1)

        返回组装好的 MTP 协议教学文本，供 Worker Agent 追加到
        自身的 System Prompt 中。

        片段内容由 KoakumaConfig.mtp_prompt 配置控制:
        - language: 语言 (zh/en)
        - role: Agent 角色 (coder/chat/default)
        - include_demo: 是否包含演示
        - include_kernel_tools: 是否包含工具列表

        Returns:
            str: MTP prompt 片段。MTP 未启用时返回空字符串。
        """
        if not self.config.koakuma.enabled:
            return ""

        prompt_config = self.config.koakuma.mtp_prompt
        if not prompt_config.enabled:
            return ""

        from hivememory.patchouli.prompts.mtp_prompt import MTPPromptBuilder, AgentRole

        builder = MTPPromptBuilder(
            role=AgentRole(prompt_config.role),
            language=prompt_config.language,
            include_demo=prompt_config.include_demo,
            include_error_handling=prompt_config.include_error_handling,
            include_kernel_tools=prompt_config.include_kernel_tools,
        )
        return builder.build()

    # ========== 数据格式转换 ==========

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
            user_id=gaze_result.identity.user_id,
        )

    def flush_buffer(
        self,
        identity: Identity,
    ) -> None:
        """
        用户端手动触发感知层 Flush

        Args:
            identity: 身份标识对象
        """
        if self._bus:
            self._bus.request("perception.flush_buffer", identity=identity)
        else:
            self._engines["perception"].flush_buffer(identity)

    def get_buffer_info(
        self,
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        获取 Buffer 信息

        Args:
            identity: 身份标识对象

        Returns:
            Dict: Buffer 信息字典
        """
        if self._bus:
            return self._bus.request("perception.get_buffer_info", identity=identity)
        return self._engines["perception"].get_buffer_info(identity)

    def add_flush_observer(self, observer) -> None:
        """
        添加 Flush 事件观察者

        Args:
            observer: 观察者回调函数
        """
        if self._bus:
            self._bus.request("librarian.add_flush_observer", observer)
        else:
            self.librarian_core.add_flush_observer(observer)


__all__ = [
    "PatchouliKernel",
]
