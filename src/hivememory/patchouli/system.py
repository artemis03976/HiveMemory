"""
帕秋莉体系 (The Patchouli System)

统一的系统入口，协调 Eye、Familiar、Core 三者交互。
这是用户（开发者）唯一需要 import 的东西。

作者: HiveMemory Team
版本: 2.1
"""

import logging
from typing import List, Optional, Dict, Any

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.protocol.models import Observation

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.librarian_core import LibrarianCore

logger = logging.getLogger(__name__)


class PatchouliSystem:
    """
    帕秋莉体系 - HiveMemory 的完整封装

    这是帕秋莉作为"分布式智能系统"的统一入口。
    用户不需要分别去实例化 Eye 或 Core，只需使用 System 类。

    架构:
        - TheEye (真理之眼): 流量入口、意图判断 (Hot)
        - RetrievalFamiliar (检索使魔): 上下文检索 (Hot)
        - LibrarianCore (馆长本体): 后台记忆维护 (Cold)

    使用示例:
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>>
        >>> # 使用默认配置
        >>> system = PatchouliSystem()
        >>>
        >>> # 处理消息流
        >>> result = system.process_interaction(
        ...     role="user",
        ...     content="帮我写贪吃蛇游戏",
        ...     context=[],
        ...     user_id="user123"
        ... )
        >>>
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
    ):
        """
        初始化帕秋莉体系

        Args:
            config: 完整的 HiveMemory 配置（可选）

        Examples:
            >>> # 使用默认配置（从环境变量和 YAML 加载）
            >>> system = PatchouliSystem()
            >>>
            >>> # 使用自定义配置文件
            >>> from hivememory.patchouli.config import load_app_config
            >>> config = load_app_config("path/to/config.yaml")
            >>> system = PatchouliSystem(config=config)
        """
        # 加载配置
        self.config = config or load_app_config()

        # 1. 初始化基础设施（单例服务）
        self._init_infrastructure()

        # 2. 构建引擎（使用私有构建方法，自底向上装配）
        self._perception_layer = self._build_perception_layer()
        self._generation_engine = self._build_generation_engine()
        self._lifecycle_engine = self._build_lifecycle_engine()
        self._gateway_engine = self._build_gateway_engine()
        self._retrieval_engine = self._build_retrieval_engine()

        # 3. 构建人格层（注入已构建的引擎）
        self.eye = TheEye(
            engine=self._gateway_engine
        )

        self.retrieval_familiar = RetrievalFamiliar(
            storage=self.storage,
            engine=self._retrieval_engine,
        )
        
        self.librarian_core = LibrarianCore(
            storage=self.storage,
            perception_layer=self._perception_layer,
            generation_engine=self._generation_engine,
            lifecycle_engine=self._lifecycle_engine,
        )

        logger.info("PatchouliSystem 帕秋莉初始化完成")

    def _init_infrastructure(self) -> None:
        """
        初始化系统基础设施组件（单例服务）
        """
        # 初始化存储层
        from hivememory.infrastructure.storage import QdrantMemoryStore
        self.storage = QdrantMemoryStore(
            qdrant_config=self.config.qdrant,
            embedding_config=self.config.embedding.default,
        )

        # 初始化感知层 Embedding 服务（用于语义相似度计算）
        from hivememory.infrastructure.embedding import get_perception_embedding_service
        self.perception_embedding_service = get_perception_embedding_service(
            config=self.config.embedding.perception
        )

        # 初始化 Gateway LLM 服务
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self.gateway_llm_service = get_gateway_llm_service(
            config=self.config.llm.gateway
        )

        # 初始化 Librarian LLM 服务
        from hivememory.infrastructure.llm import get_librarian_llm_service
        self.librarian_llm_service = get_librarian_llm_service(
            config=self.config.llm.librarian
        )

        # 初始化 Reranker 服务
        from hivememory.infrastructure.rerank import get_flag_reranker_service
        reranker_config = self.config.retrieval.retriever.reranker
        if reranker_config.enabled and reranker_config.type == "cross_encoder":
            self.reranker_service = get_flag_reranker_service(
                model_name=reranker_config.model_name,
                device=reranker_config.device,
                use_fp16=reranker_config.use_fp16,
            )
        else:
            self.reranker_service = None

    # ========== 私有构建方法 (Private Builders) ==========

    def _build_gateway_engine(self):
        """
        [私有构建器] 构建 Gateway 引擎
        """
        from hivememory.engines.gateway import(
            GatewayEngine,
            BaseInterceptor, create_interceptor,
            BaseSemanticAnalyzer, create_semantic_analyzer,
        )

        config = self.config.gateway
        
        interceptor: BaseInterceptor = create_interceptor(config.interceptor)

        semantic_analyzer: BaseSemanticAnalyzer = create_semantic_analyzer(
            config.analyzer,
            self.gateway_llm_service
        )

        return GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

    def _build_retrieval_engine(self):
        """
        [私有构建器] 构建 Retrieval 引擎
        """
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
        """
        [私有构建器] 组装 Perception 层

        根据配置选择 SemanticFlowPerceptionLayer 或 SimplePerceptionLayer
        """
        from hivememory.engines.perception import create_perception_layer

        return create_perception_layer(
            config=self.config.perception,
            embedding_service=self.perception_embedding_service,
            reranker_service=self.reranker_service,
        )

    def _build_generation_engine(self):
        """
        [私有构建器] 组装 Generation 引擎
        """
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
        """
        [私有构建器] 组装 Lifecycle 模块

        创建所有子组件并注入到管理器
        """
        from hivememory.engines.lifecycle import (
            MemoryLifecycleEngine,
            VitalityCalculator,
            DynamicReinforcementEngine,
            BaseMemoryArchiver, create_archiver,
            BaseGarbageCollector, create_garbage_collector,
        )

        vitality_calculator = VitalityCalculator(self.config.lifecycle.vitality_calculator)

        reinforcement_engine = DynamicReinforcementEngine(
            self.storage,
            vitality_calculator,
            self.config.lifecycle.reinforcement
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

    def process_interaction(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        context: Optional[List[StreamMessage]] = None,
    ) -> Dict[str, Any]:
        """
        统一交互入口 (Unified Interaction Entry)

        自动根据角色分流处理：
        - User: 触发 Hot Path (Eye -> Retrieval) 和 Cold Path (Core.observe)
        - Assistant/System: 仅触发 Cold Path (Core.observe)

        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID
            context: 对话历史上下文 (仅 User 消息需要，用于指代消解)

        Returns:
            Dict: 处理结果
                - intent: 意图 (Chat/RAG/Record)
                - memory: 检索到的记忆 (仅 User RAG)
                - rewritten: 重写后的查询 (仅 User)
        """
        # 创建 Identity 对象
        identity = Identity(user_id=user_id, agent_id=agent_id, session_id=session_id)

        if role == "user":
            return self._process_hot(
                query=content,
                context=context or [],
                identity=identity,
            )
        else:
            self._process_cold(
                role=role,
                content=content,
                identity=identity,
            )
            return {
                "intent": "record_only",
                "memory": None,
                "rewritten": None,
                "worth_saving": True,
            }

    def _process_hot(
        self,
        query: str,
        context: List[StreamMessage],
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        [Hot Path] 热链路处理：Eye -> Retrieval -> Core
        """
        # Step 1: Eye 判断与重写
        retrieval_request, observation = self.eye.gaze(
            query=query,
            context=context,
            identity=identity,
        )

        # Step 2: 投递到 LibrarianCore (使用 unified perceive)
        self.librarian_core.perceive(observation)

        # Step 3: 根据意图决定是否检索记忆
        retrieved_context = None
        if retrieval_request:
            retrieved_result = self.retrieval_familiar.retrieve(retrieval_request)
            if not retrieved_result.is_empty():
                retrieved_context = retrieved_result.rendered_context

        return {
            "intent": observation.gateway_context["intent"],
            "rewritten": observation.anchor,
            "keywords": retrieval_request.keywords if retrieval_request else [],
            "worth_saving": observation.gateway_context["worth_saving"],
            "memory": retrieved_context,
        }

    def _process_cold(
        self,
        role: str,
        content: str,
        identity: Identity,
    ) -> None:
        """
        [Cold Path] 冷链路处理：直接记录
        """
        # 构造 Observation 对象
        observation = Observation(
            role=role,
            raw_message=content,
            identity=identity,
        )

        self.librarian_core.perceive(observation)

    def retrieve(
        self,
        query: str,
        user_id: str,
        **kwargs
    ) -> str:
        """
        直接检索记忆（Hot Path 快捷入口）

        Args:
            query: 查询文本
            user_id: 用户 ID
            **kwargs: 其他检索参数

        Returns:
            str: 渲染后的记忆上下文
        """
        from hivememory.patchouli.protocol.models import RetrievalRequest

        request = RetrievalRequest(
            semantic_query=query,
            user_id=user_id,
            keywords=kwargs.get("keywords", []),
            filters=kwargs.get("filters", {}),
        )

        result = self.retrieval_familiar.retrieve(request)
        return result.rendered_context if not result.is_empty() else ""

    def flush_buffer(
        self,
        identity: Identity,
    ) -> None:
        """
        手动触发感知层 Flush

        Args:
            identity: 身份标识对象
        """
        self.librarian_core.flush_perception(identity)

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
        return self.librarian_core.get_buffer_info(identity)

    def add_flush_observer(self, observer) -> None:
        """
        添加 Flush 事件观察者

        Args:
            observer: 观察者回调函数
        """
        self.librarian_core.add_flush_observer(observer)


__all__ = [
    "PatchouliSystem",
]
