"""
HiveMemory - 分布式记忆管理系统

帕秋莉体系 v3.0+:
    - PatchouliSystem (The Facility): 记忆域容器，持有 Patchouli Runtime
    - GatewaySystem: Gateway 子系统，承载决策工作流入口
    - PatchouliRuntime (帕秋莉运行时): 中心调度器，管理微服务
        - PerceptionFamiliar (感知使魔): 话题缓冲、归档触发
        - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染
        - MemoryGenerationFamiliar / Coordinator: 记忆生成执行与编排
        - LifecycleFamiliar (生命周期使魔): 活力维护、园艺任务

使用示例:
    >>> from hivememory import PatchouliSystem, HiveMemoryConfig
    >>>
    >>> config = load_app_config()
    >>> system = PatchouliSystem(config)
    >>> result = system.process_interaction(
    ...     role="user",
    ...     content="我之前设置的 API Key 是什么？",
    ...     context=[],
    ...     user_id="user123"
    ... )

作者: HiveMemory Team
版本: 3.0
"""

from hivememory.core.models import (
    ActionReducer,
    AgentAction,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    VerificationStatus,
    Identity,
    StreamMessage,
    StreamMessageType,
    TurnEvent,
    TurnRecord,
    MetaData,
    IndexLayer,
    PayloadLayer,
    Artifacts,
    RelationLayer,
)

from hivememory.system.config import (
    load_app_config,
    HiveMemoryConfig,
    SystemCommandConfig,
    SystemGatewayConfig,
    MemoryPerceptionConfig,
    MemoryGenerationConfig,
    MemoryRetrievalConfig,
    MemoryLifecycleConfig,
    LLMConfig,
    EmbeddingConfig,
    QdrantConfig,
)

from hivememory.infrastructure.llm import (
    BaseLLMService,
    LiteLLMService,
    get_gateway_llm_service,
    get_librarian_llm_service,
)
from hivememory.infrastructure.embedding import (
    BaseEmbeddingService,
    BGEM3EmbeddingService,
    get_embedding_service,
    get_bge_m3_service,
    get_default_embedding_service,
    get_perception_embedding_service,
)
from hivememory.infrastructure.rerank import (
    BaseRerankService,
    FastEmbedRerankerService,
    get_fast_embed_reranker_service,
)
from hivememory.infrastructure.storage import QdrantMemoryStore

from hivememory.utils import (
    TimeFormatter,
    Language,
    format_time_ago,
    LLMJSONParser,
    JSONParseError,
    parse_llm_json,
    parse_llm_json_many,
    safe_parse_llm_json,
    TokenEstimator,
    EstimationStrategy,
    estimate_tokens,
)

from hivememory.engines.gateway import (
    GatewayEngine,
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    SemanticAnalysisResult,
    BaseInterceptor,
    BaseSemanticAnalyzer,
    RuleInterceptor,
    create_interceptor,
    LLMAnalyzer,
    create_semantic_analyzer,
)

from hivememory.system.application.passive import MessageBufferState

from hivememory.engines.generation import (
    DuplicateDecision,
    MemoryGenerationEngine,
    BaseMemoryExtractor,
    BaseDeduplicator,
    ExtractedMemoryDraft,
    GenerationRequest,
    MergeResult,
    LLMMemoryExtractor,
    NoOpMemoryExtractor,
    create_extractor,
    MemoryDeduplicator,
    NoOpDeduplicator,
    create_deduplicator,
)
# WriteFocus / UpdateFocus are shared core DTOs; DuplicateDecision belongs to generation.
from hivememory.core.models import (
    UpdateFocus,
    WriteFocus,
)

from hivememory.engines.retrieval import (
    RetrievalEngine,
    RetrievalQuery,
    RetrievalResult,
    SearchResult,
    SearchResults,
    BaseMemoryRetriever,
    BaseReranker,
    BaseFusion,
    FilterConverter,
    QdrantFilterConverter,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    ReciprocalRankFusion,
    AdaptiveWeightedFusion,
    NoopReranker,
    CrossEncoderReranker,
    create_retriever,
)

from hivememory.engines.lifecycle import (
    MemoryLifecycleEngine,
    BaseGarbageCollector,
    EventType,
    ReinforcementResult,
    MemoryEvent,
    ArchiveStatus,
    ArchiveRecord,
    VitalityCalculator,
    DynamicReinforcementEngine,
    PeriodicGarbageCollector,
    create_garbage_collector,
)

from hivememory.engines.perception import (
    SemanticFlowPerceptionLayer,
    BasePerceptionLayer,
    TraceItem,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    FlushEvent,
    FlushReason,
    TriggerManager,
    DECISION_MATRIX,
    BaseRelayController,
    NoOpRelayController,
    SimpleRelayController,
    LLMRelayController,
    NullPerceptionLayer,
    create_relay_controller,
    create_perception_layer,
)
from hivememory.core.protocol import InteractionPayload

from hivememory.server.models import (
    ErrorResponse,
    HealthResponse,
    ChatRequest,
    ChatTokenEvent,
    MTPStartEvent,
    MTPResultEvent,
    TopicInfoEvent,
    ChatDoneEvent,
    ChatErrorEvent,
    PassiveIngressRequest,
    PassiveIngressResponse,
    MemoryResponse,
    MemoryListResponse,
    TopicSnapshotResponse,
    TopicListResponse,
    TriggerResponse,
)


def __getattr__(name: str):
    """懒加载以避免循环导入"""
    if name == "GatewaySystem":
        from hivememory.gateway import GatewaySystem
        return GatewaySystem
    if name == "GatewayRuntime":
        from hivememory.gateway import GatewayRuntime
        return GatewayRuntime
    if name == "GatewayService":
        from hivememory.gateway import GatewayService
        return GatewayService
    if name == "GatewayState":
        from hivememory.gateway import GatewayState
        return GatewayState
    if name == "PatchouliPrepareDecision":
        from hivememory.gateway import PatchouliPrepareDecision
        return PatchouliPrepareDecision
    if name == "PatchouliRuntime":
        from hivememory.patchouli.runtime import PatchouliRuntime
        return PatchouliRuntime
    if name == "PatchouliService":
        from hivememory.patchouli.service import PatchouliService
        return PatchouliService
    if name == "PatchouliSystem":
        from hivememory.patchouli.system import PatchouliSystem
        return PatchouliSystem
    if name == "RetrievalFamiliar":
        from hivememory.patchouli.services.retrieval import RetrievalFamiliar
        return RetrievalFamiliar
    if name == "PerceptionFamiliar":
        from hivememory.patchouli.services.perception import PerceptionFamiliar
        return PerceptionFamiliar
    if name == "LifecycleFamiliar":
        from hivememory.patchouli.services.lifecycle import LifecycleFamiliar
        return LifecycleFamiliar
    if name == "MemoryGenerationFamiliar":
        from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
        return MemoryGenerationFamiliar
    if name == "MemoryGenerationCoordinator":
        from hivememory.patchouli.control.memory_generation_coordinator import MemoryGenerationCoordinator
        return MemoryGenerationCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # ========== 核心模型 ==========
    "MemoryAtom",
    "MemoryType",
    "MemoryVisibility",
    "VerificationStatus",
    "ActionReducer",
    "TurnEvent",
    "AgentAction",
    "TurnRecord",
    "Identity",
    "StreamMessage",
    "StreamMessageType",
    "MetaData",
    "IndexLayer",
    "PayloadLayer",
    "Artifacts",
    "RelationLayer",
    # ========== 配置 ==========
    "load_app_config",
    "HiveMemoryConfig",
    "SystemCommandConfig",
    "SystemGatewayConfig",
    "MemoryPerceptionConfig",
    "MemoryGenerationConfig",
    "MemoryRetrievalConfig",
    "MemoryLifecycleConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "QdrantConfig",
    # ========== LLM 服务 ==========
    "BaseLLMService",
    "LiteLLMService",
    "get_gateway_llm_service",
    "get_librarian_llm_service",
    # ========== Embedding 服务 ==========
    "BaseEmbeddingService",
    "BGEM3EmbeddingService",
    "get_embedding_service",
    "get_bge_m3_service",
    "get_default_embedding_service",
    "get_perception_embedding_service",
    # ========== Rerank 服务 ==========
    "BaseRerankService",
    "FastEmbedRerankerService",
    "get_fast_embed_reranker_service",
    # ========== 存储 ==========
    "QdrantMemoryStore",
    # ========== 工具 ==========
    "TimeFormatter",
    "Language",
    "format_time_ago",
    "LLMJSONParser",
    "JSONParseError",
    "parse_llm_json",
    "parse_llm_json_many",
    "safe_parse_llm_json",
    "TokenEstimator",
    "EstimationStrategy",
    "estimate_tokens",
    # ========== Gateway Engine ==========
    "GatewayEngine",
    "GatewayIntent",
    "GatewayResult",
    "InterceptorResult",
    "SemanticAnalysisResult",
    "BaseInterceptor",
    "BaseSemanticAnalyzer",
    "RuleInterceptor",
    "create_interceptor",
    "LLMAnalyzer",
    "create_semantic_analyzer",
    "MessageBufferState",
    # ========== Gateway 子系统 ==========
    "GatewaySystem",
    "GatewayRuntime",
    "GatewayService",
    "GatewayState",
    "PatchouliPrepareDecision",
    # ========== Generation Engine ==========
    "MemoryGenerationEngine",
    "BaseMemoryExtractor",
    "BaseDeduplicator",
    "ExtractedMemoryDraft",
    "DuplicateDecision",
    "WriteFocus",
    "GenerationRequest",
    "UpdateFocus",
    "MergeResult",
    "LLMMemoryExtractor",
    "NoOpMemoryExtractor",
    "create_extractor",
    "MemoryDeduplicator",
    "NoOpDeduplicator",
    "create_deduplicator",
    # ========== Retrieval Engine ==========
    "RetrievalEngine",
    "RetrievalQuery",
    "RetrievalResult",
    "SearchResult",
    "SearchResults",
    "BaseMemoryRetriever",
    "BaseReranker",
    "BaseFusion",
    "FilterConverter",
    "QdrantFilterConverter",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "ReciprocalRankFusion",
    "AdaptiveWeightedFusion",
    "NoopReranker",
    "CrossEncoderReranker",
    "create_retriever",
    # ========== Lifecycle Engine ==========
    "MemoryLifecycleEngine",
    "BaseGarbageCollector",
    "EventType",
    "ReinforcementResult",
    "MemoryEvent",
    "ArchiveStatus",
    "ArchiveRecord",
    "VitalityCalculator",
    "DynamicReinforcementEngine",
    "PeriodicGarbageCollector",
    "create_garbage_collector",
    # ========== Perception Engine ==========
    "SemanticFlowPerceptionLayer",
    "BasePerceptionLayer",
    "TraceItem",
    "InteractionPayload",
    "LogicalBlock",
    "BufferState",
    "SemanticBuffer",
    "FlushEvent",
    "FlushReason",
    "TriggerManager",
    "DECISION_MATRIX",
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "NullPerceptionLayer",
    "create_relay_controller",
    "create_perception_layer",
    # ========== Server Models ==========
    "ErrorResponse",
    "HealthResponse",
    "ChatRequest",
    "ChatTokenEvent",
    "MTPStartEvent",
    "MTPResultEvent",
    "TopicInfoEvent",
    "ChatDoneEvent",
    "ChatErrorEvent",
    "PassiveIngressRequest",
    "PassiveIngressResponse",
    "MemoryResponse",
    "MemoryListResponse",
    "TopicSnapshotResponse",
    "TopicListResponse",
    "TriggerResponse",
    # ========== 懒加载组件 ==========
    "PatchouliRuntime",
    "PatchouliService",
    "PatchouliSystem",
    "PerceptionFamiliar",
    "RetrievalFamiliar",
    "LifecycleFamiliar",
    "MemoryGenerationFamiliar",
    "MemoryGenerationCoordinator",
]


__version__ = "0.6.0"
