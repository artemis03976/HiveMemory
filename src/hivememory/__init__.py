"""
HiveMemory - 分布式记忆管理系统

帕秋莉体系 v3.0 (Eye + Kernel):
    - PatchouliSystem (The Facility): 外层容器，持有 Eye + Kernel
    - TheEye (真理之眼): Ingress Gateway，意图识别、查询重写
    - PatchouliKernel (帕秋莉内核): 中心调度器，管理微服务
        - RetrievalFamiliar (检索使魔): 混合检索、重排序、上下文渲染
        - LibrarianCore (馆长本体): 话题感知、记忆生成、生命周期管理

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
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    VerificationStatus,
    Identity,
    StreamMessage,
    StreamMessageType,
    MetaData,
    IndexLayer,
    PayloadLayer,
    Artifacts,
    RelationLayer,
)

from hivememory.patchouli.config import (
    load_app_config,
    HiveMemoryConfig,
    MemoryGatewayConfig,
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
    LocalEmbeddingService,
    BGEM3EmbeddingService,
    get_embedding_service,
    get_bge_m3_service,
    get_default_embedding_service,
)
from hivememory.infrastructure.rerank import (
    BaseRerankService,
    FlagRerankerService,
    get_flag_reranker_service,
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
    ObserverBufferState,
    ObserverSessionBuffer,
    ObserverBufferManager,
)

from hivememory.engines.generation import (
    MemoryGenerationEngine,
    BaseMemoryExtractor,
    BaseDeduplicator,
    ExtractedMemoryDraft,
    DuplicateDecision,
    WriteFocus,
    GenerationRequest,
    UpdateFocus,
    MergeResult,
    LLMMemoryExtractor,
    NoOpMemoryExtractor,
    create_extractor,
    MemoryDeduplicator,
    NoOpDeduplicator,
    create_deduplicator,
)

from hivememory.engines.retrieval import (
    RetrievalEngine,
    RetrievalQuery,
    RetrievalResult,
    SearchResult,
    SearchResults,
    RenderFormat,
    BaseMemoryRetriever,
    BaseContextRenderer,
    BaseReranker,
    BaseFusion,
    FilterConverter,
    QdrantFilterConverter,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    CachedRetriever,
    ReciprocalRankFusion,
    AdaptiveWeightedFusion,
    NoopReranker,
    CrossEncoderReranker,
    create_retriever,
    FullContextRenderer,
    CascadeContextRenderer,
    CompactContextRenderer,
    create_renderer,
)

from hivememory.engines.lifecycle import (
    MemoryLifecycleEngine,
    BaseMemoryArchiver,
    BaseGarbageCollector,
    EventType,
    ReinforcementResult,
    MemoryEvent,
    ArchiveStatus,
    ArchiveRecord,
    VitalityCalculator,
    DynamicReinforcementEngine,
    FileBasedArchiver,
    create_archiver,
    PeriodicGarbageCollector,
    ScheduledGarbageCollector,
    create_garbage_collector,
)

from hivememory.engines.perception import (
    SemanticFlowPerceptionLayer,
    BasePerceptionLayer,
    TraceItem,
    InteractionPayload,
    Triplet,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    FlushEvent,
    FlushReason,
    SemanticBufferManager,
    TriggerManager,
    DECISION_MATRIX,
    BaseRelayController,
    SimpleRelayController,
    LLMRelayController,
    create_relay_controller,
    create_perception_layer,
)

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
    IngestRequest,
    IngestResponse,
    MemoryResponse,
    MemoryListResponse,
    TopicSnapshotResponse,
    TopicListResponse,
    TriggerResponse,
)


def __getattr__(name: str):
    """懒加载以避免循环导入"""
    if name == "PatchouliKernel":
        from hivememory.patchouli.kernel import PatchouliKernel
        return PatchouliKernel
    if name == "PatchouliSystem":
        from hivememory.patchouli.system import PatchouliSystem
        return PatchouliSystem
    if name == "TheEye":
        from hivememory.patchouli.eye import TheEye
        return TheEye
    if name == "RetrievalFamiliar":
        from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
        return RetrievalFamiliar
    if name == "LibrarianCore":
        from hivememory.patchouli.kernel.librarian_core import LibrarianCore
        return LibrarianCore
    if name == "KoakumaRuntime":
        from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
        return KoakumaRuntime
    if name == "WorkerAgentService":
        from hivememory.patchouli.worker_agent import WorkerAgentService
        return WorkerAgentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # ========== 核心模型 ==========
    "MemoryAtom",
    "MemoryType",
    "MemoryVisibility",
    "VerificationStatus",
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
    "MemoryGatewayConfig",
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
    "LocalEmbeddingService",
    "BGEM3EmbeddingService",
    "get_embedding_service",
    "get_bge_m3_service",
    "get_default_embedding_service",
    # ========== Rerank 服务 ==========
    "BaseRerankService",
    "FlagRerankerService",
    "get_flag_reranker_service",
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
    "ObserverBufferState",
    "ObserverSessionBuffer",
    "ObserverBufferManager",
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
    "RenderFormat",
    "BaseMemoryRetriever",
    "BaseContextRenderer",
    "BaseReranker",
    "BaseFusion",
    "FilterConverter",
    "QdrantFilterConverter",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CachedRetriever",
    "ReciprocalRankFusion",
    "AdaptiveWeightedFusion",
    "NoopReranker",
    "CrossEncoderReranker",
    "create_retriever",
    "FullContextRenderer",
    "CascadeContextRenderer",
    "CompactContextRenderer",
    "create_renderer",
    # ========== Lifecycle Engine ==========
    "MemoryLifecycleEngine",
    "BaseMemoryArchiver",
    "BaseGarbageCollector",
    "EventType",
    "ReinforcementResult",
    "MemoryEvent",
    "ArchiveStatus",
    "ArchiveRecord",
    "VitalityCalculator",
    "DynamicReinforcementEngine",
    "FileBasedArchiver",
    "create_archiver",
    "PeriodicGarbageCollector",
    "ScheduledGarbageCollector",
    "create_garbage_collector",
    # ========== Perception Engine ==========
    "SemanticFlowPerceptionLayer",
    "BasePerceptionLayer",
    "TraceItem",
    "InteractionPayload",
    "Triplet",
    "LogicalBlock",
    "BufferState",
    "SemanticBuffer",
    "FlushEvent",
    "FlushReason",
    "SemanticBufferManager",
    "TriggerManager",
    "DECISION_MATRIX",
    "BaseRelayController",
    "SimpleRelayController",
    "LLMRelayController",
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
    "IngestRequest",
    "IngestResponse",
    "MemoryResponse",
    "MemoryListResponse",
    "TopicSnapshotResponse",
    "TopicListResponse",
    "TriggerResponse",
    # ========== 懒加载组件 ==========
    "PatchouliKernel",
    "PatchouliSystem",
    "TheEye",
    "RetrievalFamiliar",
    "LibrarianCore",
    "KoakumaRuntime",
    "WorkerAgentService",
]


__version__ = "0.1.0"
