from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# ========== Storage ==========

class QdrantConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    api_key: Optional[str] = Field(default=None)
    collection_name: str = Field(default="hivememory_main")
    vector_dimension: int = Field(default=1024)
    distance_metric: str = Field(default="Cosine")
    on_disk_payload: bool = Field(default=False)

    model_config = ConfigDict(extra="ignore")


# ========== Gateway ==========

class RuleInterceptorConfig(BaseModel):
    enabled: bool = Field(default=True)
    enable_system: bool = Field(default=True)
    enable_chat: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class LLMAnalyzerConfig(BaseModel):
    enabled: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class MemoryGatewayConfig(BaseModel):
    interceptor: RuleInterceptorConfig = Field(default_factory=RuleInterceptorConfig)
    analyzer: LLMAnalyzerConfig = Field(default_factory=LLMAnalyzerConfig)

    model_config = ConfigDict(extra="ignore")


# ========== Perception ==========

class SimpleRelayConfig(BaseModel):
    type: Literal["simple"] = Field(default="simple")

    model_config = ConfigDict(extra="ignore")


class LLMRelayConfig(BaseModel):
    type: Literal["llm"] = Field(default="llm")

    model_config = ConfigDict(extra="ignore")


class RelayControllerConfig(BaseModel):
    enable: bool = Field(default=True)
    engine: Union[SimpleRelayConfig, LLMRelayConfig] = Field(
        default_factory=SimpleRelayConfig,
        discriminator="type",
    )

    model_config = ConfigDict(extra="ignore")


class SemanticFlowPerceptionConfig(BaseModel):
    enable: bool = Field(default=True)
    idle_timeout_seconds: int = Field(default=900)
    scan_interval_seconds: int = Field(default=30)
    fold_token_threshold: int = Field(default=32768)
    fold_retain_recent_blocks: int = Field(default=2)
    max_resident_topics: int = Field(default=5)
    relay: RelayControllerConfig = Field(default_factory=RelayControllerConfig)

    model_config = ConfigDict(extra="ignore")


class MemoryPerceptionConfig(BaseModel):
    engine: SemanticFlowPerceptionConfig = Field(default_factory=SemanticFlowPerceptionConfig)

    model_config = ConfigDict(extra="ignore")


# ========== Generation ==========

class ExtractorConfig(BaseModel):
    enabled: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class DeduplicatorConfig(BaseModel):
    enabled: bool = Field(default=True)
    high_similarity_threshold: float = Field(default=0.95)
    low_similarity_threshold: float = Field(default=0.75)
    content_similarity_threshold: float = Field(default=0.9)
    enable_vitality_tracking: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class MemoryGenerationConfig(BaseModel):
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    deduplicator: DeduplicatorConfig = Field(default_factory=DeduplicatorConfig)

    model_config = ConfigDict(extra="ignore")


# ========== Retrieval ==========

class ReciprocalRankFusionConfig(BaseModel):
    type: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60)
    dense_weight: float = Field(default=1.0)
    sparse_weight: float = Field(default=1.0)
    final_top_k: int = Field(default=5)

    model_config = ConfigDict(extra="ignore")


class RetrievalModeConfig(BaseModel):
    dense_weight: float = Field(default=0.6)
    sparse_weight: float = Field(default=0.4)
    time_weight: float = Field(default=0.0)
    confidence_penalty_enabled: bool = Field(default=True)
    confidence_penalty_threshold: float = Field(default=0.6)
    confidence_penalty_factor: float = Field(default=0.5)
    vitality_boost_enabled: bool = Field(default=True)
    vitality_high_threshold: float = Field(default=80.0)
    vitality_high_factor: float = Field(default=1.2)
    vitality_low_threshold: float = Field(default=30.0)
    vitality_low_factor: float = Field(default=0.8)

    model_config = ConfigDict(extra="ignore")


class AdaptiveWeightedFusionConfig(BaseModel):
    type: Literal["adaptive"] = "adaptive"
    final_top_k: int = Field(default=5)
    default_mode: str = Field(default="concept")
    debug_mode: RetrievalModeConfig = Field(default_factory=lambda: RetrievalModeConfig(dense_weight=0.3, sparse_weight=0.9, time_weight=0.1))
    concept_mode: RetrievalModeConfig = Field(default_factory=lambda: RetrievalModeConfig(dense_weight=0.8, sparse_weight=0.2, time_weight=0.1, confidence_penalty_threshold=0.5, confidence_penalty_factor=0.7))
    timeline_mode: RetrievalModeConfig = Field(default_factory=lambda: RetrievalModeConfig(dense_weight=0.4, sparse_weight=0.3, time_weight=0.8, confidence_penalty_factor=0.6))
    brainstorm_mode: RetrievalModeConfig = Field(default_factory=lambda: RetrievalModeConfig(dense_weight=0.6, sparse_weight=0.1, time_weight=0.0, confidence_penalty_enabled=False, vitality_boost_enabled=False))

    model_config = ConfigDict(extra="ignore")


class RerankerConfig(BaseModel):
    enabled: bool = Field(default=True)
    model_name: str = Field(default="BAAI/bge-reranker-base")
    device: str = Field(default="cpu")
    cache_dir: Optional[str] = Field(default="data/model_cache")
    use_fp16: bool = Field(default=True)
    batch_size: int = Field(default=32)
    top_k: int = Field(default=20)
    normalize_scores: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class DenseRetrieverConfig(BaseModel):
    type: Literal["dense"] = "dense"
    enabled: bool = Field(default=True)
    top_k: int = Field(default=50)
    score_threshold: float = Field(default=0.0)
    enable_time_decay: bool = Field(default=True)
    time_decay_days: int = Field(default=30)
    enable_confidence_boost: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class SparseRetrieverConfig(BaseModel):
    type: Literal["sparse"] = "sparse"
    enabled: bool = Field(default=True)
    top_k: int = Field(default=50)
    score_threshold: float = Field(default=0.0)

    model_config = ConfigDict(extra="ignore")


class HybridRetrieverConfig(BaseModel):
    type: Literal["hybrid"] = "hybrid"
    top_k: int = Field(default=5)
    score_threshold: float = Field(default=0.0)
    enable_parallel: bool = Field(default=True)
    dense: DenseRetrieverConfig = Field(default_factory=DenseRetrieverConfig)
    sparse: SparseRetrieverConfig = Field(default_factory=SparseRetrieverConfig)
    fusion: Union[ReciprocalRankFusionConfig, AdaptiveWeightedFusionConfig] = Field(
        default_factory=ReciprocalRankFusionConfig,
        discriminator="type",
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    model_config = ConfigDict(extra="ignore")


class FullRendererConfig(BaseModel):
    type: Literal["full"] = "full"
    max_tokens: int = Field(default=2000)
    max_content_length: int = Field(default=500)
    show_artifacts: bool = Field(default=False)
    stale_days: int = Field(default=90)

    model_config = ConfigDict(extra="ignore")


class CascadeRendererConfig(BaseModel):
    type: Literal["cascade"] = "cascade"
    max_memory_tokens: int = Field(default=2000)
    full_payload_count: int = Field(default=1)
    max_content_length: int = Field(default=500)
    index_max_summary_length: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


class CompactRendererConfig(BaseModel):
    type: Literal["compact"] = "compact"
    max_memory_tokens: int = Field(default=2000)
    index_max_summary_length: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


class MemoryRetrievalConfig(BaseModel):
    renderer: Union[FullRendererConfig, CascadeRendererConfig, CompactRendererConfig] = Field(
        default_factory=FullRendererConfig,
        discriminator="type",
    )
    retriever: Union[HybridRetrieverConfig, DenseRetrieverConfig, SparseRetrieverConfig] = Field(
        default_factory=HybridRetrieverConfig,
        discriminator="type",
    )

    model_config = ConfigDict(extra="ignore")


# ========== Lifecycle ==========

class VitalityCalculatorConfig(BaseModel):
    code_snippet_weight: float = Field(default=1.0)
    fact_weight: float = Field(default=0.9)
    url_resource_weight: float = Field(default=0.8)
    reflection_weight: float = Field(default=0.7)
    user_profile_weight: float = Field(default=0.6)
    work_in_progress_weight: float = Field(default=0.5)
    default_weight: float = Field(default=0.5)
    max_access_boost: float = Field(default=20.0)
    points_per_access: float = Field(default=2.0)
    decay_lambda: float = Field(default=0.01)

    model_config = ConfigDict(extra="ignore")


class ReinforcementEngineConfig(BaseModel):
    enable_event_history: bool = Field(default=True)
    event_history_limit: int = Field(default=10000)
    hit_boost: float = Field(default=5.0)
    citation_boost: float = Field(default=20.0)
    positive_feedback_boost: float = Field(default=50.0)
    negative_feedback_penalty: float = Field(default=-50.0)
    negative_confidence_multiplier: float = Field(default=0.5)

    model_config = ConfigDict(extra="ignore")


class ArchiverConfig(BaseModel):
    archive_dir: str = Field(default="data/archived")
    compression: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class GarbageCollectorConfig(BaseModel):
    low_watermark: float = Field(default=20.0)
    batch_size: int = Field(default=10)

    model_config = ConfigDict(extra="ignore")


class MemoryLifecycleConfig(BaseModel):
    vitality_calculator: VitalityCalculatorConfig = Field(default_factory=VitalityCalculatorConfig)
    reinforcement: ReinforcementEngineConfig = Field(default_factory=ReinforcementEngineConfig)
    archiver: ArchiverConfig = Field(default_factory=ArchiverConfig)
    garbage_collector: GarbageCollectorConfig = Field(default_factory=GarbageCollectorConfig)
    high_watermark: float = Field(default=80.0)

    model_config = ConfigDict(extra="ignore")


# ========== Artifacts ==========

class ArtifactStoreConfig(BaseModel):
    enabled: bool = Field(default=True)
    root_dir: str = Field(default=".hivememory/artifacts")
    max_inline_summary_chars: int = Field(default=500)

    model_config = ConfigDict(extra="ignore")


# ========== PatchouliConfig ==========

class PatchouliConfig(BaseModel):
    storage: QdrantConfig = Field(default_factory=QdrantConfig)
    gateway: MemoryGatewayConfig = Field(default_factory=MemoryGatewayConfig)
    perception: MemoryPerceptionConfig = Field(default_factory=MemoryPerceptionConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)
    lifecycle: MemoryLifecycleConfig = Field(default_factory=MemoryLifecycleConfig)
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    artifacts: ArtifactStoreConfig = Field(default_factory=ArtifactStoreConfig)

    model_config = ConfigDict(extra="ignore")
