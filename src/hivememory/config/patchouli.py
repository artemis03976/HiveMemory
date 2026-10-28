from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ========== 存储 ==========


class QdrantConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    prefer_grpc: bool = Field(default=True)
    timeout: int = Field(default=60)
    startup_timeout_seconds: int = Field(default=30)
    api_key: str | None = Field(default=None)
    collection_name: str = Field(default="hivememory_main")
    vector_dimension: int = Field(default=1024)
    distance_metric: str = Field(default="Cosine")
    on_disk_payload: bool = Field(default=False)

    model_config = ConfigDict(extra="forbid")


# ========== 感知 ==========


class SimpleRelayConfig(BaseModel):
    type: Literal["simple"] = Field(default="simple")

    model_config = ConfigDict(extra="forbid")


class LLMRelayConfig(BaseModel):
    type: Literal["llm"] = Field(default="llm")

    model_config = ConfigDict(extra="forbid")


class RelayControllerConfig(BaseModel):
    enable: bool = Field(default=True)
    engine: SimpleRelayConfig | LLMRelayConfig = Field(
        default_factory=SimpleRelayConfig,
        discriminator="type",
    )

    model_config = ConfigDict(extra="forbid")


class SemanticFlowPerceptionConfig(BaseModel):
    enable: bool = Field(default=True)
    fold_token_threshold: int = Field(default=32768, ge=1)
    fold_retain_recent_blocks: int = Field(default=2, ge=1)
    max_resident_topics: int = Field(default=5)
    relay: RelayControllerConfig = Field(default_factory=RelayControllerConfig)

    model_config = ConfigDict(extra="forbid")


class MemoryPerceptionConfig(BaseModel):
    idle_timeout_seconds: int = Field(default=900)
    engine: SemanticFlowPerceptionConfig = Field(default_factory=SemanticFlowPerceptionConfig)

    model_config = ConfigDict(extra="forbid")


# ========== 生成 ==========


class ExtractorConfig(BaseModel):
    enabled: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class DeduplicatorConfig(BaseModel):
    enabled: bool = Field(default=True)
    high_similarity_threshold: float = Field(default=0.95)
    low_similarity_threshold: float = Field(default=0.75)
    content_similarity_threshold: float = Field(default=0.9)

    model_config = ConfigDict(extra="forbid")


class GenerationTranscriptConfig(BaseModel):
    """记忆生成转写中动作段的字符上限。

    感知估算逻辑块的 token 时使用同一组上限，折叠阈值因此反映生成实际看到
    的材料。上限只约束思考与外部工具动作的渲染，MTP 动作的一行摘要不截断。
    """

    max_thought_chars: int = Field(default=1000, ge=1, description="单条思考的字符上限")
    max_args_chars: int = Field(default=300, ge=1, description="工具参数摘要的字符上限")
    max_result_chars: int = Field(default=800, ge=1, description="单条工具结果的字符上限")
    max_turn_action_chars: int = Field(
        default=8000,
        ge=1,
        description="一轮动作段的字符上限，超出时较早的非出错动作降为一行",
    )

    model_config = ConfigDict(extra="forbid")


class MemoryGenerationConfig(BaseModel):
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    deduplicator: DeduplicatorConfig = Field(default_factory=DeduplicatorConfig)
    transcript: GenerationTranscriptConfig = Field(default_factory=GenerationTranscriptConfig)
    queue_capacity: int = Field(default=128, ge=1)
    queue_max_concurrency: int = Field(default=2, ge=1)
    queue_timeout_seconds: float | None = Field(default=300.0, gt=0)
    # 主动生成（WRITE/UPDATE）取话题最近几个 block 与 state_summary 作为材料；
    # 意图在本轮交互封口之前提交，当前一轮不在材料中。
    active_recent_blocks: int = Field(default=5, ge=0)

    model_config = ConfigDict(extra="forbid")


# ========== 检索 ==========

_FUSION_FINAL_TOP_K_REMOVED = (
    "融合配置不再支持 final_top_k：融合不再截取，最终条数在重排之后按请求的 top_k"
    "（缺省时为 retrieval.default_top_k）截取"
)


def _reject_removed_keys(data: Any, removed: dict[str, str]) -> Any:
    """检索配置段忽略未知字段；已删除的键若不显式拒绝，其设置会被静默丢弃。"""
    if isinstance(data, dict):
        for key, reason in removed.items():
            if key in data:
                raise ValueError(reason)
    return data


class ReciprocalRankFusionConfig(BaseModel):
    type: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60)
    dense_weight: float = Field(default=1.0)
    sparse_weight: float = Field(default=1.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_final_top_k(cls, data: Any) -> Any:
        return _reject_removed_keys(data, {"final_top_k": _FUSION_FINAL_TOP_K_REMOVED})


class RetrievalModeConfig(BaseModel):
    dense_weight: float = Field(default=0.6)
    sparse_weight: float = Field(default=0.4)
    confidence_penalty_enabled: bool = Field(default=True)
    confidence_penalty_threshold: float = Field(default=0.6)
    confidence_penalty_factor: float = Field(default=0.5)
    vitality_boost_enabled: bool = Field(default=True)
    vitality_high_threshold: float = Field(default=80.0)
    vitality_high_factor: float = Field(default=1.2)
    vitality_low_threshold: float = Field(default=30.0)
    vitality_low_factor: float = Field(default=0.8)

    model_config = ConfigDict(extra="forbid")


class AdaptiveWeightedFusionConfig(BaseModel):
    type: Literal["adaptive"] = "adaptive"
    default_mode: str = Field(default="concept")
    debug_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(dense_weight=0.3, sparse_weight=0.9)
    )
    concept_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.8,
            sparse_weight=0.2,
            confidence_penalty_threshold=0.5,
            confidence_penalty_factor=0.7,
        )
    )
    timeline_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.4, sparse_weight=0.3, confidence_penalty_factor=0.6
        )
    )
    brainstorm_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.6,
            sparse_weight=0.1,
            confidence_penalty_enabled=False,
            vitality_boost_enabled=False,
        )
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_final_top_k(cls, data: Any) -> Any:
        return _reject_removed_keys(data, {"final_top_k": _FUSION_FINAL_TOP_K_REMOVED})


class RerankerConfig(BaseModel):
    enabled: bool = Field(default=True)
    model_name: str = Field(default="BAAI/bge-reranker-base")
    device: str = Field(default="cpu")
    cache_dir: str | None = Field(default="data/model_cache")
    use_fp16: bool = Field(default=True)
    batch_size: int = Field(default=32)
    top_k: int = Field(default=20)
    normalize_scores: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class DenseRetrieverConfig(BaseModel):
    type: Literal["dense"] = "dense"
    enabled: bool = Field(default=True)
    top_k: int = Field(default=50)
    # 分数阈值只作用于 Dense 召回：按原始 cosine 相似度粗筛，交给重排的即粗筛后的候选。
    score_threshold: float = Field(default=0.5)
    enable_time_decay: bool = Field(default=True)
    time_decay_days: int = Field(default=30)
    enable_confidence_boost: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class SparseRetrieverConfig(BaseModel):
    type: Literal["sparse"] = "sparse"
    enabled: bool = Field(default=True)
    top_k: int = Field(default=50)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_score_threshold(cls, data: Any) -> Any:
        # BM25 原始分没有上界，阈值依赖语料，Sparse 召回不设阈值。
        return _reject_removed_keys(
            data,
            {
                "score_threshold": "Sparse 召回配置不再支持 score_threshold：分数阈值只作用于 Dense 召回"
            },
        )


class HybridRetrieverConfig(BaseModel):
    type: Literal["hybrid"] = "hybrid"
    top_k: int = Field(default=5)
    enable_parallel: bool = Field(default=True)
    dense: DenseRetrieverConfig = Field(default_factory=DenseRetrieverConfig)
    sparse: SparseRetrieverConfig = Field(default_factory=SparseRetrieverConfig)
    fusion: ReciprocalRankFusionConfig | AdaptiveWeightedFusionConfig = Field(
        default_factory=ReciprocalRankFusionConfig,
        discriminator="type",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_score_threshold(cls, data: Any) -> Any:
        # 融合分数只反映名次或混合了不同尺度的原始分，不设阈值。
        return _reject_removed_keys(
            data,
            {
                "score_threshold": (
                    "Hybrid 检索配置不再支持 score_threshold：融合分数不设阈值，"
                    "阈值只作用于 dense 召回路径"
                )
            },
        )

    @model_validator(mode="after")
    def _require_a_recall_path(self) -> "HybridRetrieverConfig":
        if not self.dense.enabled and not self.sparse.enabled:
            raise ValueError("混合检索至少需要启用 dense 或 sparse 其中一路召回")
        return self


class MemoryRetrievalConfig(BaseModel):
    retriever: HybridRetrieverConfig | DenseRetrieverConfig | SparseRetrieverConfig = Field(
        default_factory=HybridRetrieverConfig,
        discriminator="type",
    )
    # 重排序是召回之后的独立阶段，适用于任意召回策略。
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    # 请求未给出 top_k 时的结果条数；最终截取只在重排阶段之后进行一次。
    default_top_k: int = Field(default=5, ge=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_retriever_level_reranker(cls, data: Any) -> Any:
        # 检索器配置忽略未知字段；旧位置若不显式拒绝，其设置会被静默丢弃。
        if isinstance(data, dict):
            retriever = data.get("retriever")
            if isinstance(retriever, dict) and "reranker" in retriever:
                raise ValueError(
                    "retrieval.retriever.reranker 已移至 retrieval.reranker，"
                    "重排序配置不再属于具体检索器"
                )
        return data


# ========== 生命周期 ==========


class VitalityCalculatorConfig(BaseModel):
    # 固有价值权重 (I)：作为抗衰减调制因子，λ_eff = λ * (2 - I)
    # 高价值记忆 (I=1.0) 衰减最慢，低价值记忆 (I=0.5) 衰减最快
    code_snippet_weight: float = Field(default=1.0)
    fact_weight: float = Field(default=0.9)
    url_resource_weight: float = Field(default=0.8)
    reflection_weight: float = Field(default=0.7)
    user_profile_weight: float = Field(default=0.6)
    work_in_progress_weight: float = Field(default=0.5)
    default_weight: float = Field(default=0.5)

    # 衰减基础系数 λ：D(t) = exp(-λ_eff * t), λ_eff = λ * (2 - I)
    decay_lambda: float = Field(default=0.01)

    # 访问加成对数曲线系数 (A 项)：A = access_boost_coef * log(1 + access_count)
    # 对数曲线自然饱和，无需硬上限；推荐 8.0~12.0
    access_boost_coef: float = Field(default=10.0)

    # 初始强度 V_0：固定高值，与 confidence 解耦
    base_vitality: float = Field(default=100.0)

    model_config = ConfigDict(extra="forbid")


class ReinforcementEngineConfig(BaseModel):
    enable_event_history: bool = Field(default=True)
    event_history_limit: int = Field(default=10000)
    hit_boost: float = Field(default=5.0)
    citation_boost: float = Field(default=20.0)
    positive_feedback_boost: float = Field(default=50.0)
    negative_feedback_penalty: float = Field(default=-50.0)
    negative_confidence_multiplier: float = Field(default=0.5)

    model_config = ConfigDict(extra="forbid")


class ArchiverConfig(BaseModel):
    archive_dir: str = Field(default="data/archived")
    compression: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class GarbageCollectorConfig(BaseModel):
    low_watermark: float = Field(default=20.0)
    batch_size: int = Field(default=10)

    model_config = ConfigDict(extra="forbid")


class MemoryLifecycleConfig(BaseModel):
    vitality_calculator: VitalityCalculatorConfig = Field(default_factory=VitalityCalculatorConfig)
    reinforcement: ReinforcementEngineConfig = Field(default_factory=ReinforcementEngineConfig)
    archiver: ArchiverConfig = Field(default_factory=ArchiverConfig)
    garbage_collector: GarbageCollectorConfig = Field(default_factory=GarbageCollectorConfig)

    model_config = ConfigDict(extra="forbid")


# ========== Artifacts ==========


class ArtifactComponentConfig(BaseModel):
    enabled: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class ArtifactConfig(BaseModel):
    enabled: bool = Field(default=True)
    root_dir: str = Field(default=".hivememory/artifacts")
    max_inline_summary_chars: int = Field(default=500)

    interaction: ArtifactComponentConfig = Field(default_factory=ArtifactComponentConfig)
    memory: ArtifactComponentConfig = Field(default_factory=ArtifactComponentConfig)
    document: ArtifactComponentConfig = Field(default_factory=ArtifactComponentConfig)

    model_config = ConfigDict(extra="forbid")


# ========== 关停 ==========


class PatchouliShutdownConfig(BaseModel):
    generation_wait_timeout_seconds: float = Field(default=30.0, ge=0)

    model_config = ConfigDict(extra="forbid")


# ========== PatchouliConfig ==========


class PatchouliConfig(BaseModel):
    storage: QdrantConfig = Field(default_factory=QdrantConfig)
    perception: MemoryPerceptionConfig = Field(default_factory=MemoryPerceptionConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)
    lifecycle: MemoryLifecycleConfig = Field(default_factory=MemoryLifecycleConfig)
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    shutdown: PatchouliShutdownConfig = Field(default_factory=PatchouliShutdownConfig)

    model_config = ConfigDict(extra="forbid")
