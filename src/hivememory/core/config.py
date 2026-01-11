"""
HiveMemory 配置管理系统

支持:
- 从 YAML 文件加载配置
- 从环境变量覆盖配置
- 配置验证与类型检查
"""

import os
from pathlib import Path
from typing import Optional, Any, Dict, List
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


# ========== 基础设施服务配置 ==========

class LLMConfig(BaseSettings):
    """LLM 模型配置"""
    provider: str = "litellm"
    model: str = Field(..., description="模型名称,如 gpt-4o")
    api_key: str = Field(..., description="API Key")
    api_base: Optional[str] = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    model_config = SettingsConfigDict(extra="allow")


class EmbeddingConfig(BaseSettings):
    """Embedding 模型配置"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding模型名称")
    device: str = Field(default="cpu", description="运行设备: cpu/cuda/mps")
    cache_dir: Optional[str] = Field(default=None, description="模型缓存目录")
    batch_size: int = Field(default=32, gt=0)
    normalize_embeddings: bool = Field(default=True)
    dimension: int = Field(default=384, gt=0, description="向量维度")

    model_config = SettingsConfigDict(extra="allow")


class QdrantConfig(BaseSettings):
    """Qdrant 向量数据库配置"""
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    api_key: Optional[str] = Field(default=None)
    collection_name: str = Field(default="hivememory_main")
    vector_dimension: int = Field(default=1024)
    distance_metric: str = Field(default="Cosine")
    on_disk_payload: bool = Field(default=False)

    model_config = SettingsConfigDict(extra="allow")


class RedisConfig(BaseSettings):
    """Redis 配置"""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: Optional[str] = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    decode_responses: bool = Field(default=True)

    model_config = SettingsConfigDict(extra="allow")


# ========== 感知层配置 ==========

class SimplePerceptionConfig(BaseSettings):
    """SimplePerceptionLayer 配置

    简单感知层使用三重触发机制：
    - 消息数阈值触发
    - 空闲超时触发
    - 语义边界触发（可选）
    """
    message_threshold: int = Field(default=6, gt=0, description="消息数触发阈值")
    timeout_seconds: int = Field(default=900, gt=0, description="超时触发时间（秒）")
    enable_semantic_trigger: bool = Field(default=True, description="是否启用语义边界触发")

    model_config = SettingsConfigDict(extra="allow")


class SemanticFlowPerceptionConfig(BaseSettings):
    """SemanticFlowPerceptionLayer 配置

    语义流感知层使用统一语义流架构：
    - 语义吸附判定
    - Token 溢出检测与接力
    - 异步空闲超时监控
    """
    # 空闲监控配置
    idle_timeout_seconds: int = Field(
        default=900,
        gt=0,
        description="空闲超时时间（秒），默认 15 分钟"
    )
    scan_interval_seconds: int = Field(
        default=30,
        gt=0,
        description="空闲监控扫描间隔（秒）"
    )

    # 语义吸附配置
    semantic_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="语义相似度阈值，超过此值认为是同一话题"
    )
    short_text_threshold: int = Field(
        default=50,
        gt=0,
        description="短文本强吸附阈值（tokens），少于此值强制吸附"
    )
    ema_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="指数移动平均系数，用于更新话题核心向量"
    )

    # Token 限制
    max_processing_tokens: int = Field(
        default=8192,
        gt=0,
        description="单次处理的最大 Token 数"
    )

    # 高级配置
    enable_smart_summary: bool = Field(
        default=False,
        description="是否启用智能摘要（使用 LLM 生成接力摘要）"
    )

    # Embedding 配置（可选，覆盖全局 EmbeddingConfig）
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding 模型（None 则复用全局配置）"
    )
    embedding_device: Optional[str] = Field(
        default=None,
        description="运行设备：cpu 或 cuda（None 则复用全局配置）"
    )
    embedding_cache_dir: Optional[str] = Field(
        default=None,
        description="模型缓存目录（None 则复用全局配置）"
    )
    embedding_batch_size: Optional[int] = Field(
        default=None,
        description="批处理大小（None 则复用全局配置）"
    )

    model_config = SettingsConfigDict(extra="allow")


class MemoryPerceptionConfig(BaseSettings):
    """
    感知层统一入口配置

    根据 layer_type 选择对应的感知层配置：
    - "semantic_flow": 使用 semantic_flow 子配置
    - "simple": 使用 simple 子配置

    参考: PROJECT.md 2.3.1 节
    """
    # 感知层类型选择
    layer_type: str = Field(
        default="semantic_flow",
        description="感知层类型: semantic_flow 或 simple"
    )

    # 启用开关
    enable: bool = Field(
        default=True,
        description="是否启用感知层"
    )

    # 子配置（根据 layer_type 选择使用哪一个）
    semantic_flow: SemanticFlowPerceptionConfig = Field(
        default_factory=SemanticFlowPerceptionConfig,
        description="语义流感知层配置"
    )
    simple: SimplePerceptionConfig = Field(
        default_factory=SimplePerceptionConfig,
        description="简单感知层配置"
    )

    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def from_env(cls) -> "MemoryPerceptionConfig":
        """从环境变量加载感知层配置"""
        config = cls()

        env_mapping = {
            # 感知层类型
            "PERCEPTION_LAYER_TYPE": ("layer_type", str),
            # SemanticFlowPerceptionLayer 配置
            "PERCEPTION_IDLE_TIMEOUT": ("semantic_flow__idle_timeout_seconds", int),
            "PERCEPTION_SCAN_INTERVAL": ("semantic_flow__scan_interval_seconds", int),
            "PERCEPTION_SEMANTIC_THRESHOLD": ("semantic_flow__semantic_threshold", float),
            "PERCEPTION_SHORT_TEXT_THRESHOLD": ("semantic_flow__short_text_threshold", int),
            "PERCEPTION_EMA_ALPHA": ("semantic_flow__ema_alpha", float),
            "PERCEPTION_MAX_TOKENS": ("semantic_flow__max_processing_tokens", int),
            "PERCEPTION_ENABLE_SMART_SUMMARY": ("semantic_flow__enable_smart_summary", lambda x: x.lower() in ("true", "1", "yes")),
            "PERCEPTION_EMBEDDING_MODEL": ("semantic_flow__embedding_model", str),
            "PERCEPTION_EMBEDDING_DEVICE": ("semantic_flow__embedding_device", str),
            "PERCEPTION_EMBEDDING_CACHE_DIR": ("semantic_flow__embedding_cache_dir", str),
            # SimplePerceptionLayer 配置
            "PERCEPTION_SIMPLE_MESSAGE_THRESHOLD": ("simple__message_threshold", int),
            "PERCEPTION_SIMPLE_TIMEOUT": ("simple__timeout_seconds", int),
            "PERCEPTION_SIMPLE_SEMANTIC_TRIGGER": ("simple__enable_semantic_trigger", lambda x: x.lower() in ("true", "1", "yes")),
        }

        for env_key, (field_name, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    value = converter(os.environ[env_key])
                    # 处理嵌套配置 sub_config__field
                    if "__" in field_name:
                        sub_field, attr_name = field_name.split("__", 1)
                        if hasattr(config, sub_field):
                            sub_config = getattr(config, sub_field)
                            setattr(sub_config, attr_name, value)
                    else:
                        setattr(config, field_name, value)
                except (ValueError, TypeError):
                    pass

        return config


# ========== 记忆生成配置 ==========

class ExtractorConfig(BaseSettings):
    """LLMMemoryExtractor 配置"""
    # LLM 配置（可选，覆盖全局 Librarian LLM 配置）
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 配置（None 则复用全局）")

    # 提示词配置（可选）
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    user_prompt: Optional[str] = Field(default=None, description="自定义用户提示词")

    # 重试策略
    max_retries: int = Field(default=2, gt=0, le=5, description="最大重试次数")

    # 温度和 Token（覆盖 LLM 配置时使用）
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="LLM 温度参数（None 则使用全局）")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="LLM 最大 Token 数（None 则使用全局）")

    model_config = SettingsConfigDict(extra="allow")


class GaterConfig(BaseSettings):
    """价值评估器配置"""
    # 评估器类型选择
    gater_type: str = Field(default="rule", description="评估器类型: rule/llm/hybrid")

    # RuleBasedGater 配置
    min_total_length: int = Field(default=20, gt=0, description="对话总长度最小值（字符数）")
    min_substantive_length: int = Field(default=10, gt=0, description="实质内容最小长度")

    # 自定义黑白名单（可选）
    trivial_patterns: List[str] = Field(default_factory=list, description="黑名单关键词（覆盖默认）")
    valuable_patterns: List[str] = Field(default_factory=list, description="白名单关键词（补充默认）")

    # LLMAssistedGater 配置（预留）
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 评估器配置")

    model_config = SettingsConfigDict(extra="allow")


class DeduplicatorConfig(BaseSettings):
    """查重器配置"""
    high_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="高相似度阈值（TOUCH/UPDATE 分界）")
    low_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="低相似度阈值（UPDATE/CREATE 分界）")
    content_similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="内容相似度阈值")
    enable_vitality_tracking: bool = Field(default=True, description="是否启用生命周期追踪")

    model_config = SettingsConfigDict(extra="allow")


class MemoryGenerationConfig(BaseSettings):
    """记忆提取统一配置（Generation 模块）

    包含三个子模块的配置：
    - extractor: LLM 记忆提取器
    - gater: 价值评估器
    - deduplicator: 查重器
    """
    # 子配置
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig, description="LLM 提取器配置")
    gater: GaterConfig = Field(default_factory=GaterConfig, description="价值评估器配置")
    deduplicator: DeduplicatorConfig = Field(default_factory=DeduplicatorConfig, description="查重器配置")

    # 兼容性：保留旧字段（已废弃，未使用）
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0, description="最低置信度阈值（废弃，未使用）")
    max_tags: int = Field(default=5, gt=0, description="最多标签数（废弃，未使用）")

    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def from_env(cls) -> "MemoryGenerationConfig":
        """从环境变量加载配置"""
        config = cls()

        env_mapping = {
            # Extractor 配置
            "GENERATION_EXTRACTION_MAX_RETRIES": ("extractor__max_retries", int),
            "GENERATION_EXTRACTION_TEMPERATURE": ("extractor__temperature", float),
            "GENERATION_EXTRACTION_MAX_TOKENS": ("extractor__max_tokens", int),
            # Gater 配置
            "GENERATION_GATER_TYPE": ("gater__gater_type", str),
            "GENERATION_GATER_MIN_TOTAL_LENGTH": ("gater__min_total_length", int),
            "GENERATION_GATER_MIN_SUBSTANTIVE_LENGTH": ("gater__min_substantive_length", int),
            # Deduplicator 配置
            "GENERATION_DEDUPLICATOR_HIGH_THRESHOLD": ("deduplicator__high_similarity_threshold", float),
            "GENERATION_DEDUPLICATOR_LOW_THRESHOLD": ("deduplicator__low_similarity_threshold", float),
            "GENERATION_DEDUPLICATOR_CONTENT_THRESHOLD": ("deduplicator__content_similarity_threshold", float),
        }

        for env_key, (field_name, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    value = converter(os.environ[env_key])
                    if "__" in field_name:
                        sub_field, attr_name = field_name.split("__", 1)
                        if hasattr(config, sub_field):
                            sub_config = getattr(config, sub_field)
                            setattr(sub_config, attr_name, value)
                    else:
                        setattr(config, field_name, value)
                except (ValueError, TypeError):
                    pass

        return config


# ========== 记忆检索配置 ==========

class RouterConfig(BaseSettings):
    """检索路由器配置"""
    router_type: str = Field(default="simple", description="路由器类型: simple/llm/always/never")

    # SimpleRouter 配置
    min_query_length: int = Field(default=3, gt=0, description="查询最小长度")
    min_keyword_count: int = Field(default=1, gt=0, description="最小关键词数量")
    additional_keywords: List[str] = Field(default_factory=list, description="额外的检索触发关键词")

    # LLMRouter 配置（预留）
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 路由器配置")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")

    model_config = SettingsConfigDict(extra="allow")


class QueryProcessorConfig(BaseSettings):
    """查询处理器配置"""
    # 查询预处理
    enable_time_parsing: bool = Field(default=True, description="是否启用时间表达式解析")
    enable_type_detection: bool = Field(default=True, description="是否启用记忆类型检测")
    enable_query_expansion: bool = Field(default=True, description="是否启用查询扩展")

    # 查询扩展
    expansion_keywords: List[str] = Field(default_factory=list, description="扩展关键词列表")

    # LLM 扩展（预留）
    enable_llm_rewrite: bool = Field(default=False, description="是否启用 LLM 查询重写")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 配置")

    model_config = SettingsConfigDict(extra="allow")


class ContextRendererConfig(BaseSettings):
    """上下文渲染器配置"""
    # 渲染格式
    render_format: str = Field(default="xml", description="渲染格式: xml/markdown/plain")
    max_tokens: int = Field(default=2000, gt=0, description="最大 Token 数")
    max_content_length: int = Field(default=500, gt=0, description="单条记忆最大内容长度")

    # 显示选项
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    include_confidence: bool = Field(default=True, description="是否包含置信度分数")
    include_timestamp: bool = Field(default=True, description="是否包含时间戳")
    include_artifact: bool = Field(default=False, description="是否包含记忆内容")

    # 格式化选项
    title_template: str = Field(default="📝 {title}", description="标题模板")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值显示")

    # 时间阈值（用于标记"陈旧"记忆）
    old_memory_days: int = Field(default=90, gt=0, description="记忆被视为陈旧的天数")

    model_config = SettingsConfigDict(extra="allow")


class DenseRetrieverConfig(BaseSettings):
    """稠密检索配置"""
    enabled: bool = Field(default=True)
    top_k: int = Field(default=50, gt=0, description="RRF融合前的召回数量")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值")
    enable_time_decay: bool = Field(default=True)
    time_decay_days: int = Field(default=30, gt=0)
    enable_confidence_boost: bool = Field(default=True, description="是否启用置信度加权")

    model_config = SettingsConfigDict(extra="allow")


class SparseRetrieverConfig(BaseSettings):
    """稀疏检索配置"""
    enabled: bool = Field(default=True)
    top_k: int = Field(default=50, gt=0, description="RRF融合前的召回数量")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值")

    model_config = SettingsConfigDict(extra="allow")


class FusionConfig(BaseSettings):
    """RRF 融合配置"""
    rrf_k: int = Field(default=60, gt=0, description="RRF常数")
    dense_weight: float = Field(default=1.0, ge=0.0, description="稠密检索权重")
    sparse_weight: float = Field(default=1.0, ge=0.0, description="稀疏检索权重")
    final_top_k: int = Field(default=5, gt=0, description="最终返回数量")

    model_config = SettingsConfigDict(extra="allow")


class RerankerConfig(BaseSettings):
    """重排序器配置"""
    enabled: bool = Field(default=True, description="是否启用重排序")
    type: str = Field(default="cross_encoder", description="noop 或 cross_encoder")
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker 模型名称")

    # BGE-Reranker 专用配置
    device: str = Field(default="cpu", description="运行设备: cpu/cuda")
    use_fp16: bool = Field(default=True, description="是否使用 FP16 精度")
    batch_size: int = Field(default=32, gt=0, description="批处理大小")
    top_k: int = Field(default=20, gt=0, description="仅重排序前N个结果")
    normalize_scores: bool = Field(default=True, description="是否标准化分数到 0-1")

    model_config = SettingsConfigDict(extra="allow")


class HybridSearchConfig(BaseSettings):
    """混合搜索完整配置

    包含顶层字段（兼容旧配置）和子配置：
    - dense: 稠密检索配置
    - sparse: 稀疏检索配置
    - fusion: RRF 融合配置
    - reranker: 重排序配置
    """
    # 顶层字段（兼容旧 config.yaml 结构）
    top_k: int = Field(default=5, gt=0, description="最终返回数量")
    score_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="相似度阈值")
    rerank_model: Optional[str] = Field(default=None, description="重排序模型（废弃，使用 reranker.model_name）")
    enable_hybrid_search: bool = Field(default=True, description="是否启用混合检索")
    enable_parallel: bool = Field(default=True, description="是否启用并行召回")

    # 子配置
    dense: DenseRetrieverConfig = Field(default_factory=DenseRetrieverConfig)
    sparse: SparseRetrieverConfig = Field(default_factory=SparseRetrieverConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def from_env(cls) -> "HybridSearchConfig":
        """从环境变量加载检索配置"""
        config = cls()

        env_mapping = {
            # 顶层配置
            "RETRIEVAL_TOP_K": ("top_k", int),
            "RETRIEVAL_SCORE_THRESHOLD": ("score_threshold", float),
            "RETRIEVAL_ENABLE_HYBRID": ("enable_hybrid_search", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_ENABLE_PARALLEL": ("enable_parallel", lambda x: x.lower() in ("true", "1", "yes")),
            # Dense 配置
            "RETRIEVAL_DENSE_TOP_K": ("dense__top_k", int),
            "RETRIEVAL_DENSE_THRESHOLD": ("dense__score_threshold", float),
            "RETRIEVAL_DENSE_ENABLE_TIME_DECAY": ("dense__enable_time_decay", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_DENSE_TIME_DECAY_DAYS": ("dense__time_decay_days", int),
            "RETRIEVAL_DENSE_ENABLE_CONFIDENCE_BOOST": ("dense__enable_confidence_boost", lambda x: x.lower() in ("true", "1", "yes")),
            # Sparse 配置
            "RETRIEVAL_SPARSE_TOP_K": ("sparse__top_k", int),
            "RETRIEVAL_SPARSE_THRESHOLD": ("sparse__score_threshold", float),
            # Fusion 配置
            "RETRIEVAL_RRF_K": ("fusion__rrf_k", int),
            "RETRIEVAL_DENSE_WEIGHT": ("fusion__dense_weight", float),
            "RETRIEVAL_SPARSE_WEIGHT": ("fusion__sparse_weight", float),
            "RETRIEVAL_FINAL_TOP_K": ("fusion__final_top_k", int),
            # Reranker 配置
            "RETRIEVAL_RERANKER_ENABLED": ("reranker__enabled", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_RERANKER_MODEL": ("reranker__model_name", str),
            "RETRIEVAL_RERANKER_DEVICE": ("reranker__device", str),
            "RETRIEVAL_RERANKER_TOP_K": ("reranker__top_k", int),
            "RETRIEVAL_RERANKER_USE_FP16": ("reranker__use_fp16", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_RERANKER_BATCH_SIZE": ("reranker__batch_size", int),
            "RETRIEVAL_RERANKER_NORMALIZE_SCORES": ("reranker__normalize_scores", lambda x: x.lower() in ("true", "1", "yes")),
        }

        for env_key, (field_name, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    value = converter(os.environ[env_key])
                    if "__" in field_name:
                        sub_field, attr_name = field_name.split("__", 1)
                        if hasattr(config, sub_field):
                            sub_config = getattr(config, sub_field)
                            setattr(sub_config, attr_name, value)
                    else:
                        setattr(config, field_name, value)
                except (ValueError, TypeError):
                    pass

        return config


class MemoryRetrievalConfig(BaseSettings):
    """记忆检索统一配置

    包含四个子模块的配置：
    - router: 检索路由器
    - processor: 查询处理器
    - hybrid_search: 混合搜索配置
    - renderer: 上下文渲染器
    """
    # 子配置
    router: RouterConfig = Field(default_factory=RouterConfig)
    processor: QueryProcessorConfig = Field(default_factory=QueryProcessorConfig)
    renderer: ContextRendererConfig = Field(default_factory=ContextRendererConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)

    # 顶层配置（兼容旧版）
    enable_routing: bool = Field(default=True, description="是否启用路由判断")
    default_top_k: int = Field(default=5, gt=0, description="默认返回数量")
    default_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="默认相似度阈值")

    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def from_env(cls) -> "MemoryRetrievalConfig":
        """从环境变量加载配置"""
        config = cls()

        env_mapping = {
            # 顶层配置
            "RETRIEVAL_ENABLE_ROUTING": ("enable_routing", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_DEFAULT_TOP_K": ("default_top_k", int),
            "RETRIEVAL_DEFAULT_THRESHOLD": ("default_threshold", float),
            # Router 配置
            "RETRIEVAL_ROUTER_TYPE": ("router__router_type", str),
            "RETRIEVAL_MIN_QUERY_LENGTH": ("router__min_query_length", int),
            "RETRIEVAL_MIN_KEYWORD_COUNT": ("router__min_keyword_count", int),
            # Processor 配置
            "RETRIEVAL_ENABLE_TIME_PARSING": ("processor__enable_time_parsing", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_ENABLE_TYPE_DETECTION": ("processor__enable_type_detection", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_ENABLE_QUERY_EXPANSION": ("processor__enable_query_expansion", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_ENABLE_LLM_REWRITE": ("processor__enable_llm_rewrite", lambda x: x.lower() in ("true", "1", "yes")),
            # Renderer 配置
            "RETRIEVAL_RENDER_FORMAT": ("renderer__render_format", str),
            "RETRIEVAL_MAX_TOKENS": ("renderer__max_tokens", int),
            "RETRIEVAL_MAX_CONTENT_LENGTH": ("renderer__max_content_length", int),
            "RETRIEVAL_INCLUDE_METADATA": ("renderer__include_metadata", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_INCLUDE_CONFIDENCE": ("renderer__include_confidence", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_INCLUDE_TIMESTAMP": ("renderer__include_timestamp", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_INCLUDE_ARTIFACT": ("renderer__include_artifact", lambda x: x.lower() in ("true", "1", "yes")),
            "RETRIEVAL_OLD_MEMORY_DAYS": ("renderer__old_memory_days", int),
        }

        for env_key, (field_name, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    value = converter(os.environ[env_key])
                    if "__" in field_name:
                        sub_field, attr_name = field_name.split("__", 1)
                        if hasattr(config, sub_field):
                            sub_config = getattr(config, sub_field)
                            setattr(sub_config, attr_name, value)
                    else:
                        setattr(config, field_name, value)
                except (ValueError, TypeError):
                    pass

        return config


# ========== 记忆生命周期配置 ==========

class LifecycleConfig(BaseSettings):
    """记忆生命周期配置"""
    # 基础阈值
    high_watermark: float = Field(default=80.0, ge=0.0, le=100.0, description="高水位阈值")
    low_watermark: float = Field(default=20.0, ge=0.0, le=100.0, description="低水位阈值")
    decay_lambda: float = Field(default=0.01, gt=0.0, description="时间衰减系数")

    # 归档配置
    archive_dir: str = Field(default="data/archived", description="归档存储目录")
    archive_compression: bool = Field(default=True, description="是否压缩归档文件")

    # 垃圾回收配置
    gc_batch_size: int = Field(default=10, gt=0, description="每次GC最多归档数量")
    gc_interval_hours: int = Field(default=24, gt=0, description="GC执行间隔(小时)")
    gc_enable_schedule: bool = Field(default=False, description="是否启用定时GC")

    # 事件历史配置
    enable_event_history: bool = Field(default=True, description="是否记录事件历史")
    event_history_limit: int = Field(default=10000, gt=0, description="事件历史最大条数")

    # 生命力加成配置
    hit_boost: float = Field(default=5.0, description="HIT事件生命力加成")
    citation_boost: float = Field(default=20.0, description="CITATION事件生命力加成")
    positive_feedback_boost: float = Field(default=50.0, description="正面反馈生命力加成")
    negative_feedback_penalty: float = Field(default=-50.0, description="负面反馈生命力惩罚")
    negative_confidence_multiplier: float = Field(default=0.5, ge=0.0, le=1.0, description="负面反馈置信度衰减系数")

    model_config = SettingsConfigDict(extra="allow")


# ========== 记忆总配置配置 ==========

class MemoryConfig(BaseSettings):
    """记忆管理总配置"""
    perception: MemoryPerceptionConfig = Field(default_factory=MemoryPerceptionConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)

    model_config = SettingsConfigDict(extra="allow")


# ========== 工具类配置 ==========

class LoggingConfig(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None)
    console_output: bool = Field(default=True)

    model_config = SettingsConfigDict(extra="allow")


class SystemConfig(BaseSettings):
    """系统全局配置"""
    name: str = Field(default="HiveMemory")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    model_config = SettingsConfigDict(extra="allow")


# ========== 主配置类 ==========

class HiveMemoryConfig(BaseSettings):
    """
    HiveMemory 主配置类

    加载优先级:
    1. 环境变量 (.env)
    2. YAML 配置文件
    3. 默认值
    """
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # 外部服务
    llm: Dict[str, LLMConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    perception: MemoryPerceptionConfig = Field(default_factory=MemoryPerceptionConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "HiveMemoryConfig":
        """
        从 YAML 文件加载配置

        Args:
            yaml_path: YAML 配置文件路径

        Returns:
            HiveMemoryConfig 实例
        """
        # 先加载 .env 文件到环境变量
        from dotenv import load_dotenv
        yaml_path = Path(yaml_path)

        # 从 yaml 路径推导项目根目录
        project_root = yaml_path.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file, override=True)

        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # 环境变量替换
        raw_config = cls._replace_env_vars(raw_config)

        # 构建配置对象
        return cls(**raw_config)

    @staticmethod
    def _replace_env_vars(config: Any) -> Any:
        """
        递归替换配置中的环境变量占位符 ${VAR_NAME}

        Args:
            config: 配置对象 (dict/list/str)

        Returns:
            替换后的配置
        """
        if isinstance(config, dict):
            return {k: HiveMemoryConfig._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [HiveMemoryConfig._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # 提取环境变量名
            var_name = config[2:-1]
            return os.getenv(var_name, "")
        else:
            return config

    def get_worker_llm_config(self) -> LLMConfig:
        """获取 Worker Agent LLM 配置"""
        return self.llm.get("worker", LLMConfig(
            model=os.getenv("WORKER_LLM_MODEL", "gpt-4o"),
            api_key=os.getenv("WORKER_LLM_API_KEY", ""),
            api_base=os.getenv("WORKER_LLM_API_BASE"),
        ))

    def get_librarian_llm_config(self) -> LLMConfig:
        """获取 Librarian Agent LLM 配置"""
        return self.llm.get("librarian", LLMConfig(
            model=os.getenv("LIBRARIAN_LLM_MODEL", "deepseek/deepseek-chat"),
            api_key=os.getenv("LIBRARIAN_LLM_API_KEY", ""),
            api_base=os.getenv("LIBRARIAN_LLM_API_BASE"),
            temperature=0.3,
            max_tokens=8192
        ))


@lru_cache()
def get_config(config_path: Optional[str] = None) -> HiveMemoryConfig:
    """
    获取全局配置实例 (单例模式)

    Args:
        config_path: 配置文件路径, 默认为 configs/config.yaml

    Returns:
        HiveMemoryConfig 实例
    """
    if config_path is None:
        # 自动查找配置文件
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"

    if Path(config_path).exists():
        return HiveMemoryConfig.from_yaml(str(config_path))
    else:
        # 使用默认配置
        print(f"⚠️  配置文件未找到: {config_path}, 使用默认配置")
        return HiveMemoryConfig()


# 导出便捷函数
def get_worker_llm_config() -> LLMConfig:
    """快捷获取 Worker LLM 配置"""
    return get_config().get_worker_llm_config()


def get_librarian_llm_config() -> LLMConfig:
    """快捷获取 Librarian LLM 配置"""
    return get_config().get_librarian_llm_config()


# 便于外部导入
__all__ = [
    "HiveMemoryConfig",
    "SystemConfig",
    # 外部服务
    "LLMConfig",
    "EmbeddingConfig",
    "QdrantConfig",
    "RedisConfig",
    # 感知层配置
    "SimplePerceptionConfig",
    "SemanticFlowPerceptionConfig",
    "MemoryPerceptionConfig",
    # 记忆生成配置
    "ExtractorConfig",
    "GaterConfig",
    "DeduplicatorConfig",
    "MemoryGenerationConfig",
    # 记忆检索配置
    "MemoryConfig",
    "RouterConfig",
    "QueryProcessorConfig",
    "ContextRendererConfig",
    "DenseRetrieverConfig",
    "SparseRetrieverConfig",
    "FusionConfig",
    "RerankerConfig",
    "HybridSearchConfig",
    "MemoryRetrievalConfig",
    # 记忆生命周期管理配置
    "LifecycleConfig",
    # 工具类配置
    "LoggingConfig",
    # 便捷函数
    "get_config",
    "get_worker_llm_config",
    "get_librarian_llm_config",
]
