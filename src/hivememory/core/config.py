"""
HiveMemory 配置管理系统

改进版特点:
- 采用依赖注入 (DI) 模式
- 使用工厂函数初始化
- Pydantic 原生环境变量支持 (Env > YAML > Default)
- 移除手动映射逻辑
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Type
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from functools import lru_cache

logger = logging.getLogger(__name__)

# ========== YAML Source Helper ==========

def yaml_config_settings_source() -> Dict[str, Any]:
    """
    Pydantic Settings Source: 从 YAML 加载配置

    路径优先级: 环境变量 HIVEMEMORY_CONFIG_PATH > 默认 configs/config.yaml

    注意: YAML 中的值会被环境变量覆盖 (通过 Pydantic 的 env_nested_delimiter 机制)
    """
    default_path = Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml"
    config_path = os.getenv("HIVEMEMORY_CONFIG_PATH", str(default_path))
    path = Path(config_path)

    if not path.exists():
        # 如果是默认路径且不存在，返回空字典（使用默认值）
        if str(path) == str(default_path):
            logger.warning(f"默认配置文件未找到: {path}, 将使用默认值和环境变量")
            return {}
        # 如果是显式指定路径且不存在，抛出异常
        raise FileNotFoundError(f"配置文件不存在: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f) or {}
        return yaml_content
    except Exception as e:
        logger.error(f"加载 YAML 配置文件失败: {e}")
        return {}

# ========== 基础设施服务配置 ==========

class LLMConfig(BaseSettings):
    """LLM 模型配置"""
    provider: str = "litellm"
    model: Optional[str] = Field(default=None, description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API Key")
    api_base: Optional[str] = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__", env_prefix="HIVEMEMORY_")


class EmbeddingConfig(BaseSettings):
    """Embedding 模型配置"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding模型名称")
    device: str = Field(default="cpu", description="运行设备: cpu/cuda/mps")
    cache_dir: Optional[str] = Field(default=None, description="模型缓存目录")
    batch_size: int = Field(default=32, description="批处理大小")
    normalize_embeddings: bool = Field(default=True, description="是否归一化向量")
    dimension: int = Field(default=384, description="向量维度")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class QdrantConfig(BaseSettings):
    """Qdrant 向量数据库配置"""
    host: str = Field(default="localhost", description="Qdrant 主机地址")
    port: int = Field(default=6333, description="HTTP 端口")
    grpc_port: int = Field(default=6334, description="gRPC 端口")
    api_key: Optional[str] = Field(default=None, description="API Key")
    collection_name: str = Field(default="hivememory_main", description="集合名称")
    vector_dimension: int = Field(default=1024, description="向量维度")
    distance_metric: str = Field(default="Cosine", description="距离度量方式")
    on_disk_payload: bool = Field(default=False, description="是否将 Payload 存储在磁盘")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class RedisConfig(BaseSettings):
    """Redis 配置"""
    host: str = Field(default="localhost", description="Redis 主机地址")
    port: int = Field(default=6379, description="Redis 端口")
    password: Optional[str] = Field(default=None, description="Redis 密码")
    db: int = Field(default=0, description="数据库索引")
    decode_responses: bool = Field(default=True, description="是否自动解码响应")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


# ========== 感知层配置 ==========

class SimplePerceptionConfig(BaseSettings):
    """SimplePerceptionLayer 配置"""
    message_threshold: int = Field(default=6, description="消息数触发阈值")
    timeout_seconds: int = Field(default=900, description="超时触发时间（秒）")
    enable_semantic_trigger: bool = Field(default=True, description="是否启用语义边界触发")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class SemanticFlowPerceptionConfig(BaseSettings):
    """SemanticFlowPerceptionLayer 配置"""
    idle_timeout_seconds: int = Field(default=900, description="空闲超时时间（秒）")
    scan_interval_seconds: int = Field(default=30, description="空闲监控扫描间隔（秒）")
    semantic_threshold: float = Field(default=0.6, description="语义相似度阈值")
    short_text_threshold: int = Field(default=50, description="短文本强吸附阈值（tokens）")
    ema_alpha: float = Field(default=0.3, description="指数移动平均系数")
    max_processing_tokens: int = Field(default=8192, description="单次处理的最大 Token 数")
    enable_smart_summary: bool = Field(default=False, description="是否启用智能摘要")
    embedding_model: Optional[str] = Field(default=None, description="Embedding 模型（None 则复用全局配置）")
    embedding_device: Optional[str] = Field(default=None, description="运行设备（None 则复用全局配置）")
    embedding_cache_dir: Optional[str] = Field(default=None, description="模型缓存目录")
    embedding_batch_size: Optional[int] = Field(default=None, description="批处理大小")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class MemoryPerceptionConfig(BaseSettings):
    """感知层统一配置"""
    layer_type: str = Field(default="semantic_flow", description="感知层类型: semantic_flow 或 simple")
    enable: bool = Field(default=True, description="是否启用感知层")
    semantic_flow: SemanticFlowPerceptionConfig = Field(default_factory=SemanticFlowPerceptionConfig, description="语义流感知层配置")
    simple: SimplePerceptionConfig = Field(default_factory=SimplePerceptionConfig, description="简单感知层配置")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


# ========== 记忆生成配置 ==========

class ExtractorConfig(BaseSettings):
    """LLMMemoryExtractor 配置"""
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 配置（None 则复用全局）")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    user_prompt: Optional[str] = Field(default=None, description="自定义用户提示词")
    max_retries: int = Field(default=2, description="最大重试次数")
    temperature: Optional[float] = Field(default=None, description="LLM 温度参数")
    max_tokens: Optional[int] = Field(default=None, description="LLM 最大 Token 数")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class GaterConfig(BaseSettings):
    """价值评估器配置"""
    gater_type: str = Field(default="rule", description="评估器类型: rule/llm/hybrid")
    min_total_length: int = Field(default=20, description="对话总长度最小值")
    min_substantive_length: int = Field(default=10, description="实质内容最小长度")
    trivial_patterns: List[str] = Field(default_factory=list, description="黑名单关键词")
    valuable_patterns: List[str] = Field(default_factory=list, description="白名单关键词")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 评估器配置")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class DeduplicatorConfig(BaseSettings):
    """查重器配置"""
    high_similarity_threshold: float = Field(default=0.95, description="高相似度阈值（TOUCH/UPDATE 分界）")
    low_similarity_threshold: float = Field(default=0.75, description="低相似度阈值（UPDATE/CREATE 分界）")
    content_similarity_threshold: float = Field(default=0.9, description="内容相似度阈值")
    enable_vitality_tracking: bool = Field(default=True, description="是否启用生命周期追踪")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class MemoryGenerationConfig(BaseSettings):
    """记忆生成统一配置"""
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig, description="LLM 提取器配置")
    gater: GaterConfig = Field(default_factory=GaterConfig, description="价值评估器配置")
    deduplicator: DeduplicatorConfig = Field(default_factory=DeduplicatorConfig, description="查重器配置")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


# ========== 记忆检索配置 ==========

class RouterConfig(BaseSettings):
    """检索路由器配置"""
    router_type: str = Field(default="simple", description="路由器类型: simple/llm/always/never")
    min_query_length: int = Field(default=3, description="查询最小长度")
    min_keyword_count: int = Field(default=1, description="最小关键词数量")
    additional_keywords: List[str] = Field(default_factory=list, description="额外的检索触发关键词")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 路由器配置")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    
    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class QueryProcessorConfig(BaseSettings):
    """查询处理器配置"""
    enable_time_parsing: bool = Field(default=True, description="是否启用时间表达式解析")
    enable_type_detection: bool = Field(default=True, description="是否启用记忆类型检测")
    enable_query_expansion: bool = Field(default=True, description="是否启用查询扩展")
    expansion_keywords: List[str] = Field(default_factory=list, description="扩展关键词列表")
    enable_llm_rewrite: bool = Field(default=False, description="是否启用 LLM 查询重写")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM 配置")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class ContextRendererConfig(BaseSettings):
    """上下文渲染器配置"""
    render_format: str = Field(default="xml", description="渲染格式: xml/markdown/plain")
    max_tokens: int = Field(default=2000, description="最大 Token 数")
    max_content_length: int = Field(default=500, description="单条记忆最大内容长度")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    include_confidence: bool = Field(default=True, description="是否包含置信度分数")
    include_timestamp: bool = Field(default=True, description="是否包含时间戳")
    include_artifact: bool = Field(default=False, description="是否包含记忆内容")
    title_template: str = Field(default="📝 {title}", description="标题模板")
    confidence_threshold: float = Field(default=0.5, description="置信度阈值显示")
    old_memory_days: int = Field(default=90, description="记忆被视为陈旧的天数")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class DenseRetrieverConfig(BaseSettings):
    """稠密检索配置"""
    enabled: bool = Field(default=True, description="是否启用稠密检索")
    top_k: int = Field(default=50, description="RRF融合前的召回数量")
    score_threshold: float = Field(default=0.0, description="相似度阈值")
    enable_time_decay: bool = Field(default=True, description="是否启用时间衰减")
    time_decay_days: int = Field(default=30, description="时间衰减半衰期(天)")
    enable_confidence_boost: bool = Field(default=True, description="是否启用置信度加权")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class SparseRetrieverConfig(BaseSettings):
    """稀疏检索配置"""
    enabled: bool = Field(default=True, description="是否启用稀疏检索")
    top_k: int = Field(default=50, description="RRF融合前的召回数量")
    score_threshold: float = Field(default=0.0, description="相似度阈值")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class FusionConfig(BaseSettings):
    """RRF 融合配置"""
    rrf_k: int = Field(default=60, description="RRF常数")
    dense_weight: float = Field(default=1.0, description="稠密检索权重")
    sparse_weight: float = Field(default=1.0, description="稀疏检索权重")
    final_top_k: int = Field(default=5, description="最终返回数量")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class RerankerConfig(BaseSettings):
    """重排序器配置"""
    enabled: bool = Field(default=True, description="是否启用重排序")
    type: str = Field(default="cross_encoder", description="noop 或 cross_encoder")
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker 模型名称")
    device: str = Field(default="cpu", description="运行设备: cpu/cuda")
    use_fp16: bool = Field(default=True, description="是否使用 FP16 精度")
    batch_size: int = Field(default=32, description="批处理大小")
    top_k: int = Field(default=20, description="仅重排序前N个结果")
    normalize_scores: bool = Field(default=True, description="是否标准化分数到 0-1")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class HybridRetrieverConfig(BaseSettings):
    """混合检索完整配置"""
    top_k: int = Field(default=5, description="最终返回数量")
    score_threshold: float = Field(default=0.75, description="相似度阈值")
    enable_hybrid_search: bool = Field(default=True, description="是否启用混合检索")
    enable_parallel: bool = Field(default=True, description="是否启用并行召回")

    dense: DenseRetrieverConfig = Field(default_factory=DenseRetrieverConfig, description="稠密检索配置")
    sparse: SparseRetrieverConfig = Field(default_factory=SparseRetrieverConfig, description="稀疏检索配置")
    fusion: FusionConfig = Field(default_factory=FusionConfig, description="RRF 融合配置")
    reranker: RerankerConfig = Field(default_factory=RerankerConfig, description="重排序配置")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class MemoryRetrievalConfig(BaseSettings):
    """记忆检索统一配置"""
    router: RouterConfig = Field(default_factory=RouterConfig, description="检索路由器")
    processor: QueryProcessorConfig = Field(default_factory=QueryProcessorConfig, description="查询处理器")
    renderer: ContextRendererConfig = Field(default_factory=ContextRendererConfig, description="上下文渲染器")
    retriever: HybridRetrieverConfig = Field(default_factory=HybridRetrieverConfig, description="混合检索配置")
    enable_routing: bool = Field(default=True, description="是否启用路由判断")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


# ========== 记忆生命周期配置 ==========

class VitalityCalculatorConfig(BaseSettings):
    """生命力计算器配置"""
    code_snippet_weight: float = Field(default=1.0, description="代码片段权重")
    fact_weight: float = Field(default=0.9, description="事实权重")
    url_resource_weight: float = Field(default=0.8, description="URL资源权重")
    reflection_weight: float = Field(default=0.7, description="反思权重")
    user_profile_weight: float = Field(default=0.6, description="用户画像权重")
    work_in_progress_weight: float = Field(default=0.5, description="进行中权重")
    default_weight: float = Field(default=0.5, description="默认权重")
    max_access_boost: float = Field(default=20.0, description="最大访问加成")
    points_per_access: float = Field(default=2.0, description="每次访问的加成分数")
    decay_lambda: float = Field(default=0.01, description="时间衰减系数")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class ReinforcementEngineConfig(BaseSettings):
    """强化引擎配置"""
    enable_event_history: bool = Field(default=True, description="是否记录事件历史")
    event_history_limit: int = Field(default=10000, description="事件历史最大条数")
    hit_boost: float = Field(default=5.0, description="HIT 事件加成")
    citation_boost: float = Field(default=20.0, description="CITATION 事件加成")
    positive_feedback_boost: float = Field(default=50.0, description="正面反馈加成")
    negative_feedback_penalty: float = Field(default=-50.0, description="负面反馈惩罚")
    negative_confidence_multiplier: float = Field(default=0.5, description="负面反馈置信度衰减系数")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class ArchiverConfig(BaseSettings):
    """归档器配置"""
    archive_dir: str = Field(default="data/archived", description="归档目录路径")
    compression: bool = Field(default=True, description="是否使用 GZIP 压缩")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class GarbageCollectorConfig(BaseSettings):
    """垃圾回收器配置"""
    low_watermark: float = Field(default=20.0, description="低水位阈值")
    batch_size: int = Field(default=10, description="每次最多归档数量")
    enable_schedule: bool = Field(default=False, description="是否启用定时垃圾回收")
    interval_hours: int = Field(default=24, description="执行间隔(小时)")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class MemoryLifecycleConfig(BaseSettings):
    """记忆生命周期统一配置"""
    vitality_calculator: VitalityCalculatorConfig = Field(default_factory=VitalityCalculatorConfig, description="生命力计算器配置")
    reinforcement_engine: ReinforcementEngineConfig = Field(default_factory=ReinforcementEngineConfig, description="强化引擎配置")
    archiver: ArchiverConfig = Field(default_factory=ArchiverConfig, description="归档器配置")
    garbage_collector: GarbageCollectorConfig = Field(default_factory=GarbageCollectorConfig, description="垃圾回收器配置")
    high_watermark: float = Field(default=80.0, description="高水位阈值")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


# ========== 系统与日志 ==========

class LoggingConfig(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    file_path: Optional[str] = Field(default=None, description="日志文件路径")
    console_output: bool = Field(default=True, description="是否输出到控制台")

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


class SystemConfig(BaseSettings):
    """系统全局配置"""
    name: str = Field(default="HiveMemory", description="系统名称")
    version: str = Field(default="0.1.0", description="系统版本")
    debug: bool = Field(default=False, description="调试模式")
    
    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


# ========== 主配置类 ==========

class HiveMemoryConfig(BaseSettings):
    """
    HiveMemory 主配置类
    
    加载顺序:
    1. 构造函数参数 (Arguments)
    2. 环境变量 (Environment Variables, 包含 .env)
    3. YAML 配置文件 (Configs)
    4. 默认值 (Defaults)
    """
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    llm: Dict[str, LLMConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    perception: MemoryPerceptionConfig = Field(default_factory=MemoryPerceptionConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    lifecycle: MemoryLifecycleConfig = Field(default_factory=MemoryLifecycleConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix="HIVEMEMORY_"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        自定义配置源优先级:
        Init > Env > DotEnv > YAML > Secrets
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_config_settings_source,
            file_secret_settings,
        )

    # Helper methods
    def get_worker_llm_config(self) -> LLMConfig:
        """
        获取 Worker LLM 配置

        环境变量覆盖: HIVEMEMORY__LLM__WORKER__MODEL, HIVEMEMORY__LLM__WORKER__API_KEY 等
        """
        config = self.llm.get("worker", LLMConfig(model="gpt-4o"))
        return config

    def get_librarian_llm_config(self) -> LLMConfig:
        """
        获取 Librarian LLM 配置

        环境变量覆盖: HIVEMEMORY__LLM__LIBRARIAN__MODEL, HIVEMEMORY__LLM__LIBRARIAN__API_KEY 等
        """
        config = self.llm.get("librarian", LLMConfig(
            model="deepseek/deepseek-chat",
            temperature=0.3,
            max_tokens=8192
        ))
        return config


# ========== 工厂函数 (Factory) ==========

def load_app_config(config_path: Optional[str] = None) -> HiveMemoryConfig:
    """
    加载应用配置 (推荐使用的工厂函数)
    
    Args:
        config_path: 配置文件路径。如果不传，则依次查找:
                    1. 环境变量 HIVEMEMORY_CONFIG_PATH
                    2. 默认 configs/config.yaml
    
    Returns:
        HiveMemoryConfig 实例
    """
    if config_path:
        os.environ["HIVEMEMORY_CONFIG_PATH"] = str(config_path)
    
    # 实例化配置，Pydantic 会自动调用 settings_customise_sources 加载 YAML 和 Env
    return HiveMemoryConfig()


@lru_cache()
def get_config(config_path: Optional[str] = None) -> HiveMemoryConfig:
    """
    [已废弃] 获取全局配置实例 (单例模式)
    建议在代码中使用依赖注入，通过 load_app_config 获取配置后传递给组件。
    """
    logger.warning("Calling deprecated function get_config(). Please use load_app_config() and dependency injection instead.")
    return load_app_config(config_path)


# 导出便捷函数 (Delegates to get_config for backward compatibility)
def get_worker_llm_config() -> LLMConfig:
    return get_config().get_worker_llm_config()

def get_librarian_llm_config() -> LLMConfig:
    return get_config().get_librarian_llm_config()
