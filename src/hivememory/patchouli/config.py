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
from typing import Optional, Any, Dict, List, Tuple, Type, Set, Literal, Union
import yaml
from pydantic import BaseModel, Field, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

logger = logging.getLogger(__name__)

HIVEMEMORY_ENV_PREFIX = "HIVEMEMORY__"

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

class LLMConfig(BaseModel):
    """LLM 模型配置"""
    provider: str = "litellm"
    model: Optional[str] = Field(default=None, description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API Key")
    api_base: Optional[str] = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)

    model_config = ConfigDict(extra="ignore")


class LLMGlobalConfig(BaseModel):
    """LLM 全局配置集合"""
    librarian: LLMConfig = Field(default_factory=lambda: LLMConfig(model="deepseek/deepseek-chat", temperature=0.3, max_tokens=8192))
    gateway: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4o", temperature=0.0, max_tokens=512))

    model_config = ConfigDict(extra="allow")


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding模型名称")
    device: str = Field(default="cpu", description="运行设备: cpu/cuda/mps")
    cache_dir: Optional[str] = Field(default=None, description="模型缓存目录")
    batch_size: int = Field(default=32, description="批处理大小")
    normalize_embeddings: bool = Field(default=True, description="是否归一化向量")
    dimension: int = Field(default=384, description="向量维度")

    model_config = ConfigDict(extra="ignore")


class EmbeddingGlobalConfig(BaseModel):
    """Embedding 全局配置集合"""
    default: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="默认/存储层 Embedding 配置")
    perception: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="感知层 Embedding 配置")

    model_config = ConfigDict(extra="allow")


class QdrantConfig(BaseModel):
    """Qdrant 向量数据库配置"""
    host: str = Field(default="localhost", description="Qdrant 主机地址")
    port: int = Field(default=6333, description="HTTP 端口")
    grpc_port: int = Field(default=6334, description="gRPC 端口")
    api_key: Optional[str] = Field(default=None, description="API Key")
    collection_name: str = Field(default="hivememory_main", description="集合名称")
    vector_dimension: int = Field(default=1024, description="向量维度")
    distance_metric: str = Field(default="Cosine", description="距离度量方式")
    on_disk_payload: bool = Field(default=False, description="是否将 Payload 存储在磁盘")

    model_config = ConfigDict(extra="ignore")


class RedisConfig(BaseModel):
    """Redis 配置"""
    host: str = Field(default="localhost", description="Redis 主机地址")
    port: int = Field(default=6379, description="Redis 端口")
    password: Optional[str] = Field(default=None, description="Redis 密码")
    db: int = Field(default=0, description="数据库索引")
    decode_responses: bool = Field(default=True, description="是否自动解码响应")

    model_config = ConfigDict(extra="ignore")


# ========== Gateway 配置 ==========

class RuleInterceptorConfig(BaseModel):
    """L1 规则拦截器配置"""
    enabled: bool = Field(default=True, description="是否启用 L1 规则拦截器")
    enable_system: bool = Field(default=True, description="是否启用系统指令拦截")
    enable_chat: bool = Field(default=True, description="是否启用闲聊拦截")
    custom_system_patterns: List[str] = Field(default_factory=list, description="自定义系统指令模式")
    custom_chat_patterns: List[str] = Field(default_factory=list, description="自定义闲聊模式")

    model_config = ConfigDict(extra="ignore")


class LLMAnalyzerConfig(BaseModel):
    """L2 语义分析器配置"""
    enabled: bool = Field(default=True, description="是否启用 L2 语义分析")
    context_window: int = Field(default=3, description="上下文窗口大小（最近N条消息）")
    enable_memory_type_filter: bool = Field(default=True, description="是否启用记忆类型过滤")
    prompt_variant: str = Field(default="default", description="System Prompt 变体")
    prompt_language: str = Field(default="zh", description="System Prompt 语言")

    model_config = ConfigDict(extra="ignore")


class MemoryGatewayConfig(BaseModel):
    """
    Global Gateway 配置
    
    对应 GlobalGateway 类的初始化参数
    负责意图分类、查询重写、元数据提取
    """
    interceptor: RuleInterceptorConfig = Field(default_factory=RuleInterceptorConfig, description="L1 拦截器配置")
    analyzer: LLMAnalyzerConfig = Field(default_factory=LLMAnalyzerConfig, description="L2 分析器配置")

    model_config = ConfigDict(extra="ignore")

# ========== 感知层配置 ==========

class RerankerArbiterConfig(BaseModel):
    """基于 Reranker 的仲裁器配置"""
    type: Literal["reranker"] = "reranker"
    threshold: float = Field(default=0.5, description="仲裁阈值")

    model_config = ConfigDict(extra="ignore")


class SLMArbiterConfig(BaseModel):
    """基于 SLM 的仲裁器配置"""
    type: Literal["slm"] = "slm"
    prompt_template: Optional[str] = Field(default=None, description="自定义 Prompt")
    threshold: float = Field(default=0.5, description="不确定时的阈值")

    model_config = ConfigDict(extra="ignore")


class ArbiterConfig(BaseModel):
    """灰度仲裁器统一配置"""
    enabled: bool = Field(default=True, description="是否启用灰度仲裁")
    engine: Union[RerankerArbiterConfig, SLMArbiterConfig] = Field(
        default_factory=RerankerArbiterConfig,
        discriminator="type",
        description="具体仲裁器实现配置"
    )

    model_config = ConfigDict(extra="ignore")


class ContextBridgeConfig(BaseModel):
    """上下文桥接器配置"""
    context_max_length: int = Field(default=200, description="上下文最大长度（tokens）")
    context_source: str = Field(default="user", description="上下文来源（response/history）")

    model_config = ConfigDict(extra="ignore")


class SemanticAdsorberConfig(BaseModel):
    """
    SemanticBoundaryAdsorber 配置
    """
    semantic_threshold_high: float = Field(default=0.75, description="高相似度阈值（强吸附）")
    semantic_threshold_low: float = Field(default=0.40, description="低相似度阈值（强制切分）")
    short_text_threshold: int = Field(default=10, description="短文本强吸附阈值（tokens）")
    ema_alpha: float = Field(default=0.3, description="指数移动平均系数")
    
    arbiter: ArbiterConfig = Field(default_factory=ArbiterConfig, description="灰度仲裁器配置")
    context_bridge: ContextBridgeConfig = Field(default_factory=ContextBridgeConfig, description="上下文桥接器配置")

    stop_words: Optional[Set[str]] = Field(default=None, description="自定义停用词集合")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode='after')
    def validate_thresholds(self) -> 'SemanticAdsorberConfig':
        if self.semantic_threshold_low > self.semantic_threshold_high:
            raise ValueError("semantic_threshold_low 必须小于或等于 semantic_threshold_high")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha 必须在 (0, 1] 范围内")
        return self


class SimplePerceptionConfig(BaseModel):
    """SimplePerceptionLayer 配置"""
    type: Literal["simple"] = Field(default="simple", description="感知层类型")
    message_count_threshold: int = Field(default=6, description="消息数触发阈值")
    enable_semantic_trigger: bool = Field(default=True, description="是否启用语义边界触发")
    idle_timeout_seconds: int = Field(default=900, description="空闲超时时间（秒）")
    scan_interval_seconds: int = Field(default=30, description="空闲监控扫描间隔（秒）")

    model_config = ConfigDict(extra="ignore")

class SemanticFlowPerceptionConfig(BaseModel):
    """SemanticFlowPerceptionLayer 配置"""
    type: Literal["semantic_flow"] = Field(default="semantic_flow", description="感知层类型")
    max_processing_tokens: int = Field(default=8192, description="单次处理的最大 Token 数")
    enable_smart_summary: bool = Field(default=False, description="是否启用智能摘要")
    idle_timeout_seconds: int = Field(default=900, description="空闲超时时间（秒）")
    scan_interval_seconds: int = Field(default=30, description="空闲监控扫描间隔（秒）")

    adsorber: SemanticAdsorberConfig = Field(default_factory=SemanticAdsorberConfig, description="语义吸附器配置")

    model_config = ConfigDict(extra="ignore")


class MemoryPerceptionConfig(BaseModel):
    """感知层统一配置"""
    engine: Literal["semantic_flow", "simple"] = Field(
        default="semantic_flow", 
        description="感知层引擎: semantic_flow 或 simple"
    )

    model_config = ConfigDict(extra="ignore")


# ========== 记忆生成配置 ==========

class ExtractorConfig(BaseModel):
    """LLMMemoryExtractor 配置"""
    enabled: bool = Field(default=True, description="是否启用 LLM 提取器")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    user_prompt: Optional[str] = Field(default=None, description="自定义用户提示词")

    model_config = ConfigDict(extra="ignore")


class DeduplicatorConfig(BaseModel):
    """查重器配置"""
    enabled: bool = Field(default=True, description="是否启用查重器")
    high_similarity_threshold: float = Field(default=0.95, description="高相似度阈值（TOUCH/UPDATE 分界）")
    low_similarity_threshold: float = Field(default=0.75, description="低相似度阈值（UPDATE/CREATE 分界）")
    content_similarity_threshold: float = Field(default=0.9, description="内容相似度阈值")
    enable_vitality_tracking: bool = Field(default=True, description="是否启用生命周期追踪")

    model_config = ConfigDict(extra="ignore")


class MemoryGenerationConfig(BaseModel):
    """记忆生成统一配置"""
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig, description="LLM 提取器配置")
    deduplicator: DeduplicatorConfig = Field(default_factory=DeduplicatorConfig, description="查重器配置")

    model_config = ConfigDict(extra="ignore")


# ========== 记忆检索配置 ==========

class ReciprocalRankFusionConfig(BaseModel):
    """RRF 融合配置"""
    type: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, description="RRF常数")
    dense_weight: float = Field(default=1.0, description="稠密检索权重")
    sparse_weight: float = Field(default=1.0, description="稀疏检索权重")
    final_top_k: int = Field(default=5, description="最终返回数量")

    model_config = ConfigDict(extra="ignore")


class RetrievalModeConfig(BaseModel):
    """
    预设检索模式配置

    用于 AdaptiveWeightedFusion 的各检索模式参数配置。
    包含权重分配和质量乘数计算参数。
    """
    # 权重分配
    dense_weight: float = Field(default=0.6, description="稠密检索权重")
    sparse_weight: float = Field(default=0.4, description="稀疏检索权重")
    time_weight: float = Field(default=0.0, description="时间权重 (预留)")

    # 置信度惩罚
    confidence_penalty_enabled: bool = Field(default=True, description="是否启用置信度惩罚")
    confidence_penalty_threshold: float = Field(default=0.6, description="低于此值触发惩罚")
    confidence_penalty_factor: float = Field(default=0.5, description="惩罚系数 (乘以此值)")

    # 生命力加成
    vitality_boost_enabled: bool = Field(default=True, description="是否启用生命力加成")
    vitality_high_threshold: float = Field(default=80.0, description="高生命力阈值")
    vitality_high_factor: float = Field(default=1.2, description="高生命力加成系数")
    vitality_low_threshold: float = Field(default=30.0, description="低生命力阈值")
    vitality_low_factor: float = Field(default=0.8, description="低生命力惩罚系数")

    model_config = ConfigDict(extra="ignore")


class AdaptiveWeightedFusionConfig(BaseModel):
    """
    自适应加权融合配置

    支持多种预设检索模式:
    - debug: 高 sparse 权重，强置信度惩罚 (精确匹配场景)
    - concept: 高 dense 权重，弱惩罚 (概念理解场景)
    - timeline: 高 time 权重，中等惩罚 (时间相关场景)
    - brainstorm: 高 dense 权重，无惩罚 (发散思维场景)
    """
    type: Literal["adaptive"] = "adaptive"
    final_top_k: int = Field(default=5, description="最终返回数量")
    default_mode: str = Field(default="concept", description="默认检索模式")

    # 预设模式配置
    debug_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.3,
            sparse_weight=0.9,
            time_weight=0.1,
            confidence_penalty_enabled=True,
            confidence_penalty_threshold=0.6,
            confidence_penalty_factor=0.5,
            vitality_boost_enabled=True,
        ),
        description="Debug 模式: 高 sparse, 强惩罚"
    )

    concept_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.8,
            sparse_weight=0.2,
            time_weight=0.1,
            confidence_penalty_enabled=True,
            confidence_penalty_threshold=0.5,
            confidence_penalty_factor=0.7,
            vitality_boost_enabled=True,
        ),
        description="Concept 模式: 高 dense, 弱惩罚"
    )

    timeline_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.4,
            sparse_weight=0.3,
            time_weight=0.8,
            confidence_penalty_enabled=True,
            confidence_penalty_threshold=0.6,
            confidence_penalty_factor=0.6,
            vitality_boost_enabled=True,
        ),
        description="Timeline 模式: 高 time, 中等惩罚"
    )

    brainstorm_mode: RetrievalModeConfig = Field(
        default_factory=lambda: RetrievalModeConfig(
            dense_weight=0.6,
            sparse_weight=0.1,
            time_weight=0.0,
            confidence_penalty_enabled=False,
            vitality_boost_enabled=False,
        ),
        description="Brainstorm 模式: 高 dense, 无惩罚"
    )

    model_config = ConfigDict(extra="ignore")


class RerankerConfig(BaseModel):
    """重排序器配置"""
    enabled: bool = Field(default=True, description="是否启用重排序")
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker 模型名称")
    device: str = Field(default="cpu", description="运行设备: cpu/cuda")
    use_fp16: bool = Field(default=True, description="是否使用 FP16 精度")
    batch_size: int = Field(default=32, description="批处理大小")
    top_k: int = Field(default=20, description="仅重排序前N个结果")
    normalize_scores: bool = Field(default=True, description="是否标准化分数到 0-1")

    model_config = ConfigDict(extra="ignore")


class DenseRetrieverConfig(BaseModel):
    """稠密检索配置"""
    type: Literal["dense"] = "dense"
    enabled: bool = Field(default=True, description="是否启用稠密检索")
    top_k: int = Field(default=50, description="RRF融合前的召回数量")
    score_threshold: float = Field(default=0.0, description="相似度阈值")
    enable_time_decay: bool = Field(default=True, description="是否启用时间衰减")
    time_decay_days: int = Field(default=30, description="时间衰减半衰期(天)")
    enable_confidence_boost: bool = Field(default=True, description="是否启用置信度加权")

    model_config = ConfigDict(extra="ignore")


class SparseRetrieverConfig(BaseModel):
    """稀疏检索配置"""
    type: Literal["sparse"] = "sparse"
    enabled: bool = Field(default=True, description="是否启用稀疏检索")
    top_k: int = Field(default=50, description="RRF融合前的召回数量")
    score_threshold: float = Field(default=0.0, description="相似度阈值")

    model_config = ConfigDict(extra="ignore")


class HybridRetrieverConfig(BaseModel):
    """混合检索完整配置"""
    type: Literal["hybrid"] = "hybrid"
    top_k: int = Field(default=5, description="最终返回数量")
    score_threshold: float = Field(default=0.75, description="相似度阈值")
    enable_parallel: bool = Field(default=True, description="是否启用并行召回")

    dense: DenseRetrieverConfig = Field(default_factory=DenseRetrieverConfig, description="稠密检索配置")
    sparse: SparseRetrieverConfig = Field(default_factory=SparseRetrieverConfig, description="稀疏检索配置")
    fusion: Union[ReciprocalRankFusionConfig, AdaptiveWeightedFusionConfig] = Field(
        default_factory=ReciprocalRankFusionConfig, 
        description="融合配置", 
        discriminator="type"
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig, description="重排序配置")

    model_config = ConfigDict(extra="ignore")

class FullRendererConfig(BaseModel):
    """
    完整上下文渲染器配置

    渲染所有 MemoryAtom 的完整内容，超过字符上限则直接截断。
    """
    type: Literal["full"] = "full"
    render_format: str = Field(default="xml", description="渲染格式: xml/markdown")
    max_tokens: int = Field(default=2000, description="最大字符数")
    max_content_length: int = Field(default=500, description="单条记忆最大内容长度")
    show_artifacts: bool = Field(default=False, description="是否显示原始数据链接")
    stale_days: int = Field(default=90, description="记忆被视为陈旧的天数")

    model_config = ConfigDict(extra="ignore")


class CascadeRendererConfig(BaseModel):
    """
    瀑布式上下文渲染器配置

    依次完整渲染 MemoryAtom，直到 Token 预算紧张时降级为 Index 层信息。
    - Top-N 记忆强制完整渲染
    - 其余按预算瀑布式降级为 Index 视图
    """
    type: Literal["cascade"] = "cascade"
    max_memory_tokens: int = Field(default=2000, description="最大记忆 Token 预算")
    full_payload_count: int = Field(default=1, description="强制完整渲染的数量 (Top-N)")
    render_format: str = Field(default="xml", description="渲染格式: xml/markdown")
    max_content_length: int = Field(default=500, description="完整渲染时单条记忆最大内容长度")
    index_max_summary_length: int = Field(default=100, description="Index 视图摘要最大长度")
    enable_lazy_loading: bool = Field(default=False, description="是否启用懒加载提示")
    lazy_load_tool_name: str = Field(default="read_memory", description="懒加载工具名称")
    lazy_load_hint: str = Field(default="如需完整内容，请使用 read_memory(id) 工具", description="懒加载提示文本")

    model_config = ConfigDict(extra="ignore")


class CompactRendererConfig(BaseModel):
    """
    紧凑上下文渲染器配置

    仅渲染 Index 层信息 (摘要+标签)，不渲染完整 Payload。
    """
    type: Literal["compact"] = "compact"
    max_memory_tokens: int = Field(default=2000, description="最大记忆 Token 预算")
    render_format: str = Field(default="xml", description="渲染格式: xml/markdown")
    index_max_summary_length: int = Field(default=100, description="Index 视图摘要最大长度")
    enable_lazy_loading: bool = Field(default=True, description="是否启用懒加载提示")
    lazy_load_tool_name: str = Field(default="read_memory", description="懒加载工具名称")
    lazy_load_hint: str = Field(default="如需完整内容，请使用 read_memory(id) 工具", description="懒加载提示文本")

    model_config = ConfigDict(extra="ignore")


class MemoryRetrievalConfig(BaseModel):
    """记忆检索统一配置"""
    renderer: Union[FullRendererConfig, CascadeRendererConfig, CompactRendererConfig] = Field(
        default_factory=FullRendererConfig,
        description="上下文渲染器配置",
        discriminator="type"
    )
    retriever: Union[HybridRetrieverConfig, DenseRetrieverConfig, SparseRetrieverConfig] = Field(
        default_factory=HybridRetrieverConfig,
        description="检索器配置",
        discriminator="type"
    )

    model_config = ConfigDict(extra="ignore")


# ========== 记忆生命周期配置 ==========

class VitalityCalculatorConfig(BaseModel):
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

    model_config = ConfigDict(extra="ignore")


class ReinforcementEngineConfig(BaseModel):
    """强化引擎配置"""
    enable_event_history: bool = Field(default=True, description="是否记录事件历史")
    event_history_limit: int = Field(default=10000, description="事件历史最大条数")
    hit_boost: float = Field(default=5.0, description="HIT 事件加成")
    citation_boost: float = Field(default=20.0, description="CITATION 事件加成")
    positive_feedback_boost: float = Field(default=50.0, description="正面反馈加成")
    negative_feedback_penalty: float = Field(default=-50.0, description="负面反馈惩罚")
    negative_confidence_multiplier: float = Field(default=0.5, description="负面反馈置信度衰减系数")

    model_config = ConfigDict(extra="ignore")


class ArchiverConfig(BaseModel):
    """归档器配置"""
    archive_dir: str = Field(default="data/archived", description="归档目录路径")
    compression: bool = Field(default=True, description="是否使用 GZIP 压缩")

    model_config = ConfigDict(extra="ignore")


class GarbageCollectorConfig(BaseModel):
    """垃圾回收器配置"""
    low_watermark: float = Field(default=20.0, description="低水位阈值")
    batch_size: int = Field(default=10, description="每次最多归档数量")
    enable_schedule: bool = Field(default=False, description="是否启用定时垃圾回收")
    interval_hours: int = Field(default=24, description="执行间隔(小时)")

    model_config = ConfigDict(extra="ignore")


class MemoryLifecycleConfig(BaseModel):
    """记忆生命周期统一配置"""
    vitality_calculator: VitalityCalculatorConfig = Field(default_factory=VitalityCalculatorConfig, description="生命力计算器配置")
    reinforcement: ReinforcementEngineConfig = Field(default_factory=ReinforcementEngineConfig, description="强化引擎配置")
    archiver: ArchiverConfig = Field(default_factory=ArchiverConfig, description="归档器配置")
    garbage_collector: GarbageCollectorConfig = Field(default_factory=GarbageCollectorConfig, description="垃圾回收器配置")

    high_watermark: float = Field(default=80.0, description="高水位阈值")

    model_config = ConfigDict(extra="ignore")


# ========== 系统与日志 ==========

class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    file_path: Optional[str] = Field(default=None, description="日志文件路径")
    console_output: bool = Field(default=True, description="是否输出到控制台")

    model_config = ConfigDict(extra="ignore")


class SystemConfig(BaseModel):
    """系统全局配置"""
    name: str = Field(default="HiveMemory", description="系统名称")
    version: str = Field(default="0.1.0", description="系统版本")
    debug: bool = Field(default=False, description="调试模式")

    model_config = ConfigDict(extra="ignore")


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

    llm: LLMGlobalConfig = Field(default_factory=LLMGlobalConfig)
    embedding: EmbeddingGlobalConfig = Field(default_factory=EmbeddingGlobalConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    gateway: MemoryGatewayConfig = Field(default_factory=MemoryGatewayConfig)
    perception: MemoryPerceptionConfig = Field(default_factory=MemoryPerceptionConfig)
    generation: MemoryGenerationConfig = Field(default_factory=MemoryGenerationConfig)
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    lifecycle: MemoryLifecycleConfig = Field(default_factory=MemoryLifecycleConfig)

    model_config = SettingsConfigDict(
        env_file=(".env", "configs/.env", "configs\\.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix=HIVEMEMORY_ENV_PREFIX
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
    def get_librarian_llm_config(self) -> LLMConfig:
        """
        获取 Librarian LLM 配置

        环境变量覆盖: HIVEMEMORY__LLM__LIBRARIAN__MODEL, HIVEMEMORY__LLM__LIBRARIAN__API_KEY 等
        """
        return self.llm.librarian

    def get_gateway_llm_config(self) -> LLMConfig:
        """
        获取 Gateway LLM 配置

        环境变量覆盖: HIVEMEMORY__LLM__GATEWAY__MODEL, HIVEMEMORY__LLM__GATEWAY__API_KEY 等
        """
        return self.llm.gateway


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


def get_librarian_llm_config() -> LLMConfig:
    """
    便捷函数: 获取 Librarian LLM 配置
    自动加载全局配置并返回 librarian 部分
    """
    return load_app_config().get_librarian_llm_config()


def get_gateway_llm_config() -> LLMConfig:
    """
    便捷函数: 获取 Gateway LLM 配置
    自动加载全局配置并返回 gateway 部分
    """
    return load_app_config().get_gateway_llm_config()
