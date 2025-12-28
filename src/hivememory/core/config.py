"""
HiveMemory 配置管理系统

支持:
- 从 YAML 文件加载配置
- 从环境变量覆盖配置
- 配置验证与类型检查
"""

import os
from pathlib import Path
from typing import Optional, Any, Dict
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


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
    model_name: str = Field(default="BAAI/bge-m3", description="Embedding模型名称")
    device: str = Field(default="cpu", description="运行设备: cpu/cuda/mps")
    batch_size: int = Field(default=32, gt=0)
    normalize_embeddings: bool = Field(default=True)
    dimension: int = Field(default=1024, gt=0, description="向量维度")

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


class BufferConfig(BaseSettings):
    """对话缓冲配置"""
    max_messages: int = Field(default=5, gt=0, description="最大消息数")
    timeout_seconds: int = Field(default=900, gt=0, description="超时时间(秒)")

    model_config = SettingsConfigDict(extra="allow")


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


class ExtractionConfig(BaseSettings):
    """记忆提取配置"""
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    max_tags: int = Field(default=5, gt=0)

    model_config = SettingsConfigDict(extra="allow")


class MemoryConfig(BaseSettings):
    """记忆管理总配置"""
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    model_config = SettingsConfigDict(extra="allow")


class RetrievalConfig(BaseSettings):
    """检索配置"""
    top_k: int = Field(default=5, gt=0)
    score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    rerank_model: Optional[str] = Field(default=None)
    enable_hybrid_search: bool = Field(default=True)

    model_config = SettingsConfigDict(extra="allow")


class APIConfig(BaseSettings):
    """API 服务配置"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, gt=0, lt=65536)
    reload: bool = Field(default=True)
    workers: int = Field(default=1, gt=0)

    model_config = SettingsConfigDict(extra="allow")


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


class HiveMemoryConfig(BaseSettings):
    """
    HiveMemory 主配置类

    加载优先级:
    1. 环境变量 (.env)
    2. YAML 配置文件
    3. 默认值
    """
    system: SystemConfig = Field(default_factory=SystemConfig)
    llm: Dict[str, LLMConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

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
    "LLMConfig",
    "EmbeddingConfig",
    "QdrantConfig",
    "RedisConfig",
    "MemoryConfig",
    "RetrievalConfig",
    "APIConfig",
    "LoggingConfig",
    "get_config",
    "get_worker_llm_config",
    "get_librarian_llm_config",
]
