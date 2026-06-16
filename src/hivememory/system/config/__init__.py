import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Type
import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

from hivememory.system.config.shared import (
    LLMConfig, LLMGlobalConfig,
    EmbeddingConfig, EmbeddingGlobalConfig,
    SharedConfig,
)
from hivememory.system.config.patchouli import (
    QdrantConfig,
    RuleInterceptorConfig, LLMAnalyzerConfig, MemoryGatewayConfig,
    SimpleRelayConfig, LLMRelayConfig, RelayControllerConfig,
    SemanticFlowPerceptionConfig, MemoryPerceptionConfig,
    ExtractorConfig, DeduplicatorConfig, MemoryGenerationConfig,
    ReciprocalRankFusionConfig, RetrievalModeConfig, AdaptiveWeightedFusionConfig,
    RerankerConfig, DenseRetrieverConfig, SparseRetrieverConfig, HybridRetrieverConfig,
    FullRendererConfig, CascadeRendererConfig, CompactRendererConfig, MemoryRetrievalConfig,
    VitalityCalculatorConfig, ReinforcementEngineConfig, ArchiverConfig,
    GarbageCollectorConfig, MemoryLifecycleConfig,
    ArtifactStoreConfig,
    PatchouliConfig,
)
from hivememory.system.config.alice import (
    MTPPromptConfig, KoakumaConfig, AgentRuntimeConfig,
    AliceConfig,
)

logger = logging.getLogger(__name__)

HIVEMEMORY_ENV_PREFIX = "HIVEMEMORY__"


def get_default_config_file_path() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent / "configs" / "config.yaml"


def get_config_file_path() -> Path:
    configured_path = os.getenv("HIVEMEMORY_CONFIG_PATH")
    if configured_path:
        return Path(configured_path)
    return get_default_config_file_path()


def yaml_config_settings_source() -> Dict[str, Any]:
    default_path = get_default_config_file_path()
    path = get_config_file_path()

    if not path.exists():
        if str(path) == str(default_path):
            logger.warning(f"默认配置文件未找到: {path}, 将使用默认值和环境变量")
            return {}
        raise FileNotFoundError(f"配置文件不存在: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f) or {}
        return yaml_content
    except Exception as e:
        logger.error(f"加载 YAML 配置文件失败: {e}")
        return {}


# ========== Top-level infrastructure configs ==========

class SystemConfig(BaseModel):
    name: str = Field(default="HiveMemory")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    model_config = ConfigDict(extra="ignore")


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None)
    console_output: bool = Field(default=True)
    websocket_enabled: bool = Field(default=False)
    websocket_namespaces: List[str] = Field(default_factory=lambda: ["hivememory.*"])
    websocket_level: str = Field(default="INFO")
    websocket_buffer_size: int = Field(default=100)
    websocket_max_rate: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


class MaintenanceTasksConfig(BaseModel):
    observer_idle_flush_interval_seconds: float = Field(default=5.0)
    observer_idle_flush_timeout_seconds: float = Field(default=30.0)
    enable_observer_idle_flush: bool = Field(default=True)
    perception_idle_flush_interval_seconds: float = Field(default=30.0)
    enable_perception_idle_flush: bool = Field(default=True)
    lifecycle_gc_interval_hours: int = Field(default=24)
    enable_lifecycle_gc: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class SchedulerConfig(BaseModel):
    enabled: bool = Field(default=True)
    tick_seconds: float = Field(default=1.0)
    shutdown_wait_seconds: float = Field(default=5.0)
    tasks: MaintenanceTasksConfig = Field(default_factory=MaintenanceTasksConfig)

    model_config = ConfigDict(extra="ignore")


class RuntimeEventsConfig(BaseModel):
    enabled: bool = Field(default=True)
    buffer_size: int = Field(default=1000)
    subscriber_queue_size: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


class I18nConfig(BaseModel):
    default_language: str = Field(default="zh")
    fallback_language: str = Field(default="en")
    supported_languages: List[str] = Field(default_factory=lambda: ["zh", "en"])

    model_config = ConfigDict(extra="ignore")


# ========== Root config ==========

class HiveMemoryConfig(BaseSettings):
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    runtime_events: RuntimeEventsConfig = Field(default_factory=RuntimeEventsConfig)
    i18n: I18nConfig = Field(default_factory=I18nConfig)

    shared: SharedConfig = Field(default_factory=SharedConfig)
    patchouli: PatchouliConfig = Field(default_factory=PatchouliConfig)
    alice: AliceConfig = Field(default_factory=AliceConfig)

    model_config = SettingsConfigDict(
        env_file=(".env", "configs/.env", "configs\\.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix=HIVEMEMORY_ENV_PREFIX,
    )

    @model_validator(mode="after")
    def sync_i18n_default_language(self) -> "HiveMemoryConfig":
        from hivememory.i18n.resolver import set_default_language
        set_default_language(self.i18n.default_language)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_config_settings_source,
            file_secret_settings,
        )

    def get_librarian_llm_config(self) -> LLMConfig:
        return self.shared.llm.librarian

    def get_gateway_llm_config(self) -> LLMConfig:
        return self.shared.llm.gateway

    def get_worker_llm_config(self) -> LLMConfig:
        return self.shared.llm.worker


def load_app_config(config_path: Optional[str] = None) -> HiveMemoryConfig:
    if config_path:
        os.environ["HIVEMEMORY_CONFIG_PATH"] = str(config_path)
    return HiveMemoryConfig()


def get_librarian_llm_config() -> LLMConfig:
    return load_app_config().get_librarian_llm_config()


def get_gateway_llm_config() -> LLMConfig:
    return load_app_config().get_gateway_llm_config()


__all__ = [
    # shared
    "LLMConfig", "LLMGlobalConfig",
    "EmbeddingConfig", "EmbeddingGlobalConfig",
    "SharedConfig",
    # patchouli
    "QdrantConfig",
    "RuleInterceptorConfig", "LLMAnalyzerConfig", "MemoryGatewayConfig",
    "SimpleRelayConfig", "LLMRelayConfig", "RelayControllerConfig",
    "SemanticFlowPerceptionConfig", "MemoryPerceptionConfig",
    "ExtractorConfig", "DeduplicatorConfig", "MemoryGenerationConfig",
    "ReciprocalRankFusionConfig", "RetrievalModeConfig", "AdaptiveWeightedFusionConfig",
    "RerankerConfig", "DenseRetrieverConfig", "SparseRetrieverConfig", "HybridRetrieverConfig",
    "FullRendererConfig", "CascadeRendererConfig", "CompactRendererConfig", "MemoryRetrievalConfig",
    "VitalityCalculatorConfig", "ReinforcementEngineConfig", "ArchiverConfig",
    "GarbageCollectorConfig", "MemoryLifecycleConfig",
    "ArtifactStoreConfig",
    "PatchouliConfig",
    # alice
    "MTPPromptConfig", "KoakumaConfig", "AgentRuntimeConfig",
    "AliceConfig",
    # top-level
    "SystemConfig", "LoggingConfig",
    "MaintenanceTasksConfig", "SchedulerConfig",
    "RuntimeEventsConfig", "I18nConfig",
    "HiveMemoryConfig",
    # factory
    "load_app_config", "get_librarian_llm_config", "get_gateway_llm_config",
    "HIVEMEMORY_ENV_PREFIX",
    "get_config_file_path", "get_default_config_file_path",
    "yaml_config_settings_source",
]
