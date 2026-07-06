import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Type
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

from hivememory.system.config.shared import (
    LLMConfig, LLMGlobalConfig,
    EmbeddingConfig, EmbeddingGlobalConfig,
    ProviderCredentials,
    SharedConfig,
)
from hivememory.system.config.gateway import (
    RuleInterceptorConfig, LLMAnalyzerConfig, SystemCommandConfig, SystemGatewayConfig,
)
from hivememory.system.config.patchouli import (
    QdrantConfig,
    SimpleRelayConfig, LLMRelayConfig, RelayControllerConfig,
    SemanticFlowPerceptionConfig, MemoryPerceptionConfig,
    ExtractorConfig, DeduplicatorConfig, MemoryGenerationConfig,
    ReciprocalRankFusionConfig, RetrievalModeConfig, AdaptiveWeightedFusionConfig,
    RerankerConfig, DenseRetrieverConfig, SparseRetrieverConfig, HybridRetrieverConfig,
    MemoryRetrievalConfig,
    VitalityCalculatorConfig, ReinforcementEngineConfig, ArchiverConfig,
    GarbageCollectorConfig, MemoryLifecycleConfig,
    ArtifactComponentConfig, ArtifactConfig,
    PatchouliShutdownConfig,
    PatchouliConfig,
)
from hivememory.system.config.memory_compiler import (
    CascadeContextStrategyConfig,
    CompactContextStrategyConfig,
    FullContextStrategyConfig,
    MemoryCompilerConfig,
    RetrievalContextCompileConfig,
    RetrievalContextStrategyConfig,
)
from hivememory.system.config.alice import (
    MTPPromptConfig, KoakumaConfig, AgentRuntimeConfig,
    AliceConfig,
)

logger = logging.getLogger(__name__)

HIVEMEMORY_ENV_PREFIX = "HIVEMEMORY__"
LEGACY_ENV_ALIASES: Dict[str, Tuple[str, ...]] = {
    "LLM__GATEWAY__PROVIDER": ("shared", "llm", "gateway", "provider"),
    "LLM__GATEWAY__MODEL": ("shared", "llm", "gateway", "model"),
    "LLM__GATEWAY__API_KEY": ("shared", "llm", "gateway", "api_key"),
    "LLM__GATEWAY__API_BASE": ("shared", "llm", "gateway", "api_base"),
    "LLM__GATEWAY__TEMPERATURE": ("shared", "llm", "gateway", "temperature"),
    "LLM__GATEWAY__MAX_TOKENS": ("shared", "llm", "gateway", "max_tokens"),
    "LLM__LIBRARIAN__PROVIDER": ("shared", "llm", "librarian", "provider"),
    "LLM__LIBRARIAN__MODEL": ("shared", "llm", "librarian", "model"),
    "LLM__LIBRARIAN__API_KEY": ("shared", "llm", "librarian", "api_key"),
    "LLM__LIBRARIAN__API_BASE": ("shared", "llm", "librarian", "api_base"),
    "LLM__LIBRARIAN__TEMPERATURE": ("shared", "llm", "librarian", "temperature"),
    "LLM__LIBRARIAN__MAX_TOKENS": ("shared", "llm", "librarian", "max_tokens"),
    "QDRANT__HOST": ("patchouli", "storage", "host"),
    "QDRANT__PORT": ("patchouli", "storage", "port"),
    "QDRANT__GRPC_PORT": ("patchouli", "storage", "grpc_port"),
    "QDRANT__PREFER_GRPC": ("patchouli", "storage", "prefer_grpc"),
    "QDRANT__TIMEOUT": ("patchouli", "storage", "timeout"),
    "QDRANT__DEPLOYMENT": ("patchouli", "storage", "deployment"),
    "QDRANT__DATA_DIR": ("patchouli", "storage", "data_dir"),
    "QDRANT__BINARY_PATH": ("patchouli", "storage", "binary_path"),
    "QDRANT__STARTUP_TIMEOUT_SECONDS": ("patchouli", "storage", "startup_timeout_seconds"),
    "QDRANT__API_KEY": ("patchouli", "storage", "api_key"),
    "QDRANT__COLLECTION_NAME": ("patchouli", "storage", "collection_name"),
    "QDRANT__VECTOR_DIMENSION": ("patchouli", "storage", "vector_dimension"),
    "QDRANT__DISTANCE_METRIC": ("patchouli", "storage", "distance_metric"),
    "QDRANT__ON_DISK_PAYLOAD": ("patchouli", "storage", "on_disk_payload"),
}


def get_default_config_file_path() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent / "configs" / "config.yaml"


def get_config_file_path() -> Path:
    configured_path = os.getenv("HIVEMEMORY_CONFIG_PATH")
    if configured_path:
        return Path(configured_path)
    return get_default_config_file_path()


def _set_nested_value(data: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    target = data
    for part in path[:-1]:
        next_target = target.get(part)
        if not isinstance(next_target, dict):
            next_target = {}
            target[part] = next_target
        target = next_target
    target[path[-1]] = value


def _load_dotenv_sources() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for env_file in (".env", "configs/.env", "configs\\.env"):
        path = Path(env_file)
        if not path.exists():
            continue
        for key, value in dotenv_values(path, encoding="utf-8").items():
            if value is not None:
                values[key] = value
    return values


def legacy_env_alias_settings_source() -> Dict[str, Any]:
    """Map pre-config-split env vars into the current nested config schema."""
    raw_values = _load_dotenv_sources()
    raw_values.update(os.environ)
    normalized = {key.upper(): value for key, value in raw_values.items()}

    aliased: Dict[str, Any] = {}
    for legacy_key, target_path in LEGACY_ENV_ALIASES.items():
        value = normalized.get(f"{HIVEMEMORY_ENV_PREFIX}{legacy_key}")
        if value is not None:
            _set_nested_value(aliased, target_path, value)
    return aliased


def provider_credentials_settings_source() -> Dict[str, Any]:
    """扫描 HIVEMEMORY__PROVIDERS__<NAME>__API_KEY / __API_BASE 环境变量。

    provider 名是动态的（deepseek / openai / anthropic ...），无法用静态别名表
    枚举，故单独扫描：把匹配的变量映射到 shared.providers.<name>.{api_key,api_base}。
    provider 名统一小写，以便与 ModelDefinition.provider 匹配。
    """
    raw_values = _load_dotenv_sources()
    raw_values.update(os.environ)

    prefix = f"{HIVEMEMORY_ENV_PREFIX}PROVIDERS__"
    result: Dict[str, Any] = {}
    for key, value in raw_values.items():
        if value is None:
            continue
        upper = key.upper()
        if not upper.startswith(prefix):
            continue
        # 剩余部分形如 "DEEPSEEK__API_KEY"
        remainder = upper[len(prefix):]
        parts = remainder.split("__")
        if len(parts) != 2:
            continue
        provider_name, field = parts[0].lower(), parts[1].lower()
        if field not in ("api_key", "api_base"):
            continue
        _set_nested_value(result, ("shared", "providers", provider_name, field), value)
    return result


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
    gateway: SystemGatewayConfig = Field(default_factory=SystemGatewayConfig)
    memory_compiler: MemoryCompilerConfig = Field(default_factory=MemoryCompilerConfig)
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
            legacy_env_alias_settings_source,
            provider_credentials_settings_source,
            yaml_config_settings_source,
            file_secret_settings,
        )

    def get_librarian_llm_config(self) -> LLMConfig:
        return self.shared.llm.librarian

    def get_gateway_llm_config(self) -> LLMConfig:
        return self.shared.llm.gateway


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
    "ProviderCredentials",
    "SharedConfig",
    # gateway
    "RuleInterceptorConfig", "LLMAnalyzerConfig", "SystemCommandConfig", "SystemGatewayConfig",
    # patchouli
    "QdrantConfig",
    "SimpleRelayConfig", "LLMRelayConfig", "RelayControllerConfig",
    "SemanticFlowPerceptionConfig", "MemoryPerceptionConfig",
    "ExtractorConfig", "DeduplicatorConfig", "MemoryGenerationConfig",
    "ReciprocalRankFusionConfig", "RetrievalModeConfig", "AdaptiveWeightedFusionConfig",
    "RerankerConfig", "DenseRetrieverConfig", "SparseRetrieverConfig", "HybridRetrieverConfig",
    "MemoryRetrievalConfig",
    "VitalityCalculatorConfig", "ReinforcementEngineConfig", "ArchiverConfig",
    "GarbageCollectorConfig", "MemoryLifecycleConfig",
    "ArtifactComponentConfig", "ArtifactConfig",
    "PatchouliShutdownConfig",
    "PatchouliConfig",
    # memory compiler
    "FullContextStrategyConfig",
    "CascadeContextStrategyConfig",
    "CompactContextStrategyConfig",
    "RetrievalContextStrategyConfig",
    "RetrievalContextCompileConfig",
    "MemoryCompilerConfig",
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
    "LEGACY_ENV_ALIASES",
    "get_config_file_path", "get_default_config_file_path",
    "yaml_config_settings_source", "legacy_env_alias_settings_source",
    "provider_credentials_settings_source",
]
