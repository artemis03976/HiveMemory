"""应用根配置与加载：HiveMemoryConfig 聚合各配置段，并从 YAML / .env / 环境变量加载。

只供组合根（``hivememory.system``）与入口（``hivememory.server``）使用；其余包
只导入自己需要的配置段模块（``hivememory.config.<section>``），不得持有根配置或
自行调用加载函数（由分层测试守护）。
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from hivememory.config.alice import AliceConfig
from hivememory.config.attachments import AttachmentCompilerConfig, AttachmentParserConfig
from hivememory.config.gateway import SystemGatewayConfig
from hivememory.config.import_bus import ImportBusConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.config.patchouli import PatchouliConfig
from hivememory.config.runtime import RuntimeEventsConfig, SchedulerConfig
from hivememory.config.shared import LLMConfig, SharedConfig
from hivememory.config.workspace import WorkspaceConfig

logger = logging.getLogger(__name__)

HIVEMEMORY_ENV_PREFIX = "HIVEMEMORY__"
LEGACY_ENV_ALIASES: dict[str, tuple[str, ...]] = {
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
    "QDRANT__STARTUP_TIMEOUT_SECONDS": ("patchouli", "storage", "startup_timeout_seconds"),
    "QDRANT__API_KEY": ("patchouli", "storage", "api_key"),
    "QDRANT__COLLECTION_NAME": ("patchouli", "storage", "collection_name"),
    "QDRANT__VECTOR_DIMENSION": ("patchouli", "storage", "vector_dimension"),
    "QDRANT__DISTANCE_METRIC": ("patchouli", "storage", "distance_metric"),
    "QDRANT__ON_DISK_PAYLOAD": ("patchouli", "storage", "on_disk_payload"),
}


def get_default_config_file_path() -> Path:
    # src/hivememory/config/app.py → 仓库根目录
    return Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


def get_config_file_path() -> Path:
    configured_path = os.getenv("HIVEMEMORY_CONFIG_PATH")
    if configured_path:
        return Path(configured_path)
    return get_default_config_file_path()


def _set_nested_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for part in path[:-1]:
        next_target = target.get(part)
        if not isinstance(next_target, dict):
            next_target = {}
            target[part] = next_target
        target = next_target
    target[path[-1]] = value


def _load_dotenv_sources() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_file in (".env", "configs/.env", "configs\\.env"):
        path = Path(env_file)
        if not path.exists():
            continue
        for key, value in dotenv_values(path, encoding="utf-8").items():
            if value is not None:
                values[key] = value
    return values


def environment_value(name: str) -> str | None:
    """按配置加载器的同一来源读取一个变量：进程环境变量优先，其次是 ``.env`` 文件。

    密钥按约定可以写在 ``.env`` 中；本地运行时 ``.env`` 不会导出为进程环境
    变量，只读进程环境变量就读不到写在那里的值。
    """
    value = os.environ.get(name)
    return value if value is not None else _load_dotenv_sources().get(name)


def legacy_env_alias_settings_source() -> dict[str, Any]:
    """把 ``LEGACY_ENV_ALIASES`` 中的扁平环境变量名映射到嵌套配置结构，已有部署无需改名。"""
    raw_values = _load_dotenv_sources()
    raw_values.update(os.environ)
    normalized = {key.upper(): value for key, value in raw_values.items()}

    aliased: dict[str, Any] = {}
    for legacy_key, target_path in LEGACY_ENV_ALIASES.items():
        value = normalized.get(f"{HIVEMEMORY_ENV_PREFIX}{legacy_key}")
        if value is not None:
            _set_nested_value(aliased, target_path, value)
    return aliased


def provider_credentials_settings_source() -> dict[str, Any]:
    """扫描 HIVEMEMORY__PROVIDERS__<NAME>__API_KEY / __API_BASE 环境变量。

    provider 名是动态的（deepseek / openai / anthropic ...），无法用静态别名表
    枚举，故单独扫描：把匹配的变量映射到 shared.providers.<name>.{api_key,api_base}。
    provider 名统一小写，以便与 ModelDefinition.provider 匹配。
    """
    raw_values = _load_dotenv_sources()
    raw_values.update(os.environ)

    prefix = f"{HIVEMEMORY_ENV_PREFIX}PROVIDERS__"
    result: dict[str, Any] = {}
    for key, value in raw_values.items():
        if value is None:
            continue
        upper = key.upper()
        if not upper.startswith(prefix):
            continue
        # 剩余部分形如 "DEEPSEEK__API_KEY"
        remainder = upper[len(prefix) :]
        parts = remainder.split("__")
        if len(parts) != 2:
            continue
        provider_name, field = parts[0].lower(), parts[1].lower()
        if field not in ("api_key", "api_base"):
            continue
        _set_nested_value(result, ("shared", "providers", provider_name, field), value)
    return result


def yaml_config_settings_source() -> dict[str, Any]:
    default_path = get_default_config_file_path()
    path = get_config_file_path()

    if not path.exists():
        if str(path) == str(default_path):
            logger.warning(f"默认配置文件未找到: {path}, 将使用默认值和环境变量")
            return {}
        raise FileNotFoundError(f"配置文件不存在: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"加载 YAML 配置文件失败: {e}")
        return {}
    _reject_unknown_sections(yaml_content, path)
    return yaml_content


def _reject_unknown_sections(yaml_content: Any, path: Path) -> None:
    """配置文件的顶层键必须是已声明的配置段，写错或已删除的段让加载失败。

    根配置对环境变量与 .env 保持宽松（其中有不属于配置树的变量），配置
    文件只承载配置树，因此在这里单独检查，避免已删除的段被静默丢弃。
    """
    if not isinstance(yaml_content, dict):
        return
    unknown = sorted(str(key) for key in set(yaml_content) - set(HiveMemoryConfig.model_fields))
    if unknown:
        raise ValueError(f"配置文件包含未知的顶层配置段 {unknown}: {path}")


class _LoaderSettingsSource(PydanticBaseSettingsSource):
    """把返回整份配置 dict 的装载函数适配为 pydantic-settings source。"""

    def __init__(self, settings_cls: type[BaseSettings], load: Callable[[], dict[str, Any]]):
        super().__init__(settings_cls)
        self._load = load

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # 装载函数一次产出整份 dict，不按字段取值。
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._load()


# ========== 顶层基础设施配置 ==========


class SystemConfig(BaseModel):
    name: str = Field(default="HiveMemory")
    debug: bool = Field(default=False)
    # server 自身作为调用来源经统一认证网关认证时使用的 principal 标识；
    # 必须与 configs/system_principals.yaml 中的接入登记一致。
    server_principal_id: str = Field(default="hivememory:http-server")

    model_config = ConfigDict(extra="forbid")


LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(BaseModel):
    level: LogLevelName = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: str | None = Field(default=None)
    console_output: bool = Field(default=True)
    websocket_enabled: bool = Field(default=False)
    websocket_namespaces: list[str] = Field(default_factory=lambda: ["hivememory.*"])
    websocket_level: LogLevelName = Field(default="INFO")
    websocket_buffer_size: int = Field(default=100)
    websocket_max_rate: int = Field(default=100)

    model_config = ConfigDict(extra="forbid")


class I18nConfig(BaseModel):
    default_language: str = Field(default="zh")

    model_config = ConfigDict(extra="forbid")


# ========== 根配置 ==========


class HiveMemoryConfig(BaseSettings):
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    runtime_events: RuntimeEventsConfig = Field(default_factory=RuntimeEventsConfig)
    i18n: I18nConfig = Field(default_factory=I18nConfig)

    shared: SharedConfig = Field(default_factory=SharedConfig)
    gateway: SystemGatewayConfig = Field(default_factory=SystemGatewayConfig)
    import_bus: ImportBusConfig = Field(default_factory=ImportBusConfig)
    memory_compiler: MemoryCompilerConfig = Field(default_factory=MemoryCompilerConfig)
    patchouli: PatchouliConfig = Field(default_factory=PatchouliConfig)
    alice: AliceConfig = Field(default_factory=AliceConfig)
    attachment_parser: AttachmentParserConfig = Field(default_factory=AttachmentParserConfig)
    attachment_compiler: AttachmentCompilerConfig = Field(default_factory=AttachmentCompilerConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)

    model_config = SettingsConfigDict(
        env_file=(".env", "configs/.env", "configs\\.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix=HIVEMEMORY_ENV_PREFIX,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _LoaderSettingsSource(settings_cls, legacy_env_alias_settings_source),
            _LoaderSettingsSource(settings_cls, provider_credentials_settings_source),
            _LoaderSettingsSource(settings_cls, yaml_config_settings_source),
            file_secret_settings,
        )

    def get_librarian_llm_config(self) -> LLMConfig:
        return self.shared.llm.librarian

    def get_gateway_llm_config(self) -> LLMConfig:
        return self.shared.llm.gateway


def load_app_config(config_path: str | None = None) -> HiveMemoryConfig:
    if config_path:
        os.environ["HIVEMEMORY_CONFIG_PATH"] = str(config_path)
    return HiveMemoryConfig()


def get_librarian_llm_config() -> LLMConfig:
    return load_app_config().get_librarian_llm_config()


def get_gateway_llm_config() -> LLMConfig:
    return load_app_config().get_gateway_llm_config()


__all__ = [
    # Import Bus
    "ImportBusConfig",
    # 顶层
    "SystemConfig",
    "LoggingConfig",
    "I18nConfig",
    "HiveMemoryConfig",
    # 加载与工厂函数
    "load_app_config",
    "get_librarian_llm_config",
    "get_gateway_llm_config",
    "HIVEMEMORY_ENV_PREFIX",
    "LEGACY_ENV_ALIASES",
    "environment_value",
    "get_config_file_path",
    "get_default_config_file_path",
    "yaml_config_settings_source",
    "legacy_env_alias_settings_source",
    "provider_credentials_settings_source",
]
