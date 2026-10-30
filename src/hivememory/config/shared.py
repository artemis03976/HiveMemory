from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


class LLMConfig(BaseModel):
    provider: str = "litellm"
    # model_id：引用注册表 ID（如 "deepseek-chat"）。
    # 设置时，由 ModelRegistry.resolve_for_llm_config() 填入 model/api_key/api_base；
    # 未设置时回落到直接使用 model 字段（向后兼容旧配置）。
    model_id: str | None = Field(default=None)
    model: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)
    temperature: float = Field(default=DEFAULT_TEMPERATURE)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)
    context_window: int | None = Field(default=None, gt=0, description="模型总上下文窗口 token 数")
    top_p: float = Field(default=DEFAULT_TOP_P)

    model_config = ConfigDict(extra="forbid")


class ProviderCredentials(BaseModel):
    """单个模型提供商（provider）的凭证。

    api_key / api_base 通常留空于 config.yaml（该文件被 git 跟踪），
    由环境变量 HIVEMEMORY__PROVIDERS__<NAME>__API_KEY / __API_BASE 注入。
    ModelRegistry 在解析模型时，按模型的 provider 字段查此表补齐凭证。
    """

    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")


class LLMGlobalConfig(BaseModel):
    # model_id 引用注册表；temperature/max_tokens 是组件级调参，覆盖注册表模型默认值。
    # worker 已完全由 agent profile + ModelRegistry 在运行时决定，不再需要静态配置段。
    librarian: LLMConfig = Field(
        default_factory=lambda: LLMConfig(model_id="default", temperature=0.3, max_tokens=8192)
    )
    gateway: LLMConfig = Field(
        default_factory=lambda: LLMConfig(model_id="default", temperature=0.0, max_tokens=512)
    )

    model_config = ConfigDict(extra="forbid")


# 推理模型的驻留方式：resident 加载后一直驻留到进程结束；idle_unload 空闲超过时限即卸载，
# 下一次使用时重新加载。
ModelResidency = Literal["resident", "idle_unload"]


class EmbeddingConfig(BaseModel):
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    cache_dir: str | None = Field(default=None)
    residency: ModelResidency = Field(default="idle_unload", description="嵌入模型的驻留方式")
    idle_timeout_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="idle_unload 模式下，嵌入模型空闲多少秒后卸载",
    )

    model_config = ConfigDict(extra="forbid")


class EmbeddingGlobalConfig(BaseModel):
    default: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    model_config = ConfigDict(extra="forbid")


class SharedConfig(BaseModel):
    llm: LLMGlobalConfig = Field(default_factory=LLMGlobalConfig)
    embedding: EmbeddingGlobalConfig = Field(default_factory=EmbeddingGlobalConfig)
    # 提供商凭证表：key 为 provider 名（如 "deepseek"、"openai"），
    # 供 ModelRegistry 按模型的 provider 字段解析 api_key / api_base。
    providers: dict[str, ProviderCredentials] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
