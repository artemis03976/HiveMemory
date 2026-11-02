from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderCredentials(BaseModel):
    """单个模型提供商（provider）的凭证。

    来自环境变量 HIVEMEMORY__PROVIDERS__<NAME>__API_KEY / __API_BASE，或经 Provider
    设置保存在 providers.secrets.yaml，由 ProviderRegistry 管理，不属于主配置。
    ModelRegistry 在解析模型时，按模型的 provider 字段查注册表补齐凭证。
    """

    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)

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
    embedding: EmbeddingGlobalConfig = Field(default_factory=EmbeddingGlobalConfig)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_moved_llm(cls, data: Any) -> Any:
        # 两段组件模型配置各属一个子系统；指出新位置，比通用的未知键错误更好迁移。
        if isinstance(data, dict) and "llm" in data:
            raise ValueError(
                "shared.llm 已移至各自子系统的配置：shared.llm.gateway 改为 gateway.llm，"
                "shared.llm.librarian 改为 patchouli.llm"
            )
        return data
