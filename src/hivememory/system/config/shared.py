from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


class LLMConfig(BaseModel):
    provider: str = "litellm"
    model: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = Field(default=None)
    temperature: float = Field(default=DEFAULT_TEMPERATURE)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)
    top_p: float = Field(default=DEFAULT_TOP_P)

    model_config = ConfigDict(extra="ignore")


class ProviderCredentials(BaseModel):
    """单个模型提供商（provider）的凭证。

    api_key / api_base 通常留空于 config.yaml（该文件被 git 跟踪），
    由环境变量 HIVEMEMORY__PROVIDERS__<NAME>__API_KEY / __API_BASE 注入。
    ModelRegistry 在解析模型时，按模型的 provider 字段查此表补齐凭证。
    """
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = Field(default=None)

    model_config = ConfigDict(extra="ignore")


class LLMGlobalConfig(BaseModel):
    librarian: LLMConfig = Field(default_factory=lambda: LLMConfig(model="deepseek/deepseek-chat", temperature=0.3, max_tokens=8192))
    gateway: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4o", temperature=0.0, max_tokens=512))
    worker: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4o", temperature=0.7, max_tokens=4096))

    model_config = ConfigDict(extra="allow")


class EmbeddingConfig(BaseModel):
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="cpu")
    cache_dir: Optional[str] = Field(default=None)
    batch_size: int = Field(default=32)
    normalize_embeddings: bool = Field(default=True)
    dimension: int = Field(default=384)

    model_config = ConfigDict(extra="ignore")


class EmbeddingGlobalConfig(BaseModel):
    default: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    model_config = ConfigDict(extra="allow")


class SharedConfig(BaseModel):
    llm: LLMGlobalConfig = Field(default_factory=LLMGlobalConfig)
    embedding: EmbeddingGlobalConfig = Field(default_factory=EmbeddingGlobalConfig)
    # 提供商凭证表：key 为 provider 名（如 "deepseek"、"openai"），
    # 供 ModelRegistry 按模型的 provider 字段解析 api_key / api_base。
    providers: Dict[str, ProviderCredentials] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")
