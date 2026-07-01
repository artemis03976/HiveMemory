from typing import Optional
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

    model_config = ConfigDict(extra="ignore")
