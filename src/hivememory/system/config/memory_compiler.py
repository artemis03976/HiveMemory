from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class FullContextStrategyConfig(BaseModel):
    """Compile every retrieved memory as full prompt context."""

    type: Literal["full"] = "full"
    max_tokens: int = Field(default=2000)
    max_content_length: int = Field(default=500)
    stale_days: int = Field(default=90)

    model_config = ConfigDict(extra="ignore")


class CascadeContextStrategyConfig(BaseModel):
    """Compile top results as full context and later results as index context."""

    type: Literal["cascade"] = "cascade"
    max_memory_tokens: int = Field(default=2000)
    full_payload_count: int = Field(default=3)
    max_content_length: int = Field(default=500)
    index_max_summary_length: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


class CompactContextStrategyConfig(BaseModel):
    """Compile every retrieved memory as compact index context."""

    type: Literal["compact"] = "compact"
    max_memory_tokens: int = Field(default=2000)
    index_max_summary_length: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


RetrievalContextStrategyConfig = Annotated[
    Union[
        FullContextStrategyConfig,
        CascadeContextStrategyConfig,
        CompactContextStrategyConfig,
    ],
    Field(discriminator="type"),
]


class RetrievalContextCompileConfig(BaseModel):
    strategy: RetrievalContextStrategyConfig = Field(
        default_factory=CompactContextStrategyConfig,
    )

    model_config = ConfigDict(extra="ignore")


class MemoryCompilerConfig(BaseModel):
    retrieval_context: RetrievalContextCompileConfig = Field(
        default_factory=RetrievalContextCompileConfig
    )

    model_config = ConfigDict(extra="ignore")
