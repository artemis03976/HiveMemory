from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class FullContextStrategyConfig(BaseModel):
    """把每条检索到的记忆编译为完整提示词上下文。"""

    type: Literal["full"] = "full"
    max_tokens: int = Field(default=2000)
    max_content_length: int = Field(default=500)
    stale_days: int = Field(default=90)

    model_config = ConfigDict(extra="ignore")


class CascadeContextStrategyConfig(BaseModel):
    """把头部结果编译为完整上下文，其余编译为索引上下文。"""

    type: Literal["cascade"] = "cascade"
    max_memory_tokens: int = Field(default=2000)
    full_payload_count: int = Field(default=3)
    max_content_length: int = Field(default=500)
    index_max_summary_length: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


class CompactContextStrategyConfig(BaseModel):
    """把每条检索到的记忆编译为紧凑索引上下文。"""

    type: Literal["compact"] = "compact"
    max_memory_tokens: int = Field(default=2000)
    index_max_summary_length: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


RetrievalContextStrategyConfig = Annotated[
    FullContextStrategyConfig | CascadeContextStrategyConfig | CompactContextStrategyConfig,
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
