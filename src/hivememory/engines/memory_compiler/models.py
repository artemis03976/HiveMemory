"""MemoryCompiler 核心数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class MemoryCompileTarget(str, Enum):
    """编译目标 — 决定输出格式与内容选择策略。"""

    PROMPT_FULL = "prompt_full"
    PROMPT_INDEX = "prompt_index"
    MTP_READ = "mtp_read"
    SHARED_CONTEXT = "shared_context"
    DENSE_EMBEDDING = "dense_embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    AGENT_PROFILE_MENU = "agent_profile_menu"
    RUNNABLE_TOOL = "runnable_tool"


class MemoryEnvelopeTarget(str, Enum):
    """Envelope target — decides how compiled artifacts are delivered."""

    RETRIEVAL_CONTEXT = "retrieval_context"
    MTP_READ_RESPONSE = "mtp_read_response"
    SHARED_CONTEXT_INJECTION = "shared_context_injection"


class CompiledMemoryArtifact(BaseModel):
    """compile() 调用的结构化输出。"""

    target: MemoryCompileTarget
    text: str
    source_kind: str = ""
    alias: Optional[str] = None
    memory_id: Optional[str] = None
    status: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEnvelopeSection(BaseModel):
    """A named section in a memory envelope."""

    kind: str
    artifacts: List[CompiledMemoryArtifact] = Field(default_factory=list)
    empty_text: Optional[str] = None


class CompiledMemoryEnvelope(BaseModel):
    """wrap() 调用的结构化输出。"""

    target: MemoryEnvelopeTarget
    text: str
    sections: List[MemoryEnvelopeSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FullStrategyConfig(BaseModel):
    """完整渲染策略：按字符上限截断。"""

    strategy: Literal["full"] = "full"
    max_tokens: int = 2000
    max_content_length: int = 500
    stale_days: int = 90


class CascadeStrategyConfig(BaseModel):
    """瀑布策略：Top-N 完整渲染 + 其余 Index，受 token 预算限制。"""

    strategy: Literal["cascade"] = "cascade"
    max_memory_tokens: int = 2000
    full_payload_count: int = 3
    max_content_length: int = 500
    index_max_summary_length: int = 100


class CompactStrategyConfig(BaseModel):
    """紧凑策略：仅渲染 Index 层。"""

    strategy: Literal["compact"] = "compact"
    max_memory_tokens: int = 2000
    index_max_summary_length: int = 100


RetrievalStrategyConfig = Annotated[
    Union[FullStrategyConfig, CascadeStrategyConfig, CompactStrategyConfig],
    Field(discriminator="strategy"),
]


class MemoryCompileOptions(BaseModel):
    """单次编译的选项参数。"""

    max_content_length: int = 500
    max_summary_length: int = 100
    stale_days: int = 90
    include_header_footer: bool = False
    requested_alias: Optional[str] = None
    canonical_alias: Optional[str] = None
    format: Optional[Literal["xml", "markdown", "plain"]] = None
    language: Optional[str] = None
    # Phase A: envelope 编译时的检索渲染策略
    retrieval_strategy_config: Optional[RetrievalStrategyConfig] = None
