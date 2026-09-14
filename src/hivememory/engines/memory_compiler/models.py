"""MemoryCompiler 核心数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from hivememory.system.config.memory_compiler import RetrievalContextStrategyConfig


class MemoryCompileTarget(str, Enum):
    """单元（Unit）编译 target。"""

    PROMPT_FULL = "prompt_full"
    PROMPT_INDEX = "prompt_index"
    MTP_READ = "mtp_read"
    SHARED_CONTEXT = "shared_context"
    DENSE_EMBEDDING = "dense_embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    AGENT_PROFILE_MENU = "agent_profile_menu"
    RUNNABLE_TOOL = "runnable_tool"


class MemoryEnvelopeTarget(str, Enum):
    """信封（Envelope）编译 target。"""

    RETRIEVAL_CONTEXT = "retrieval_context"
    MTP_READ_RESPONSE = "mtp_read_response"
    SHARED_CONTEXT_INJECTION = "shared_context_injection"


class CompiledMemory(BaseModel):
    """compile() 对单元产物与 Envelope 文本的统一输出。"""

    target: MemoryCompileTarget | MemoryEnvelopeTarget
    text: str
    sections: List["MemoryEnvelopeSection"] = Field(default_factory=list)
    source_kind: str = ""
    alias: Optional[str] = None
    memory_id: Optional[str] = None
    status: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEnvelopeSection(BaseModel):
    """编译后记忆 Envelope 中的命名小节。"""

    kind: str
    artifacts: List[CompiledMemory] = Field(default_factory=list)
    empty_text: Optional[str] = None


# 兼容类型别名，指向统一的运行时模型。
CompiledMemoryArtifact = CompiledMemory
CompiledMemoryEnvelope = CompiledMemory


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
    retrieval_strategy_config: Optional[RetrievalContextStrategyConfig] = None
