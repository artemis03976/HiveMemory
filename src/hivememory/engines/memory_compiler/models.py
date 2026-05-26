"""MemoryCompiler 核心数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MemoryCompileTarget(str, Enum):
    """编译目标 — 决定输出格式与内容选择策略。"""

    PROMPT_FULL = "prompt_full"
    PROMPT_INDEX = "prompt_index"
    MTP_READ = "mtp_read"
    MTP_ACK = "mtp_ack"
    MTP_REDIRECT_NOTICE = "mtp_redirect_notice"
    SHARED_CONTEXT = "shared_context"
    DENSE_EMBEDDING = "dense_embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    AGENT_PROFILE_MENU = "agent_profile_menu"
    RUNNABLE_TOOL = "runnable_tool"


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


class MemoryCompileOptions(BaseModel):
    """单次编译的选项参数。"""

    max_content_length: int = 500
    max_summary_length: int = 100
    stale_days: int = 90
    include_header_footer: bool = False
    requested_alias: Optional[str] = None
    canonical_alias: Optional[str] = None
    format: Optional[Literal["xml", "markdown", "plain"]] = None
