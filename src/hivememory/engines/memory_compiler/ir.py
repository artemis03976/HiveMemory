"""Memory IR — Phase 2A 最小单元中间表示。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryEnvelopeTarget,
)


class MemoryIdentityIR(BaseModel):
    source_kind: Literal["atom", "pending", "resolve_result"]
    alias: Optional[str] = None
    redirected_from: Optional[str] = None
    memory_id: Optional[str] = None


class MemoryContentIR(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    instruction: Optional[str] = None
    tags: List[str] = []
    memory_type: Optional[str] = None


class MemoryStatusIR(BaseModel):
    source_state: Optional[str] = None
    source_verb: Optional[Literal["WRITE", "UPDATE"]] = None
    is_terminal: bool = False
    is_redirect: bool = False
    is_discarded: bool = False
    message: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class MemoryUnitIR(BaseModel):
    identity: MemoryIdentityIR
    content: MemoryContentIR
    status: MemoryStatusIR
    metadata: Dict[str, Any] = {}


class MemorySectionIR(BaseModel):
    kind: str
    # Phase A: 结构化单元，由 envelope 层按策略编译。
    # retrieval 场景下，MemoryUnitIR.metadata 应注入检索元数据（score/rank）。
    units: List[MemoryUnitIR] = Field(default_factory=list)
    # 向后兼容：已编译的 artifact 列表；优先使用 units。
    artifacts: List[CompiledMemoryArtifact] = Field(default_factory=list)
    empty_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryBundleIR(BaseModel):
    purpose: MemoryEnvelopeTarget
    sections: List[MemorySectionIR] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
