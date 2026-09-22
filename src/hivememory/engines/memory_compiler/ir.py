"""Memory IR — Phase 2A 最小单元中间表示。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryEnvelopeTarget,
)


class MemoryIdentityIR(BaseModel):
    source_kind: Literal["atom", "pending", "resolve_result"]
    alias: str | None = None
    redirected_from: str | None = None
    memory_id: str | None = None


class MemoryContentIR(BaseModel):
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    instruction: str | None = None
    tags: list[str] = []
    memory_type: str | None = None


class MemoryStatusIR(BaseModel):
    source_state: str | None = None
    source_verb: Literal["WRITE", "UPDATE"] | None = None
    is_terminal: bool = False
    is_redirect: bool = False
    is_discarded: bool = False
    message: str | None = None
    reason: str | None = None
    error: str | None = None


class MemoryUnitIR(BaseModel):
    identity: MemoryIdentityIR
    content: MemoryContentIR
    status: MemoryStatusIR
    metadata: dict[str, Any] = {}


class MemorySectionIR(BaseModel):
    kind: str
    # Phase A: 结构化单元，由 envelope 层按策略编译。
    # retrieval 场景下，MemoryUnitIR.metadata 应注入检索元数据（score/rank）。
    units: list[MemoryUnitIR] = Field(default_factory=list)
    # 向后兼容：已编译的 artifact 列表；优先使用 units。
    artifacts: list[CompiledMemoryArtifact] = Field(default_factory=list)
    empty_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryBundleIR(BaseModel):
    purpose: MemoryEnvelopeTarget
    sections: list[MemorySectionIR] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
