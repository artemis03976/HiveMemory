"""Memory IR — Phase 2A 最小单元中间表示。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class MemoryIdentityIR(BaseModel):
    source_kind: Literal["atom", "pending", "resolve_result"]
    alias: Optional[str] = None
    redirected_from: Optional[str] = None
    memory_id: Optional[str] = None


class MemoryContentIR(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []
    memory_type: Optional[str] = None


class MemoryStatusIR(BaseModel):
    source_state: Optional[str] = None
    resolve_state: Optional[str] = None
    settlement_state: Optional[str] = None
    source_verb: Optional[Literal["WRITE", "UPDATE"]] = None
    is_terminal: bool = False
    message: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class MemoryUnitIR(BaseModel):
    identity: MemoryIdentityIR
    content: MemoryContentIR
    status: MemoryStatusIR
    metadata: Dict[str, Any] = {}
