"""Memory 相关的 Response 模型"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hivememory.core.models import MemoryAtom


class MemoryResponse(BaseModel):
    """MemoryAtom 的可序列化子集"""
    id: str
    title: str
    summary: str
    memory_type: str
    tags: List[str]
    alias: Optional[str] = None
    content: str
    created_at: datetime
    updated_at: datetime
    confidence_score: float
    vitality_score: float
    user_id: str
    access_count: int = 0

    @classmethod
    def from_atom(cls, atom: MemoryAtom) -> "MemoryResponse":
        return cls(
            id=str(atom.id),
            title=atom.index.title,
            summary=atom.index.summary,
            memory_type=atom.index.memory_type.value if hasattr(atom.index.memory_type, 'value') else str(atom.index.memory_type),
            tags=atom.index.tags,
            alias=atom.index.alias,
            content=atom.payload.content,
            created_at=atom.meta.created_at,
            updated_at=atom.meta.updated_at,
            confidence_score=atom.meta.confidence_score,
            vitality_score=atom.meta.vitality_score,
            user_id=atom.meta.user_id,
            access_count=atom.meta.access_count,
        )


class MemoryUpdateRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    alias: Optional[str] = None
    tags: Optional[List[str]] = None
    agent_config: Optional[Dict[str, Any]] = None


class MemoryFeedbackRequest(BaseModel):
    positive: bool
    source: str = "ui.memory_ref"


class MemoryFeedbackResponse(BaseModel):
    success: bool
    id: str
    positive: bool
    previous_vitality: float
    new_vitality: float
    previous_confidence: float
    new_confidence: float
    event_type: str


class MemoryListResponse(BaseModel):
    memories: List[MemoryResponse]
    total: int
