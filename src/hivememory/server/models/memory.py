"""Memory 相关的 Response 模型"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from hivememory.core.models import MemoryAccessPolicy, MemoryAtom, MemoryVisibility

_ALLOWED_MEMORY_TYPES = {
    "CODE_SNIPPET",
    "FACT",
    "URL_RESOURCE",
    "REFLECTION",
    "USER_PROFILE",
    "WORK_IN_PROGRESS",
}


def _reject_team_policy(policy: MemoryAccessPolicy | None) -> MemoryAccessPolicy | None:
    """管理接口只开放 PUBLIC 与 PRIVATE。

    生产中没有 Team 归属的来源（执行者的 ``team_id`` 从不赋值），TEAM 记忆因此
    对任何 Agent 都不可见；在入口拒绝，避免用户把记忆设成谁都读不到的状态。
    """
    if policy is not None and policy.visibility == MemoryVisibility.TEAM:
        raise ValueError("管理接口不接受 TEAM 可见性")
    return policy


class MemoryResponse(BaseModel):
    """MemoryAtom 的可序列化子集"""

    id: str
    title: str
    summary: str
    memory_type: str
    tags: list[str]
    alias: str | None = None
    content: str
    created_at: datetime
    updated_at: datetime
    confidence_score: float
    vitality_score: float
    user_id: str
    access_count: int = 0
    access_policy: MemoryAccessPolicy

    @classmethod
    def from_atom(cls, atom: MemoryAtom) -> "MemoryResponse":
        return cls(
            id=str(atom.id),
            title=atom.index.title,
            summary=atom.index.summary,
            memory_type=atom.index.memory_type.value,
            tags=atom.index.tags,
            alias=atom.index.alias,
            content=atom.payload.content,
            created_at=atom.meta.created_at,
            updated_at=atom.meta.updated_at,
            confidence_score=atom.meta.lifecycle.confidence_score,
            vitality_score=atom.meta.lifecycle.vitality_score,
            user_id=atom.workspace_identity.owner_user_id,
            access_count=atom.meta.lifecycle.access_count,
            access_policy=atom.meta.access_policy,
        )


class MemoryUpdateRequest(BaseModel):
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    alias: str | None = None
    tags: list[str] | None = None
    agent_config: dict[str, Any] | None = None
    access_policy: MemoryAccessPolicy | None = Field(
        default=None, description="新的 Agent 读取策略；省略表示不修改"
    )

    @field_validator("access_policy")
    @classmethod
    def validate_access_policy(cls, v: MemoryAccessPolicy | None) -> MemoryAccessPolicy | None:
        return _reject_team_policy(v)


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
    memories: list[MemoryResponse]
    total: int
    next_cursor: str | None = None


class MemoryCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(default="", max_length=500)
    content: str = Field(..., min_length=1)
    memory_type: str
    tags: list[str] = []
    alias: str | None = Field(default=None, max_length=60)
    access_policy: MemoryAccessPolicy | None = Field(
        default=None, description="Agent 读取策略；省略时为 PUBLIC"
    )

    @field_validator("access_policy")
    @classmethod
    def validate_access_policy(cls, v: MemoryAccessPolicy | None) -> MemoryAccessPolicy | None:
        return _reject_team_policy(v)

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        if v not in _ALLOWED_MEMORY_TYPES:
            raise ValueError(f"memory_type 必须为以下之一: {sorted(_ALLOWED_MEMORY_TYPES)}")
        return v
