"""
Artifact 数据模型 - v0.5.0 数据持久化与溯源层

对应设计文档: V0.5.0DataDurabilityAndAsyncColdPathPlan.md Phase 1 & Phase 2
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class ArtifactType(str, Enum):
    """Artifact 类型枚举"""
    INTERACTION = "interaction"
    DOCUMENT = "document"
    MEMORY_CREATION = "memory_creation"
    MEMORY_VERSION = "memory_version"
    MEMORY_ARCHIVE = "memory_archive"


class SourceIntent(str, Enum):
    """记忆来源意图"""
    ARCHIVE = "ARCHIVE"
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    IMPORT = "IMPORT"
    MANUAL = "MANUAL"
    SYSTEM = "SYSTEM"


# ============ 轻量引用 ============

class ArtifactRef(BaseModel):
    """Artifact 轻量引用 - 存储在 MemoryAtom.payload.artifacts.refs 中"""
    artifact_id: str
    artifact_type: ArtifactType
    uri: str = Field(description="文件系统路径或远程 URI")
    sha256: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    summary: str = ""

    model_config = ConfigDict(extra="ignore")


# ============ 基础模型 ============

class BaseArtifact(BaseModel):
    """所有 Artifact 共有元数据"""
    artifact_id: str = Field(default_factory=lambda: f"art_{uuid4().hex}")
    artifact_type: ArtifactType
    schema_version: str = "1"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    owner_user_id: str = ""
    owner_agent_id: str = ""
    title: str = ""
    summary: str = ""
    content_hash: str = ""  # 由 ArtifactStore 在写入时填充

    model_config = ConfigDict(extra="ignore")


# ============ InteractionArtifact (Phase 2) ============

class InteractionTurnSnapshot(BaseModel):
    """单轮交互快照 - 原始 LogicalBlock.turn 的 JSON 冻结视图。

    不包含任何记忆归属信息（memory_id / alias / source_intent / capture_policy），
    仅保留交互本身的内容真相。
    """
    block_id: str
    turn_id: str
    created_at: Optional[float] = None

    user_id: str = ""
    agent_id: str = ""
    team_id: Optional[str] = None

    user_query: str = ""
    rewritten_query: Optional[str] = None
    assistant_final_text: str = ""

    # 使用 dict 快照而非强类型对象，避免 runtime 模型变更时破坏 artifact 读取
    turn_events: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_traces: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class InteractionArtifact(BaseArtifact):
    """话题原始交互 Artifact - 记录一个 topic 内的完整交互轨迹。

    data body 不嵌入记忆归属信息（memory_id / generation_view /
    source_intent / source_mode / capture_policy），保持原始性。
    """
    artifact_type: Literal[ArtifactType.INTERACTION] = ArtifactType.INTERACTION

    topic_id: str
    topic_title: str = ""
    topic_summary: str = ""

    turns: List[InteractionTurnSnapshot] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=datetime.now)


# ============ DocumentArtifact ============

class DocumentLocator(BaseModel):
    """文档定位符 - 精确指向文档内的位置"""
    page: Optional[int] = None
    heading_path: Optional[str] = None
    section: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    quote: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class DocumentArtifact(BaseArtifact):
    """外部文档来源 Artifact"""
    artifact_type: ArtifactType = ArtifactType.DOCUMENT
    source_type: Literal["url", "file", "pdf", "markdown", "html", "repo", "unknown"] = "unknown"
    source_uri: str = ""
    canonical_uri: Optional[str] = None
    mime_type: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    locators: List[DocumentLocator] = Field(default_factory=list)
    snapshot_ref: Optional[ArtifactRef] = None
    extracted_text_ref: Optional[ArtifactRef] = None


# ============ MemoryCreationArtifact ============

class MemoryInputRef(BaseModel):
    """记忆输入引用 - 记录生成时引用了哪些已有记忆"""
    memory_id: str
    alias: Optional[str] = None
    used_as: str = "source"  # source | context | base

    model_config = ConfigDict(extra="ignore")


class MemoryCreationArtifact(BaseArtifact):
    """记忆创建 Artifact - 记录初次生成的输入与产物"""
    artifact_type: ArtifactType = ArtifactType.MEMORY_CREATION
    memory_id: str = ""
    memory_alias: Optional[str] = None
    source_intent: SourceIntent = SourceIntent.WRITE
    generation_view: Dict[str, Any] = Field(default_factory=dict)
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: List[MemoryInputRef] = Field(default_factory=list)
    created_title: str = ""
    created_summary: str = ""
    created_tags: List[str] = Field(default_factory=list)
    created_memory_type: str = ""


# ============ MemoryVersionArtifact ============

class MemoryVersionArtifact(BaseArtifact):
    """记忆版本快照 Artifact - 记录每次 UPDATE/MERGE 的前后内容"""
    artifact_type: ArtifactType = ArtifactType.MEMORY_VERSION
    memory_id: str = ""
    memory_alias: Optional[str] = None
    version_from: int = 0
    version_to: int = 1
    update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "UPDATE"
    previous_content: str = ""
    new_content: str = ""
    changelog: str = ""
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: List[MemoryInputRef] = Field(default_factory=list)
    changed_at: datetime = Field(default_factory=datetime.now)


# ============ MemoryArchiveArtifact ============

class MemoryArchiveArtifact(BaseArtifact):
    """记忆归档 Artifact - L3 冷存储归档记录，含复活密钥"""
    artifact_type: ArtifactType = ArtifactType.MEMORY_ARCHIVE
    canonical_alias: str = ""
    alias_history: List[str] = Field(default_factory=list)
    memory_type: str = ""
    archived_at: datetime = Field(default_factory=datetime.now)
    original_vitality: float = 0.0
    storage_uri: str = ""
    compressed_size_bytes: Optional[int] = None
    revival_keys: List[str] = Field(default_factory=list)


# ============ MemoryProvenance ============

class MemoryProvenance(BaseModel):
    """记忆生命周期事件记录"""
    action: Literal[
        "created", "updated", "merged", "touched", "archived",
        "resurrected", "manual_edit", "future_split", "future_merge"
    ]
    at: datetime = Field(default_factory=datetime.now)
    source_intent: Optional[SourceIntent] = None
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    note: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
