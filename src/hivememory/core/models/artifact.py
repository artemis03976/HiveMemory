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
    INTERACTION = "interaction"
    DOCUMENT = "document"
    MEMORY_CREATION = "memory_creation"
    MEMORY_VERSION = "memory_version"

# ============ 轻量引用 ============

class ArtifactRef(BaseModel):
    """Artifact 轻量引用指针 - 存储在 MemoryAtom.payload.artifacts.refs 中"""
    artifact_id: str
    artifact_type: ArtifactType

    uri: str = Field(default="", description="文件系统路径或远程 URI")
    sha256: str = ""

    created_at: datetime = Field(default_factory=datetime.now)
    
    summary: str = ""

    model_config = ConfigDict(extra="ignore")


# ============ 基础模型 ============

class BaseArtifact(BaseModel):
    """所有 Artifact 共有元数据。写入后不再修改（append-only）。"""
    artifact_id: str = Field(default_factory=lambda: f"art_{uuid4().hex}")
    artifact_type: ArtifactType

    schema_version: str = "1"
    created_at: datetime = Field(default_factory=datetime.now)
    content_hash: Optional[str] = None  # 由 ArtifactStore 在写入时填充

    owner_user_id: str = ""
    owner_agent_id: str = ""

    title: str = ""
    summary: str = ""

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
    """话题原始交互 Artifact - 不内嵌归属 memory 信息。"""
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
    heading_path: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    quote: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class DocumentArtifact(BaseArtifact):
    """外部文档引用快照（point-in-time citation）- 写入后不可变。"""
    artifact_type: Literal[ArtifactType.DOCUMENT] = ArtifactType.DOCUMENT

    source_type: Literal["url", "file", "pdf", "markdown", "html", "repo", "unknown"] = "unknown"
    source_uri: Optional[str] = None
    canonical_uri: Optional[str] = None
    mime_type: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None

    locators: List[DocumentLocator] = Field(default_factory=list)
    snapshot_uri: Optional[str] = None        # 原始内容快照的物理存储地址
    snapshot_hash: Optional[str] = None       # 快照内容 sha256
    extracted_text_uri: Optional[str] = None  # 提取后纯文本的物理存储地址


# ============ MemoryCreationArtifact / MemoryVersionArtifact ============

class MemoryInputRef(BaseModel):
    """记忆输入引用 - 记录生成时引用了哪些已有记忆"""
    memory_id: str
    alias: Optional[str] = None
    title: Optional[str] = None
    version: Optional[int] = None
    used_as: Literal["context", "citation", "update_target"] = "context"

    model_config = ConfigDict(extra="ignore")


class MemoryVersionSnapshot(BaseModel):
    """记忆原子某一版本下所有可变字段的完整快照。"""
    content: str
    alias: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    memory_type: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class MemoryCreationArtifact(BaseArtifact):
    """记忆创建 Artifact - genesis provenance，一旦写入不再更新。

    不保存 alias / title / tags 等可变字段，这些由 initial_version_ref 所指向的
    MemoryVersionArtifact(v1).snapshot_after 持有。
    """
    artifact_type: Literal[ArtifactType.MEMORY_CREATION] = ArtifactType.MEMORY_CREATION

    memory_id: str = ""
    source_intent: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"] = "WRITE"

    generation_view: Dict[str, Any] = Field(default_factory=dict)  # GenerationContext.model_dump()
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: List[MemoryInputRef] = Field(default_factory=list)
    initial_version_ref: Optional[ArtifactRef] = None  # 指向 MemoryVersionArtifact(v1)


class MemoryVersionArtifact(BaseArtifact):
    """记忆版本快照 - 完整状态快照链（类似 git commit）。

    v1 对应初始创建状态（update_source="CREATE"，snapshot_before=None）。
    后续版本 snapshot_before/after 均包含全量可变字段，支持任意版本独立重建。
    """
    artifact_type: Literal[ArtifactType.MEMORY_VERSION] = ArtifactType.MEMORY_VERSION

    memory_id: str = ""
    version_number: int = 1
    update_source: Literal["CREATE", "UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "CREATE"

    snapshot_before: Optional[MemoryVersionSnapshot] = None  # v1 时为 None
    snapshot_after: MemoryVersionSnapshot

    changelog: Optional[str] = None
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: List[MemoryInputRef] = Field(default_factory=list)
    changed_at: datetime = Field(default_factory=datetime.now)


# ============ MemoryProvenance ============

class MemoryProvenance(BaseModel):
    """记忆生命周期事件记录"""
    action: Literal[
        "created", "updated", "merged", "touched", "archived",
        "resurrected", "manual_edit", "future_split", "future_merge"
    ]
    at: datetime = Field(default_factory=datetime.now)
    source_intent: Optional[Literal[
        "ARCHIVE", "WRITE", "UPDATE", "IMPORT", "MANUAL", "SYSTEM"
    ]] = None
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    vitality: Optional[float] = None  # 归档、触碰等生命周期事件时记录当前生命值
    note: Optional[str] = None

    model_config = ConfigDict(extra="ignore")