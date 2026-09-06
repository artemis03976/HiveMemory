"""
Artifact 数据模型 - v0.5.0 数据持久化与溯源层

当前设计见 docs/patchouli/artifacts.md；历史实施稿见
docs/archive/plans/implementation/v0.5.0-data-durability-and-async-cold-path.md。
"""

from collections.abc import Iterable
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models.identity import ActorIdentity
from hivememory.core.models.workspace import (
    IdentityScope,
    WorkspaceIdentity,
    require_identity_scope,
)


class ArtifactType(str, Enum):
    INTERACTION = "interaction"
    DOCUMENT = "document"
    MEMORY_CREATION = "memory_creation"
    MEMORY_VERSION = "memory_version"


def normalize_contributing_agent_ids(value: Iterable[str]) -> tuple[str, ...]:
    """归一化内容贡献者集合：去重并保持首次出现顺序。

    贡献者表达"哪些具体 Agent 的工作产出了内容"，不表达资产归属或授权
    目标。保留 ``SYSTEM_AGENT_ID`` 表示"没有具体 Agent 作为操作来源主体"，
    不是内容贡献者，与空白标识一并丢弃。
    """
    normalized: list[str] = []
    for agent_id in value:
        stripped = agent_id.strip() if isinstance(agent_id, str) else ""
        if not stripped or stripped == SYSTEM_AGENT_ID:
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized)


class WorkspaceArtifactKey(BaseModel):
    """Artifact 在存储层使用的 owner/workspace/ID 复合资源键。"""

    workspace_identity: WorkspaceIdentity
    artifact_id: str = Field(min_length=1)

    @classmethod
    def from_identity_scope(
        cls,
        identity_scope: IdentityScope,
        artifact_id: str,
    ) -> Self:
        """从完整访问作用域构造 Artifact 复合键；缺失/错误类型作用域在边界内拒绝。"""
        identity_scope = require_identity_scope(identity_scope)
        return cls(
            workspace_identity=identity_scope.workspace_identity,
            artifact_id=artifact_id,
        )

    model_config = ConfigDict(frozen=True)

# ============ 轻量引用 ============

class ArtifactRef(BaseModel):
    """Artifact 轻量引用指针 - 存储在 MemoryAtom.payload.artifacts.refs 中"""
    artifact_id: str = Field(min_length=1)
    artifact_type: ArtifactType

    workspace_identity: WorkspaceIdentity

    uri: str = Field(default="", description="文件系统路径或远程 URI")
    sha256: str = ""

    created_at: datetime = Field(default_factory=datetime.now)

    summary: str = ""

    model_config = ConfigDict(extra="ignore", frozen=True)


# ============ 基础模型 ============

class BaseArtifact(BaseModel):
    """所有 Artifact 共有元数据。写入后不再修改（append-only）。

    资产归属由 ``workspace_identity`` 单一表达：Artifact 是 Workspace 资产，
    不存在 Agent owner 语义；来源 provenance 按具体 Artifact 类型定义字段。
    """
    artifact_id: str = Field(default_factory=lambda: f"art_{uuid4().hex}", min_length=1)
    artifact_type: ArtifactType

    schema_version: str = "1"
    created_at: datetime = Field(default_factory=datetime.now)
    content_hash: Optional[str] = None  # 由 ArtifactStore 在写入时填充

    workspace_identity: WorkspaceIdentity

    title: str = ""
    summary: str = ""

    model_config = ConfigDict(extra="ignore")


# ============ InteractionArtifact (Phase 2) ============

class InteractionTurnSnapshot(BaseModel):
    """单轮交互快照 - 原始 LogicalBlock.turn 的 JSON 冻结视图。

    不包含任何记忆归属信息（memory_id / alias / source_intent / capture_policy），
    仅保留交互本身的内容真相。

    执行者身份由单一 ``actor_identity`` 字段承载（W0 收敛后不再平铺
    user_id / agent_id / team_id 三元组）；历史平铺 JSON 经
    :meth:`_upgrade_legacy_flat_actor` 在读取时升级，不做批量回写。
    """
    block_id: str
    turn_id: str
    created_at: Optional[float] = None

    actor_identity: ActorIdentity

    user_query: str = ""
    rewritten_query: Optional[str] = None
    assistant_final_text: str = ""

    # 使用 dict 快照而非强类型对象，避免 runtime 模型变更时破坏 artifact 读取
    turn_events: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_traces: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_flat_actor(cls, data: Any) -> Any:
        """旧平铺 JSON 读取兼容：user_id/agent_id/team_id 重建为 actor_identity。

        仅做读取升级，不批量回写已存储 artifact：

        - 旧数据缺少具体 Agent 时使用保留 ``SYSTEM_AGENT_ID``，表示没有
          具体 Agent 作为操作来源主体，不得回落到 ``omni_doll`` 等真实 Agent；
        - 缺少用户归属（user_id 为空）等无法安全推断的字段时 fail closed，
          交由迁移诊断处理，不猜测归属。
        """
        if not isinstance(data, dict) or "actor_identity" in data:
            return data

        legacy_user_id = (data.get("user_id") or "").strip()
        if not legacy_user_id:
            raise ValueError(
                "旧版 InteractionTurnSnapshot 缺少用户归属（user_id 为空），"
                "无法安全推断 actor_identity，已按 fail closed 拒绝读取；"
                "请通过迁移诊断流程处理该 artifact"
            )
        data["actor_identity"] = {
            "user_id": legacy_user_id,
            "agent_id": (data.get("agent_id") or "").strip() or SYSTEM_AGENT_ID,
            "team_id": data.get("team_id"),
        }
        return data

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

    @classmethod
    def from_memory_atom(cls, memory: Any) -> "MemoryVersionSnapshot":
        """Build the canonical mutable-field snapshot for a MemoryAtom."""
        memory_type = memory.index.memory_type
        return cls(
            content=memory.payload.content,
            alias=memory.index.alias,
            title=memory.index.title,
            summary=memory.index.summary,
            tags=list(memory.index.tags),
            memory_type=memory_type.value if hasattr(memory_type, "value") else memory_type,
        )


class MemoryCreationArtifact(BaseArtifact):
    """记忆创建 Artifact - genesis record，一旦写入不再更新。

    不保存 alias / title / tags 等可变字段，这些由 initial_version_ref 所指向的
    MemoryVersionArtifact(v1).snapshot_after 持有。

    来源 provenance 与 MemoryAtom 语义一致：``source_agent_id`` 记录操作来源
    （SETTLE 等没有具体 Agent 的操作使用保留 ``SYSTEM_AGENT_ID``），
    ``contributing_agent_ids`` 记录实际贡献内容的 Agent 集合。
    """
    artifact_type: Literal[ArtifactType.MEMORY_CREATION] = ArtifactType.MEMORY_CREATION

    memory_id: str = ""
    source_intent: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"] = "WRITE"
    source_agent_id: str = Field(..., min_length=1)
    contributing_agent_ids: tuple[str, ...] = Field(default_factory=tuple)

    generation_view: Dict[str, Any] = Field(default_factory=dict)  # GenerationContext.model_dump()
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: List[MemoryInputRef] = Field(default_factory=list)
    initial_version_ref: Optional[ArtifactRef] = None  # 指向 MemoryVersionArtifact(v1)

    @field_validator("contributing_agent_ids")
    @classmethod
    def _normalize_contributors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """去重并保持首次出现顺序；system 不是内容贡献者。"""
        return normalize_contributing_agent_ids(value)


class MemoryVersionArtifact(BaseArtifact):
    """记忆版本快照 - 完整状态快照链（类似 git commit）。

    v1 对应初始创建状态（update_source="CREATE"，snapshot_before=None）。
    后续版本 snapshot_before/after 均包含全量可变字段，支持任意版本独立重建。

    来源 provenance 与 MemoryAtom 语义一致（见 MemoryCreationArtifact）；
    版本更新保留已有来源字段，不引入 Agent owner 语义。
    """
    artifact_type: Literal[ArtifactType.MEMORY_VERSION] = ArtifactType.MEMORY_VERSION

    memory_id: str = ""
    version_number: int = 1
    update_source: Literal["CREATE", "UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "CREATE"
    source_agent_id: str = Field(..., min_length=1)
    contributing_agent_ids: tuple[str, ...] = Field(default_factory=tuple)

    snapshot_before: Optional[MemoryVersionSnapshot] = None  # v1 时为 None
    snapshot_after: MemoryVersionSnapshot

    changelog: Optional[str] = None
    source_artifacts: List[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: List[MemoryInputRef] = Field(default_factory=list)
    changed_at: datetime = Field(default_factory=datetime.now)

    @field_validator("contributing_agent_ids")
    @classmethod
    def _normalize_contributors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """去重并保持首次出现顺序；system 不是内容贡献者。"""
        return normalize_contributing_agent_ids(value)


# ============ MemoryEventLog ============

class MemoryEventType(str, Enum):
    CREATED = "created"
    VERSIONED = "versioned"
    ARCHIVED = "archived"
    REVIVED = "revived"


class MemoryEventLog(BaseModel):
    """Lifecycle event log entry attached to one MemoryAtom."""

    event_type: MemoryEventType
    at: datetime = Field(default_factory=datetime.now)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    note: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
