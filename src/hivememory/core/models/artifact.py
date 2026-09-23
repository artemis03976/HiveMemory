"""
Artifact 数据模型 - v0.5.0 数据持久化与溯源层

当前设计见 docs/patchouli/artifacts.md；历史实施稿见
docs/archive/plans/implementation/v0.5.0-data-durability-and-async-cold-path.md。

Memory 相关 Artifact（memory_creation / memory_version）自 schema "2" 起使用
结构化 ``MemoryProvenance`` 与完整原子 JSON 快照；旧 schema "1" 记录只读保留，
不由新写入产生。Interaction/Document Artifact 布局保持不变。
"""

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.models.identity import ActorIdentity
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.core.models.workspace import (
    IdentityScope,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.utils.time import require_utc, utc_now

if TYPE_CHECKING:
    from hivememory.core.models.memory import MemoryAtom


class ArtifactType(str, Enum):
    INTERACTION = "interaction"
    DOCUMENT = "document"
    MEMORY_CREATION = "memory_creation"
    MEMORY_VERSION = "memory_version"


# 完整 MemoryAtom canonical JSON 快照必须携带的顶层键（A2-P §5.1）。
_MEMORY_SNAPSHOT_REQUIRED_KEYS = ("schema_version", "id", "meta", "index", "payload", "relations")


def validate_memory_atom_snapshot(snapshot: Mapping[str, Any]) -> None:
    """校验完整 MemoryAtom canonical JSON 快照的结构约束。

    快照必须是携带 schema "2.1" 的完整原子 JSON 对象；缺失必需顶层键、未知
    schema 或不是对象都直接拒绝，不允许裁剪原子或精简投影伪装成完整快照。
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("Memory 快照必须是完整 MemoryAtom 的 JSON 对象")
    missing = [key for key in _MEMORY_SNAPSHOT_REQUIRED_KEYS if snapshot.get(key) is None]
    if missing:
        raise ValueError(f"Memory 快照缺少必需顶层键: {missing}")
    if snapshot["schema_version"] != "2.1":
        raise ValueError(
            f"Memory 快照内嵌 schema 必须是 '2.1'，收到: {snapshot['schema_version']!r}"
        )


def snapshot_memory_atom(memory: "MemoryAtom") -> dict[str, Any]:
    """生成完整 MemoryAtom 的 canonical JSON 快照。

    快照即捕获时点的完整原子序列化结果（A2-P §5.1）；生成后立即按结构约束
    校验，写入历史后不可变，后续状态变化不回写快照。
    """
    snapshot = memory.model_dump(mode="json")
    validate_memory_atom_snapshot(snapshot)
    return snapshot


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

    created_at: datetime = Field(default_factory=utc_now)

    summary: str = ""

    @field_validator("created_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """引用时间是持久化业务时间，必须 timezone-aware 并规范化为 UTC。"""
        return require_utc(value)

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
    created_at: datetime = Field(default_factory=utc_now)
    content_hash: str | None = None  # 由 ArtifactStore 在写入时填充

    workspace_identity: WorkspaceIdentity

    title: str = ""
    summary: str = ""

    @field_validator("created_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """Artifact 创建时间是持久化业务时间，必须 timezone-aware 并规范化为 UTC。"""
        return require_utc(value)

    model_config = ConfigDict(extra="ignore")


# ============ InteractionArtifact (Phase 2) ============


class InteractionTurnSnapshot(BaseModel):
    """单轮交互快照 - 原始 LogicalBlock.turn 的 JSON 冻结视图。

    不包含任何记忆归属信息（memory_id / alias / source_intent / capture_policy），
    仅保留交互本身的内容真相。

    执行者身份由单一 ``actor_identity`` 字段承载（W0 收敛后不再平铺
    user_id / agent_id / team_id 三元组）；历史平铺 JSON 的读取升级分支已随
    legacy 数据迁移完成而删除，缺少 ``actor_identity`` 的旧记录 fail closed，
    由迁移工具的 canonical replacement 处理。
    """

    block_id: str
    turn_id: str
    created_at: float | None = None

    actor_identity: ActorIdentity

    user_query: str = ""
    rewritten_query: str | None = None
    assistant_final_text: str = ""

    # 使用 dict 快照而非强类型对象，避免 runtime 模型变更时破坏 artifact 读取
    turn_events: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    semantic_traces: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class InteractionArtifact(BaseArtifact):
    """话题原始交互 Artifact - 不内嵌归属 memory 信息。"""

    artifact_type: Literal[ArtifactType.INTERACTION] = ArtifactType.INTERACTION

    topic_id: str
    topic_title: str = ""
    topic_summary: str = ""

    turns: list[InteractionTurnSnapshot] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=utc_now)

    @field_validator("captured_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """交互捕获时间是持久化业务时间，必须 timezone-aware 并规范化为 UTC。"""
        return require_utc(value)


# ============ DocumentArtifact ============


class DocumentLocator(BaseModel):
    """文档定位符 - 精确指向文档内的位置"""

    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    section: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    quote: str | None = None

    model_config = ConfigDict(extra="ignore")


class DocumentArtifact(BaseArtifact):
    """外部文档引用快照（point-in-time citation）- 写入后不可变。"""

    artifact_type: Literal[ArtifactType.DOCUMENT] = ArtifactType.DOCUMENT

    source_type: Literal["url", "file", "pdf", "markdown", "html", "repo", "unknown"] = "unknown"
    source_uri: str | None = None
    canonical_uri: str | None = None
    mime_type: str | None = None
    retrieved_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None

    locators: list[DocumentLocator] = Field(default_factory=list)
    snapshot_uri: str | None = None  # 原始内容快照的物理存储地址
    snapshot_hash: str | None = None  # 快照内容 sha256
    extracted_text_uri: str | None = None  # 提取后纯文本的物理存储地址

    @field_validator("retrieved_at")
    @classmethod
    def _require_utc_optional(cls, value: datetime | None) -> datetime | None:
        """取回时间是持久化业务时间；缺失保持 None，存在时必须为 UTC-aware。"""
        if value is None:
            return None
        return require_utc(value)


# ============ MemoryCreationArtifact / MemoryVersionArtifact ============


class MemoryInputRef(BaseModel):
    """记忆输入引用 - 记录生成时引用了哪些已有记忆"""

    memory_id: str
    alias: str | None = None
    title: str | None = None
    version: int | None = None
    used_as: Literal["context", "citation", "update_target"] = "context"

    model_config = ConfigDict(extra="ignore")


class MemoryCreationArtifact(BaseArtifact):
    """记忆创建 Artifact - genesis record，一旦写入不再更新。

    不保存 alias / title / tags 等可变字段，这些由 initial_version_ref 所指向的
    MemoryVersionArtifact(v1).snapshot_after 持有。

    schema "2" 起来源 provenance 复用 core 的 ``MemoryProvenance`` 结构化值：
    ``source_agent_id`` 记录操作来源（SETTLE 等没有具体 Agent 的操作使用保留
    ``SYSTEM_AGENT_ID``），``contributing_agent_ids`` 记录实际贡献内容的集合。
    """

    artifact_type: Literal[ArtifactType.MEMORY_CREATION] = ArtifactType.MEMORY_CREATION

    schema_version: Literal["2"] = "2"

    memory_id: str = ""
    source_intent: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"] = "WRITE"
    provenance: MemoryProvenance

    generation_view: dict[str, Any] = Field(default_factory=dict)  # GenerationContext.model_dump()
    source_artifacts: list[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: list[MemoryInputRef] = Field(default_factory=list)
    initial_version_ref: ArtifactRef | None = None  # 指向 MemoryVersionArtifact(v1)


class MemoryVersionArtifact(BaseArtifact):
    """记忆版本快照 - 完整状态快照链（类似 git commit）。

    v1 对应初始创建状态（update_source="CREATE"，snapshot_before=None）。
    schema "2" 起 snapshot_before/after 直接嵌入捕获时完整 MemoryAtom 的
    canonical JSON 对象（经 :func:`validate_memory_atom_snapshot` 约束），
    不再使用裁剪型快照模型；内嵌 Memory 的 schema ("2.1") 与本 Artifact 的
    schema ("2") 独立演进。

    来源 provenance 复用 core 的 ``MemoryProvenance``；版本更新保留已有来源
    事实，不引入 Agent owner 语义。
    """

    artifact_type: Literal[ArtifactType.MEMORY_VERSION] = ArtifactType.MEMORY_VERSION

    schema_version: Literal["2"] = "2"

    memory_id: str = ""
    version_number: int = Field(default=1, ge=1)
    update_source: Literal["CREATE", "UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "CREATE"
    provenance: MemoryProvenance

    snapshot_before: dict[str, Any] | None = None  # v1 时为 None
    snapshot_after: dict[str, Any]

    changelog: str | None = None
    source_artifacts: list[ArtifactRef] = Field(default_factory=list)
    source_memory_refs: list[MemoryInputRef] = Field(default_factory=list)
    changed_at: datetime = Field(default_factory=utc_now)

    @field_validator("changed_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """版本捕获时间是持久化业务时间，必须 timezone-aware 并规范化为 UTC。"""
        return require_utc(value)

    @model_validator(mode="after")
    def _validate_snapshots(self) -> "MemoryVersionArtifact":
        """快照字段必须是受约束的完整原子 JSON；v1 不允许携带 before。"""
        validate_memory_atom_snapshot(self.snapshot_after)
        if self.snapshot_before is not None:
            validate_memory_atom_snapshot(self.snapshot_before)
        if (
            self.version_number == 1
            and self.update_source == "CREATE"
            and self.snapshot_before is not None
        ):
            raise ValueError("CREATE v1 版本记录不允许携带 snapshot_before")
        return self


# ============ MemoryEventLog ============


class MemoryEventType(str, Enum):
    CREATED = "created"
    VERSIONED = "versioned"
    ARCHIVED = "archived"
    REVIVED = "revived"


class MemoryEventLog(BaseModel):
    """挂在单个 MemoryAtom 上的生命周期事件日志条目。"""

    event_type: MemoryEventType
    at: datetime = Field(default_factory=utc_now)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    note: str | None = None

    @field_validator("at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """事件时间是持久化业务时间，必须 timezone-aware 并规范化为 UTC。"""
        return require_utc(value)

    model_config = ConfigDict(extra="ignore")
