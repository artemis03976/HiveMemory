"""
HiveMemory 核心数据模型 - 记忆领域

基于 PROJECT.md 3.1 节的"记忆原子模型"设计
采用冰山存储架构:
- Layer 1 (Index): 向量化检索层
- Layer 2 (Payload): 内容负载层
- Layer 3 (Artifacts): 原始数据层
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models.artifact import ArtifactRef, MemoryEventLog
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.core.models.workspace import (
    IdentityScope,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.utils.time import require_utc, utc_now


class MemoryType(str, Enum):
    """记忆类型枚举 - 用于区分记忆的应用场景"""

    CODE_SNIPPET = "CODE_SNIPPET"  # 代码片段、函数实现
    FACT = "FACT"  # 事实、业务规则、参数定义
    URL_RESOURCE = "URL_RESOURCE"  # 外部文档快照
    REFLECTION = "REFLECTION"  # 经验总结、错误反思
    USER_PROFILE = "USER_PROFILE"  # 用户偏好、习惯
    WORK_IN_PROGRESS = "WORK_IN_PROGRESS"  # 未完成的任务状态
    AGENT_PROFILE = "AGENT_PROFILE"  # 人偶图纸 (多智能体系统)


class MemoryVisibility(str, Enum):
    """所属 Workspace 内的执行者读取策略。"""

    PUBLIC = "PUBLIC"  # Workspace 内所有已获准进入的执行者可读
    PRIVATE = "PRIVATE"  # 仅策略指定的 Agent 可读
    TEAM = "TEAM"  # 仅策略指定的 Team 可读


class MemoryAccessPolicy(BaseModel):
    """Memory v2 的 Workspace 内读取策略。"""

    visibility: MemoryVisibility
    target_agent_id: str | None = None
    target_team_id: str | None = None

    @field_validator("target_agent_id", "target_team_id")
    @classmethod
    def _normalize_target(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Memory read policy target 不能为空")
        if normalized == SYSTEM_AGENT_ID:
            # 保留 system 只表示"没有具体 Agent 作为操作来源主体"，
            # 不是可授权的执行主体，不得成为可见性 target。
            raise ValueError("Memory read policy target 不得使用保留 system actor")
        return normalized

    @model_validator(mode="after")
    def _validate_target(self) -> "MemoryAccessPolicy":
        if self.visibility == MemoryVisibility.PUBLIC:
            if self.target_agent_id is not None or self.target_team_id is not None:
                raise ValueError("PUBLIC policy 不得携带 target")
        elif self.visibility == MemoryVisibility.PRIVATE:
            if self.target_agent_id is None or self.target_team_id is not None:
                raise ValueError("PRIVATE policy 必须且只能携带 target_agent_id")
        elif self.visibility == MemoryVisibility.TEAM:
            if self.target_agent_id is not None or self.target_team_id is None:
                raise ValueError("TEAM policy 必须且只能携带 target_team_id")
        return self

    @classmethod
    def public(cls) -> "MemoryAccessPolicy":
        """显式构造 Workspace-local PUBLIC 策略。"""
        return cls(visibility=MemoryVisibility.PUBLIC)


class WorkspaceMemoryKey(BaseModel):
    """已授权内部路径使用的 Memory 复合资源键。"""

    workspace_identity: WorkspaceIdentity
    memory_id: UUID

    @classmethod
    def from_identity_scope(
        cls,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> "WorkspaceMemoryKey":
        """从完整访问作用域创建 Memory 复合键；缺失/错误类型作用域在边界内拒绝。"""
        identity_scope = require_identity_scope(identity_scope)
        return cls(
            workspace_identity=identity_scope.workspace_identity,
            memory_id=memory_id,
        )

    model_config = ConfigDict(frozen=True)


class VerificationStatus(str, Enum):
    """验证状态枚举"""

    VERIFIED = "VERIFIED"  # 已验证(如运行成功的代码)
    UNVERIFIED = "UNVERIFIED"  # 未验证(LLM推理)
    DEPRECATED = "DEPRECATED"  # 已过时
    HALLUCINATION = "HALLUCINATION"  # 确认为幻觉


class MemoryLifecycleState(BaseModel):
    """Memory 生命周期动态状态（schema 2.1 的 ``meta.lifecycle`` 聚合）。

    只收纳可在资源存续期间独立变化的动态字段；内容事实（``meta.version``、
    ``created_at``、``updated_at``）与归属/策略不在其中。全部字段仅允许经
    受限 ``patch_payload()`` 白名单路径更新，任何一次 patch 按字段提交完整
    替换值。

    ``decay_anchor_at`` 是遗忘衰减的唯一时间基准：创建时等于 ``created_at``，
    仅 CITATION 与内容修订推进；HIT、普通反馈和评分刷新不推进，也不得用
    ``last_accessed_at`` 替代（两者语义不可互换）。
    """

    access_count: int = Field(default=0, ge=0, description="被引用次数")
    last_accessed_at: datetime | None = Field(
        default=None, description="最近一次成功访问/引用事件时间"
    )
    event_vitality_boost: float = Field(
        default=0.0, ge=-100.0, le=100.0, description="事件累积加成 (B 项)"
    )
    vitality_score: float = Field(
        default=100.0, ge=0.0, le=100.0, description="最近一次计算并保存的生命力分数"
    )
    confidence_score: float = Field(default=0.6, ge=0.0, le=1.0, description="置信度分数")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED, description="验证状态"
    )
    decay_anchor_at: datetime = Field(description="遗忘衰减的计算基准（创建时等于创建时间）")

    @field_validator("last_accessed_at", "decay_anchor_at")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        """生命周期时间必须是 timezone-aware 并规范化为 UTC；可空字段跳过。"""
        if value is None:
            return None
        return require_utc(value)


# ============ Layer 1: Meta (元数据层) ============


class MetaData(BaseModel):
    """
    元数据 - Memory schema 2.1 的唯一归属、来源、读取策略与生命周期信息。

    四个概念各自独立承载，不得互相替代：
    - 资产归属：``workspace_identity``，单一权威；
    - 来源记录：``provenance``（MemoryProvenance），只记录 provenance 事实，
      不参与授权；
    - 读取策略：``access_policy``，可见性的唯一依据；
    - 生命周期动态状态：``lifecycle``（MemoryLifecycleState），由受限状态
      更新路径维护。

    ``created_at`` / ``updated_at`` 是内容事实：前者仅在创建时设置，后者仅
    由实际内容修订推进；访问、反馈、策略修改、评分和迁移都不更新它们。
    """

    created_at: datetime = Field(default_factory=utc_now, description="创建时间")
    updated_at: datetime = Field(default_factory=utc_now, description="最后更新时间")

    workspace_identity: WorkspaceIdentity = Field(description="Memory 的唯一持久化归属")

    provenance: MemoryProvenance = Field(description="操作来源与内容贡献者记录（非授权字段）")

    access_policy: MemoryAccessPolicy = Field(description="所属 Workspace 内的执行者读取策略")

    version: int = Field(
        default=1, ge=1, description="内容修订序号，同一资源每次实际内容提交推进一次"
    )

    lifecycle: MemoryLifecycleState = Field(description="生命周期动态状态（受限 patch 维护）")

    @field_validator("created_at", "updated_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """内容时间是持久化业务时间，必须 timezone-aware 并规范化为 UTC。"""
        return require_utc(value)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "workspace_identity": {
                    "owner_user_id": "user_123",
                    "workspace_key": "main_workspace",
                    "workspace_id": "main_workspace",
                },
                "provenance": {
                    "source_agent_id": "coder_agent_01",
                    "source_team_id": "team_core",
                    "contributing_agent_ids": ["coder_agent_01"],
                },
                "access_policy": {"visibility": "PUBLIC"},
                "lifecycle": {
                    "confidence_score": 0.9,
                    "verification_status": "UNVERIFIED",
                    "decay_anchor_at": "2026-09-22T12:00:00Z",
                },
            }
        },
    )


# ============ Layer 2: Index (索引层 - 用于向量化) ============


class IndexLayer(BaseModel):
    """
    索引层 - 仅此层参与 Embedding 向量化
    高度浓缩的语义信息,优化检索准确性
    """

    title: str = Field(..., min_length=1, max_length=200, description="简洁的标题")
    summary: str = Field(..., min_length=10, max_length=500, description="一句话摘要")
    tags: list[str] = Field(default_factory=list, description="动态语义标签")
    memory_type: MemoryType = Field(..., description="记忆类型")
    alias: str | None = Field(
        default=None, max_length=60, description="语义化别名 (snake_case, e.g. code_quicksort_impl)"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """验证标签格式并去重"""
        # 去重并转小写
        unique_tags = list(set(tag.lower().strip() for tag in v if tag.strip()))
        return unique_tags

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Python utils: parse_date 函数实现",
                "summary": "基于 datetime 库实现的日期解析工具，支持 ISO8601 与多种自定义格式。",
                "tags": ["python", "datetime", "utils", "code-implementation"],
                "memory_type": "CODE_SNIPPET",
                "alias": "code_parse_date",
            }
        }
    )


# ============ Layer 3: Payload (负载层 - 注入Context) ============


class Artifacts(BaseModel):
    """
    Artifacts - 原始数据与溯源信息
    通常不加载到 Context, 仅按需查询

    只保留 append-only 的 Artifact 引用、生命周期事件与冷存储/复活定位；
    可版本化的 Agent Profile 内容在 ``PayloadLayer.agent_config``。
    """

    # ---- v0.5.0 正式溯源层 ----
    refs: list[ArtifactRef] = Field(
        default_factory=list, description="ArtifactRef 列表 - 指向本记忆关联的所有 Artifact"
    )
    events: list[MemoryEventLog] = Field(
        default_factory=list, description="MemoryEventLog 列表 - 记忆生命周期事件流水"
    )
    cold_archive_uri: str | None = Field(
        default=None, description="归档物理存储地址（文件路径或对象存储 URI）"
    )
    cold_archive_hash: str | None = Field(
        default=None, description="归档内容 sha256，用于完整性校验"
    )
    revival_keys: list[str] = Field(default_factory=list, description="L3 复活密钥列表")

    model_config = ConfigDict(extra="ignore")


class PayloadLayer(BaseModel):
    """
    负载层 - 实际注入 Context 的内容
    经过Librarian清洗重写的结构化内容
    """

    content: str = Field(..., description="Markdown格式的核心内容")

    # 可版本化的 Agent Profile 内容（人偶图纸配置）。
    # 变化按内容版本处理但不触发向量重算。
    agent_config: dict[str, Any] | None = Field(
        default=None,
        description="人偶图纸配置: {model_name, temperature, permissions: {allowed_mtp_verbs, allowed_sys_tools}}",
    )

    artifacts: Artifacts = Field(default_factory=Artifacts, description="原始数据存根")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "```python\ndef parse_date(s): ...\n```\n\n**使用注意**：处理UTC时间时需确保...",
            }
        }
    )


# ============ Layer 4: Relations (关系层 - 预留) ============


class RelationLayer(BaseModel):
    """
    关系层 - 用于知识图谱关联 (未来实现)
    """

    relates_to: list[str] = Field(default_factory=list, description="相关记忆ID列表")
    supersedes: list[str] = Field(default_factory=list, description="被此记忆覆盖的旧记忆ID")
    depends_on: list[str] = Field(default_factory=list, description="依赖的记忆ID列表")


# ============ 主模型: MemoryAtom ============


class MemoryAtom(BaseModel):
    """
    记忆原子 - 系统的最小存储单元

    完整的"冰山模型":
    - meta: 管理信息
    - index: 检索优化层 (向量化)
    - payload: 内容负载层 (Context注入)
    - relations: 关系图谱 (预留)
    """

    schema_version: Literal["2.1"] = Field(
        default="2.1",
        description="Memory 领域与持久化契约版本（schema 2.1）",
    )
    id: UUID = Field(default_factory=uuid4, description="Workspace 内的记忆标识符")

    meta: MetaData
    index: IndexLayer
    payload: PayloadLayer
    relations: RelationLayer = Field(default_factory=RelationLayer)

    @property
    def workspace_identity(self) -> WorkspaceIdentity:
        """返回唯一的 Memory ownership 权威。"""
        return self.meta.workspace_identity

    def get_alias(self) -> str:
        """
        获取或生成语义化别名

        优先使用 IndexLayer 中存储的正式别名 (由 Generation Engine 在记忆创建时生成)。
        如果不存在，则基于 memory_type 和 title 生成临时别名作为 fallback。
        """
        alias = self.index.alias
        if alias:
            return alias

        type_prefix = self.index.memory_type.value.lower().split("_")[0]
        title = self.index.title or "untitled"
        alias = title.lower().replace(" ", "_").replace("-", "_")
        alias = "".join(c for c in alias if c.isalnum() or c == "_")
        alias = alias[:40]
        return f"{type_prefix}_{alias}"

    def to_qdrant_payload(self) -> dict[str, Any]:
        """
        转换为 Qdrant Payload 格式，并原子投影 Workspace 索引字段。

        平铺字段只服务存储预过滤；领域读取仍以 ``workspace_identity`` 为准。
        """
        meta_payload = self.meta.model_dump()
        workspace = self.workspace_identity
        meta_payload.update(
            {
                "owner_user_id": workspace.owner_user_id,
                "workspace_key": workspace.workspace_key,
                "workspace_id": workspace.workspace_id,
            }
        )
        return {
            "schema_version": self.schema_version,
            "id": str(self.id),
            "meta": meta_payload,
            "index": {
                **self.index.model_dump(),
            },
            "payload": self.payload.model_dump(),
            "relations": self.relations.model_dump(),
        }

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "meta": {
                    "workspace_identity": {
                        "owner_user_id": "user_123",
                        "workspace_key": "main_workspace",
                        "workspace_id": "main_workspace",
                    },
                    "provenance": {"source_agent_id": "coder_01"},
                    "access_policy": {"visibility": "PUBLIC"},
                    "lifecycle": {"decay_anchor_at": "2026-09-22T12:00:00Z"},
                },
                "index": {
                    "title": "Python date parsing utility",
                    "summary": "Robust date parser supporting multiple formats",
                    "tags": ["python", "utils", "datetime"],
                    "memory_type": "CODE_SNIPPET",
                },
                "payload": {"content": "```python\ndef parse_date(s): ...\n```"},
            }
        }
    )
