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
from typing import Any, Dict, List, Literal, Optional, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivememory.core.errors import OwnerMismatchError
from hivememory.core.models.artifact import ArtifactRef, MemoryEventLog
from hivememory.core.models.interaction import Identity
from hivememory.core.models.workspace import WorkspaceAccessContext, WorkspaceIdentity


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
    target_agent_id: Optional[str] = None
    target_team_id: Optional[str] = None

    @field_validator("target_agent_id", "target_team_id")
    @classmethod
    def _normalize_target(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Memory read policy target 不能为空")
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


class MemoryCreationContext(BaseModel):
    """生成数据面创建 Memory 所需的最小、不可变作用域。"""

    actor_identity: Identity
    workspace_identity: WorkspaceIdentity

    @model_validator(mode="after")
    def _require_same_owner(self) -> Self:
        if self.actor_identity.user_id != self.workspace_identity.owner_user_id:
            raise OwnerMismatchError(
                details={
                    "actor_user_id": self.actor_identity.user_id,
                    "owner_user_id": self.workspace_identity.owner_user_id,
                }
            )
        return self

    @classmethod
    def from_access_context(
        cls,
        access_context: WorkspaceAccessContext,
    ) -> "MemoryCreationContext":
        """从已验证的访问上下文冻结生成所需的 owner 与来源。"""
        return cls(
            actor_identity=access_context.actor_identity,
            workspace_identity=access_context.workspace_identity,
        )

    model_config = ConfigDict(frozen=True)


MemoryReadScope = WorkspaceAccessContext | MemoryCreationContext
"""读取/检索可使用完整 AccessContext，或生成数据面的窄创建上下文。"""


def require_memory_read_scope(scope: MemoryReadScope) -> MemoryReadScope:
    """拒绝裸 Identity、WorkspaceIdentity 或缺失的 Memory 读取作用域。"""
    if not isinstance(scope, (WorkspaceAccessContext, MemoryCreationContext)):
        from hivememory.core.errors import ScopeRequiredError

        raise ScopeRequiredError()
    return scope


class WorkspaceMemoryKey(BaseModel):
    """已授权内部路径使用的 Memory 复合资源键。"""

    workspace_identity: WorkspaceIdentity
    memory_id: UUID

    @classmethod
    def from_access_context(
        cls,
        access_context: WorkspaceAccessContext,
        memory_id: UUID,
    ) -> "WorkspaceMemoryKey":
        """从完整访问上下文创建 Memory 复合键。"""
        return cls(
            workspace_identity=access_context.workspace_identity,
            memory_id=memory_id,
        )

    model_config = ConfigDict(frozen=True)


class VerificationStatus(str, Enum):
    """验证状态枚举"""
    VERIFIED = "VERIFIED"  # 已验证(如运行成功的代码)
    UNVERIFIED = "UNVERIFIED"  # 未验证(LLM推理)
    DEPRECATED = "DEPRECATED"  # 已过时
    HALLUCINATION = "HALLUCINATION"  # 确认为幻觉


# ============ Layer 1: Meta (元数据层) ============

class MetaData(BaseModel):
    """
    元数据 - Memory v2 的唯一归属、来源、读取策略与生命周期信息。
    """
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    last_accessed_at: Optional[datetime] = Field(default=None, description="最后访问时间")

    workspace_identity: WorkspaceIdentity = Field(description="Memory 的唯一持久化归属")
    source_agent_id: str = Field(..., min_length=1, description="创建来源 Agent ID")
    source_team_id: Optional[str] = Field(default=None, description="创建来源 Team ID")

    # TODO: 会话ID应由artifact保存
    session_id: Optional[str] = Field(default=None, description="原始会话ID")

    access_policy: MemoryAccessPolicy = Field(description="所属 Workspace 内的执行者读取策略")
    version: int = Field(default=1, description="版本号,用于乐观锁")

    # 生命周期管理
    access_count: int = Field(default=0, description="被引用次数")
    vitality_score: float = Field(default=100.0, ge=0.0, le=100.0, description="生命力分数 (0-100)")
    # 事件累积加成 (B 项)：HIT/CITATION/FEEDBACK 等事件的累计影响。
    # 与 vitality_score 解耦存储，由 VitalityCalculator 在重算时合并进最终分数。
    event_vitality_boost: float = Field(default=0.0, ge=-100.0, le=100.0, description="事件累积加成 (B 项)")

    # 置信度与验证
    confidence_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="置信度分数"
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="验证状态"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "source_agent_id": "coder_agent_01",
                "workspace_identity": {
                    "owner_user_id": "user_123",
                    "workspace_key": "main_workspace",
                    "workspace_id": "main_workspace",
                },
                "access_policy": {"visibility": "PUBLIC"},
                "confidence_score": 0.9,
                "verification_status": "VERIFIED"
            }
        }
    )


# ============ Layer 2: Index (索引层 - 用于向量化) ============

class IndexLayer(BaseModel):
    """
    索引层 - 仅此层参与 Embedding 向量化
    高度浓缩的语义信息,优化检索准确性
    """
    title: str = Field(..., min_length=1, max_length=200, description="简洁的标题")
    summary: str = Field(..., min_length=10, max_length=500, description="一句话摘要")
    tags: List[str] = Field(default_factory=list, description="动态语义标签")
    memory_type: MemoryType = Field(..., description="记忆类型")
    alias: Optional[str] = Field(
        default=None, max_length=60,
        description="语义化别名 (snake_case, e.g. code_quicksort_impl)"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """验证标签格式并去重"""
        # 去重并转小写
        unique_tags = list(set(tag.lower().strip() for tag in v if tag.strip()))
        return unique_tags

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Python utils: parse_date 函数实现",
                "summary": "基于 datetime 库实现的日期解析工具，支持 ISO8601 及多种自定义格式。",
                "tags": ["python", "datetime", "utils", "code-implementation"],
                "memory_type": "CODE_SNIPPET",
                "alias": "code_parse_date"
            }
        }
    )


# ============ Layer 3: Payload (负载层 - 注入Context) ============

class Artifacts(BaseModel):
    """
    Artifacts - 原始数据与溯源信息
    通常不加载到 Context, 仅按需查询
    """
    agent_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="人偶图纸配置: {model_name, temperature, permissions: {allowed_mtp_verbs, allowed_sys_tools}}"
    )

    # ---- v0.5.0 正式溯源层 ----
    refs: List[ArtifactRef] = Field(
        default_factory=list,
        description="ArtifactRef 列表 - 指向本记忆关联的所有 Artifact"
    )
    events: List[MemoryEventLog] = Field(
        default_factory=list,
        description="MemoryEventLog 列表 - 记忆生命周期事件流水"
    )
    cold_archive_uri: Optional[str] = Field(
        default=None,
        description="归档物理存储地址（文件路径或对象存储 URI）"
    )
    cold_archive_hash: Optional[str] = Field(
        default=None,
        description="归档内容 sha256，用于完整性校验"
    )
    revival_keys: List[str] = Field(
        default_factory=list,
        description="L3 复活密钥列表"
    )

    model_config = ConfigDict(extra="ignore")


class PayloadLayer(BaseModel):
    """
    负载层 - 实际注入 Context 的内容
    经过Librarian清洗重写的结构化内容
    """
    content: str = Field(..., description="Markdown格式的核心内容")

    # 兼容性字段：artifact 系统关闭时，它作为轻量历史 fallback 供检索/提示词参考。
    # TODO(history-compiler): 后续 MTP RUN 历史信息编译实现后，统一决定
    # history_summary 是继续作为 fallback 保留，还是完全迁移到 MemoryVersionArtifact。
    history_summary: List[str] = Field(
        default_factory=list,
        description="简化的版本历史；artifact 禁用时作为 fallback，展示逻辑需进入历史信息编译"
    )

    artifacts: Artifacts = Field(
        default_factory=Artifacts,
        description="原始数据存根"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "```python\ndef parse_date(date_str):\n    ...\n```\n\n**使用注意**：处理UTC时间时需确保...",
                "history_summary": [
                    "2025-01-01: 初始实现",
                    "2025-01-10: 添加时区支持"
                ]
            }
        }
    )


# ============ Layer 4: Relations (关系层 - 预留) ============

class RelationLayer(BaseModel):
    """
    关系层 - 用于知识图谱关联 (未来实现)
    """
    relates_to: List[str] = Field(default_factory=list, description="相关记忆ID列表")
    supersedes: List[str] = Field(default_factory=list, description="被此记忆覆盖的旧记忆ID")
    depends_on: List[str] = Field(default_factory=list, description="依赖的记忆ID")


# ============ 主模型: MemoryAtom ============

class MemoryAtom(BaseModel):
    """
    记忆原子 - 系统的最小存储单元

    完整的"冰山模型":
    - meta: 管理信息
    - index: 检索优化层 (向量化)
    - payload: 内容负载层 (Context注入)
    - relations: 关系图谱层 (预留)
    """
    schema_version: Literal[2] = Field(
        default=2,
        description="Memory 领域与持久化契约版本",
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
        if getattr(self.index, 'alias', None):
            return self.index.alias

        type_prefix = self.index.memory_type.value.lower().split("_")[0]
        title = self.index.title or "untitled"
        alias = title.lower().replace(" ", "_").replace("-", "_")
        alias = "".join(c for c in alias if c.isalnum() or c == "_")
        alias = alias[:40]
        return f"{type_prefix}_{alias}"

    def to_qdrant_payload(self) -> Dict[str, Any]:
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
                    "source_agent_id": "coder_01",
                    "workspace_identity": {
                        "owner_user_id": "user_123",
                        "workspace_key": "main_workspace",
                        "workspace_id": "main_workspace",
                    },
                    "access_policy": {"visibility": "PUBLIC"},
                    "confidence_score": 0.9
                },
                "index": {
                    "title": "Python date parsing utility",
                    "summary": "Robust date parser supporting multiple formats",
                    "tags": ["python", "utils", "datetime"],
                    "memory_type": "CODE_SNIPPET"
                },
                "payload": {
                    "content": "```python\ndef parse_date(s): ...\n```"
                }
            }
        }
    )
