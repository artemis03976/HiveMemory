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
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


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
    """记忆可见性枚举 - 用于权限控制"""
    PRIVATE = "PRIVATE"  # 仅创建者Agent可见
    WORKSPACE = "WORKSPACE"  # 工作组共享
    PUBLIC = "PUBLIC"  # 全局可见


class VerificationStatus(str, Enum):
    """验证状态枚举"""
    VERIFIED = "VERIFIED"  # 已验证(如运行成功的代码)
    UNVERIFIED = "UNVERIFIED"  # 未验证(LLM推理)
    DEPRECATED = "DEPRECATED"  # 已过时
    HALLUCINATION = "HALLUCINATION"  # 确认为幻觉


# ============ Layer 1: Meta (元数据层) ============

class MetaData(BaseModel):
    """
    元数据 - 记忆的生命周期与权限管理
    """
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    last_accessed_at: Optional[datetime] = Field(default=None, description="最后访问时间")

    source_agent_id: str = Field(..., description="来源Agent ID")
    user_id: str = Field(..., description="归属用户ID")
    team_id: Optional[str] = Field(default=None, description="团队 ID（用于 Workspace 作用域过滤）")
    session_id: Optional[str] = Field(default=None, description="原始会话ID")

    visibility: MemoryVisibility = Field(
        default=MemoryVisibility.PUBLIC,
        description="可见性级别"
    )
    version: int = Field(default=1, description="版本号,用于乐观锁")

    # 生命周期管理
    access_count: int = Field(default=0, description="被引用次数")
    vitality_score: float = Field(default=100.0, ge=0.0, le=100.0, description="生命力分数 (0-100)")

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
        json_schema_extra={
            "example": {
                "source_agent_id": "coder_agent_01",
                "user_id": "user_123",
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
    raw_source_url: Optional[str] = Field(default=None, description="原始URL")
    file_path: Optional[str] = Field(default=None, description="文件路径")

    context_ref: List[Dict[str, str]] = Field(
        default_factory=list,
        description="溯源链: [{session_id, msg_id}, ...]"
    )

    full_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="完整版本历史 (Git-like)"
    )

    # 人偶图纸配置 (多智能体系统)
    agent_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="人偶图纸配置: {model_name, temperature, permissions: {allowed_mtp_verbs, allowed_sys_tools}}"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "raw_source_url": "https://docs.python.org/3/library/datetime.html",
                "file_path": "/project/utils/date_helper.py",
                "context_ref": [
                    {"session_id": "sess_01", "msg_id": "msg_05"}
                ]
            }
        }
    )


class PayloadLayer(BaseModel):
    """
    负载层 - 实际注入 Context 的内容
    经过Librarian清洗重写的结构化内容
    """
    content: str = Field(..., description="Markdown格式的核心内容")

    history_summary: List[str] = Field(
        default_factory=list,
        description="简化的版本历史 (用于Context注入)"
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
    id: UUID = Field(default_factory=uuid4, description="全局唯一标识符")

    meta: MetaData
    index: IndexLayer
    payload: PayloadLayer
    relations: RelationLayer = Field(default_factory=RelationLayer)

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
        转换为 Qdrant Payload 格式 (不含embedding向量)
        """
        return {
            "id": str(self.id),
            "meta": self.meta.model_dump(),
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
                    "user_id": "user_123",
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
