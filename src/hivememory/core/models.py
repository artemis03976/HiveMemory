"""
HiveMemory 核心数据模型

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

from pydantic import BaseModel, Field, field_validator

from hivememory.utils import TimeFormatter, Language


# ============ 枚举类型定义 ============

class MemoryType(str, Enum):
    """记忆类型枚举 - 用于区分记忆的应用场景"""
    CODE_SNIPPET = "CODE_SNIPPET"  # 代码片段、函数实现
    FACT = "FACT"  # 事实、业务规则、参数定义
    URL_RESOURCE = "URL_RESOURCE"  # 外部文档快照
    REFLECTION = "REFLECTION"  # 经验总结、错误反思
    USER_PROFILE = "USER_PROFILE"  # 用户偏好、习惯
    WORK_IN_PROGRESS = "WORK_IN_PROGRESS"  # 未完成的任务状态


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


class FlushReason(str, Enum):
    """缓冲区刷新原因枚举"""
    SEMANTIC_DRIFT = "semantic_drift"  # 语义漂移（话题切换）
    TOKEN_OVERFLOW = "token_overflow"  # Token 溢出
    IDLE_TIMEOUT = "idle_timeout"  # 空闲超时
    MANUAL = "manual"  # 手动触发
    SHORT_TEXT_ADSORB = "short_text_adsorb"  # 短文本强吸附
    MESSAGE_COUNT = "message_count"  # 消息数量达到阈值（兼容旧版本）


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
    session_id: Optional[str] = Field(default=None, description="原始会话ID")

    visibility: MemoryVisibility = Field(
        default=MemoryVisibility.PUBLIC,
        description="可见性级别"
    )
    version: int = Field(default=1, description="版本号,用于乐观锁")

    # 生命周期管理
    access_count: int = Field(default=0, description="被引用次数")
    vitality_score: float = Field(default=1.0, description="生命力分数 (0-1), 用于GC")

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

    class Config:
        json_schema_extra = {
            "example": {
                "source_agent_id": "coder_agent_01",
                "user_id": "user_123",
                "confidence_score": 0.9,
                "verification_status": "VERIFIED"
            }
        }


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

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """验证标签格式并去重"""
        # 去重并转小写
        unique_tags = list(set(tag.lower().strip() for tag in v if tag.strip()))
        # 限制最多5个标签
        return unique_tags[:5]

    def get_embedding_text(self) -> str:
        """
        构建用于Embedding的文本
        格式: Title: {title}\nType: {type}\nTags: {tags}\nSummary: {summary}
        """
        return (
            f"Title: {self.title}\n"
            f"Type: {self.memory_type.value}\n"
            f"Tags: {', '.join(self.tags)}\n"
            f"Summary: {self.summary}"
        )

    def get_sparse_context(self) -> str:
        """
        构建用于稀疏向量生成的上下文

        格式: "{title} {title} {tags_string} {tags_string} {summary}"

        Title 和 tags 重复出现以增加其在稀疏向量中的权重。
        这用于 BGE-M3 的稀疏向量生成，捕获精准实体匹配。

        Returns:
            str: 稀疏向量上下文

        Examples:
            >>> index = IndexLayer(
            ...     title="Python parse_date 函数",
            ...     summary="基于 datetime 库的日期解析工具",
            ...     tags=["python", "datetime", "utils"],
            ...     memory_type=MemoryType.CODE_SNIPPET
            ... )
            >>> index.get_sparse_context()
            "Python parse_date 函数 Python parse_date 函数 python datetime utils python datetime utils 基于 datetime 库的日期解析工具"
        """
        tags_string = " ".join(self.tags)
        return (
            f"{self.title} {self.title} "
            f"{tags_string} {tags_string} "
            f"{self.summary}"
        )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Python utils: parse_date 函数实现",
                "summary": "基于 datetime 库实现的日期解析工具，支持 ISO8601 及多种自定义格式。",
                "tags": ["python", "datetime", "utils", "code-implementation"],
                "memory_type": "CODE_SNIPPET"
            }
        }


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

    class Config:
        json_schema_extra = {
            "example": {
                "raw_source_url": "https://docs.python.org/3/library/datetime.html",
                "file_path": "/project/utils/date_helper.py",
                "context_ref": [
                    {"session_id": "sess_01", "msg_id": "msg_05"}
                ]
            }
        }


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

    class Config:
        json_schema_extra = {
            "example": {
                "content": "```python\ndef parse_date(date_str):\n    ...\n```\n\n**使用注意**：处理UTC时间时需确保...",
                "history_summary": [
                    "2025-01-01: 初始实现",
                    "2025-01-10: 添加时区支持"
                ]
            }
        }


# ============ Layer 4: Relations (关系层 - 预留) ============

class RelationLayer(BaseModel):
    """
    关系层 - 用于知识图谱关联 (阶段3+实现)
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

    def render_for_context(self, show_artifacts: bool = False, language: Language = Language.ENGLISH) -> str:
        """
        渲染为适合注入 LLM Context 的 Markdown 格式

        Args:
            show_artifacts: 是否显示原始数据信息
            language: 时间格式化语言 (默认英文)

        Returns:
            格式化的 Markdown 文本
        """
        formatter = TimeFormatter(language=language)
        # 基础信息块
        lines = [
            f"**[{self.index.title}]**",
            f"*Type*: `{self.index.memory_type.value}`",
            f"*Tags*: {', '.join(f'#{tag}' for tag in self.index.tags)}",
            f"*Updated*: {formatter.format(self.meta.updated_at)}",
            f"*Confidence*: {self._format_confidence()}",
            "",
            "---",
            "",
            self.payload.content
        ]

        # 版本历史 (如果存在)
        if self.payload.history_summary:
            lines.extend([
                "",
                "**Change Log:**",
                *[f"- {item}" for item in self.payload.history_summary]
            ])

        # 原始数据引用 (可选)
        if show_artifacts and self.payload.artifacts.raw_source_url:
            lines.extend([
                "",
                f"*Source*: {self.payload.artifacts.raw_source_url}"
            ])

        return "\n".join(lines)

    def _format_confidence(self) -> str:
        """格式化置信度"""
        score = self.meta.confidence_score
        if score >= 0.9:
            return f"✓ {score:.1%} (High)"
        elif score >= 0.7:
            return f"~ {score:.1%} (Medium)"
        else:
            return f"? {score:.1%} (Low - Verify)"

    class Config:
        json_schema_extra = {
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
