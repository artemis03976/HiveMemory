"""
HiveMemory - Retrieval 模块数据模型

定义了记忆检索模块的所有数据模型和配置类。

作者: HiveMemory Team
版本: 0.1.0
"""

from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from hivememory.core.models import MemoryAtom, MemoryType, Identity
from hivememory.utils.memory_atom_renderer import RenderFormat


# ========== 数据模型 ==========


class QueryFilters(BaseModel):
    """
    结构化过滤条件模型

    用于检索引擎的过滤条件传递。
    定义了检索时可用的所有过滤维度。

    identity 统一承载请求者身份 (user_id, agent_id, team_id)，
    用于 Visibility Scope Filtering (MutiAgentSystem.md §3.3):
      - PUBLIC: 全局可见
      - WORKSPACE: team_id 匹配时可见
      - PRIVATE: source_agent_id 匹配当前 agent_id 时可见

    Attributes:
        identity: 请求者身份标识 (替代原 user_id + source_agent_id)
        memory_type: 记忆类型过滤
        time_range: 时间范围过滤
        tags: 标签过滤
        min_confidence: 最小置信度过滤
    """
    identity: Optional[Identity] = None
    memory_type: Optional[MemoryType] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    tags: List[str] = Field(default_factory=list)
    min_confidence: float = 0.0

    # ---- 向后兼容属性 ----

    @property
    def user_id(self) -> Optional[str]:
        """兼容属性: 从 identity 中提取 user_id"""
        return self.identity.user_id if self.identity else None

    @property
    def source_agent_id(self) -> Optional[str]:
        """兼容属性: 从 identity 中提取 agent_id"""
        return self.identity.agent_id if self.identity else None

    def is_empty(self) -> bool:
        """检查过滤条件是否为空"""
        return (
            self.identity is None
            and self.memory_type is None
            and self.time_range is None
            and len(self.tags) == 0
            and self.min_confidence == 0.0
        )


class RetrievalQuery(BaseModel):
    """
    处理后的结构化查询
    
    包含:
    - 语义查询文本（用于向量检索）
    - 提取的关键词
    - 结构化过滤条件
    """
    semantic_query: str  # 用于向量检索的语义查询
    keywords: List[str] = Field(default_factory=list)  # 提取的关键词
    filters: QueryFilters = Field(default_factory=QueryFilters)  # 过滤条件
    
    def get_search_text(self) -> str:
        """
        获取用于检索的完整文本
        
        仅返回语义查询，不附加关键词，以避免污染稠密向量
        关键词应仅用于稀疏检索或BM25
        """
        return self.semantic_query


class SearchResult(BaseModel):
    """
    单个检索结果

    包含:
    - 记忆原子
    - 相似度分数
    - 匹配原因（用于解释）
    """
    memory: MemoryAtom
    score: float
    match_reason: str = ""

    # 可选的额外信息
    vector_score: float = 0.0  # 原始向量相似度
    boost_applied: float = 0.0  # 应用的加权

    @model_validator(mode='after')
    def set_default_match_reason(self) -> 'SearchResult':
        """初始化后处理"""
        if not self.match_reason:
            self.match_reason = f"语义匹配 (score: {self.score:.2f})"
        return self


class SearchResults(BaseModel):
    """
    检索结果集合

    包含:
    - 结果列表
    - 检索元信息
    """
    results: List[SearchResult] = Field(default_factory=list)
    total_candidates: int = 0  # 初始候选数量
    latency_ms: float = 0.0  # 检索耗时

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def get_memories(self) -> List[MemoryAtom]:
        """获取所有记忆原子"""
        return [r.memory for r in self.results]

    def is_empty(self) -> bool:
        return len(self.results) == 0


class RetrievalResult(BaseModel):
    """
    RetrievalEngine 统一输出数据模型

    面向上层业务（如 RetrievalFamiliar）的结构化返回：
    - memories: 便于后续业务处理（访问统计、排序、二次过滤等）
    - rendered_context: 可直接注入 prompt 的上下文
    """

    memories: List[MemoryAtom] = Field(default_factory=list)
    rendered_context: str = ""
    latency_ms: float = 0.0
    memories_count: int = 0
    search_results: Optional[SearchResults] = None

    def is_empty(self) -> bool:
        return len(self.memories) == 0

    def get_context_for_prompt(self) -> str:
        if self.is_empty():
            return ""
        return self.rendered_context


# ========== 导出列表 ==========

__all__ = [
    "QueryFilters",
    "RetrievalQuery",
    "SearchResult",
    "SearchResults",
    "RetrievalResult",
]
