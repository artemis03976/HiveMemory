"""
HiveMemory - Retrieval 模块数据模型

定义了记忆检索模块的所有数据模型和配置类。

数据模型列表:
- SearchResult: 单个检索结果
- SearchResults: 检索结果集合

配置类:
- 配置类从 core.config 导入，避免重复定义

作者: HiveMemory Team
版本: 0.1.0
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from hivememory.core.models import MemoryAtom, MemoryType


# ========== 数据模型 ==========

class QueryFilters(BaseModel):
    """
    结构化过滤条件
    
    用于混合检索时的元数据过滤
    """
    memory_type: Optional[MemoryType] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    tags: List[str] = Field(default_factory=list)
    source_agent_id: Optional[str] = None
    user_id: Optional[str] = None
    min_confidence: float = 0.0
    visibility: Optional[str] = None
    
    def to_qdrant_filter(self) -> Dict[str, Any]:
        """
        转换为 Qdrant 过滤条件格式
        
        Returns:
            字典格式的过滤条件
        """
        filters = {}
        
        if self.memory_type:
            filters["index.memory_type"] = self.memory_type.value
            
        if self.user_id:
            filters["meta.user_id"] = self.user_id
            
        if self.source_agent_id:
            filters["meta.source_agent_id"] = self.source_agent_id
            
        if self.min_confidence > 0:
            filters["meta.confidence_score"] = {"gte": self.min_confidence}
            
        if self.visibility:
            filters["meta.visibility"] = self.visibility
            
        # 注意: 时间范围和标签需要特殊处理，这里暂不支持
        # Qdrant 的时间过滤需要存储为数字时间戳
        
        return filters
    
    def is_empty(self) -> bool:
        """检查过滤条件是否为空"""
        return (
            self.memory_type is None
            and self.time_range is None
            and len(self.tags) == 0
            and self.source_agent_id is None
            and self.user_id is None
            and self.min_confidence == 0.0
            and self.visibility is None
        )


class ProcessedQuery(BaseModel):
    """
    处理后的结构化查询
    
    包含:
    - 语义查询文本（用于向量检索）
    - 提取的关键词
    - 结构化过滤条件
    """
    semantic_query: str  # 用于向量检索的语义查询
    original_query: str  # 原始查询
    keywords: List[str] = Field(default_factory=list)  # 提取的关键词
    filters: QueryFilters = Field(default_factory=QueryFilters)  # 过滤条件
    rewritten: bool = False  # 是否经过改写
    
    def get_search_text(self) -> str:
        """
        获取用于检索的完整文本
        
        仅返回语义查询，不附加关键词，以避免污染稠密向量
        关键词应仅用于稀疏检索或BM25
        """
        return self.semantic_query


# ========== 检索结果数据模型 ==========

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
    query_used: str = ""  # 实际使用的查询

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def get_memories(self) -> List[MemoryAtom]:
        """获取所有记忆原子"""
        return [r.memory for r in self.results]

    def is_empty(self) -> bool:
        return len(self.results) == 0


class RenderFormat(str, Enum):
    """
    渲染格式枚举

    Attributes:
        XML: XML 标签格式
        MARKDOWN: Markdown 格式
    """
    XML = "xml"
    MARKDOWN = "markdown"


# ========== 导出列表 ==========

__all__ = [
    # 数据模型
    "QueryFilters",
    "ProcessedQuery",
    "SearchResult",
    "SearchResults",

    # 枚举
    "RenderFormat",
]
