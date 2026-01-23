"""
HiveMemory - Retrieval 模块数据模型

定义了记忆检索模块的所有数据模型和配置类。

作者: HiveMemory Team
版本: 0.1.0
"""

from typing import List
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from hivememory.core.models import MemoryAtom
from hivememory.patchouli.protocol.models import QueryFilters


# ========== 数据模型 ==========

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


# ========== 导出列表 ==========

__all__ = [
    "QueryFilters",
    "RetrievalQuery",
    "SearchResult",
    "SearchResults",
]
