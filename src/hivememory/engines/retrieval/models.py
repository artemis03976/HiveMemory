"""
HiveMemory - Retrieval 模块数据模型

定义了记忆检索模块的所有数据模型和配置类。

作者: HiveMemory Team
"""

from pydantic import BaseModel, Field

from hivememory.core.models import ActorIdentity, MemoryAtom, WorkspaceIdentity
from hivememory.core.models.query import QueryFilters


class RetrievalQuery(BaseModel):
    """
    处理后的结构化查询

    包含:
    - 语义查询文本（用于向量检索）
    - 提取的关键词
    - 结构化过滤条件
    """

    semantic_query: str  # 用于向量检索的语义查询
    keywords: list[str] = Field(default_factory=list)  # 提取的关键词
    filters: QueryFilters = Field(default_factory=QueryFilters)  # 过滤条件
    belong_to: WorkspaceIdentity  # 资源归属的 Workspace 硬边界
    from_actor: ActorIdentity  # 本次检索的可见性主体

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


class SearchResults(BaseModel):
    """
    检索结果集合

    包含:
    - 结果列表
    - 检索元信息
    """

    results: list[SearchResult] = Field(default_factory=list)
    total_candidates: int = 0  # 初始候选数量
    latency_ms: float = 0.0  # 检索耗时

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def get_memories(self) -> list[MemoryAtom]:
        """获取所有记忆原子"""
        return [r.memory for r in self.results]

    def is_empty(self) -> bool:
        return len(self.results) == 0


class RetrievalResult(BaseModel):
    """
    RetrievalEngine 统一输出数据模型
    """

    memories: list[MemoryAtom] = Field(default_factory=list)
    latency_ms: float = 0.0
    memories_count: int = 0
    search_results: SearchResults | None = None

    def is_empty(self) -> bool:
        return len(self.memories) == 0


# ========== 导出列表 ==========

__all__ = [
    "QueryFilters",
    "RetrievalQuery",
    "SearchResult",
    "SearchResults",
    "RetrievalResult",
]
