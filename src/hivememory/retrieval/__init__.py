"""
HiveMemory - 记忆检索模块 (MemoryRetrieval)

该模块负责智能检索相关记忆并注入到对话上下文中。

核心组件:
- RetrievalRouter: 判断查询是否需要记忆
- QueryProcessor: 查询预处理与重写
- DenseRetriever: 稠密向量检索器 (语义匹配)
- SparseRetriever: 稀疏向量检索器 (BGE-M3/BM25, 精准实体匹配)
- HybridSearcher: 混合检索引擎 (稠密 + 稀疏 + RRF 融合)
- ReciprocalRankFusion: RRF 结果融合
- ContextRenderer: 上下文渲染器
- RetrievalEngine: 统一入口门面

对应设计文档: PROJECT.md 第 5 章

状态: Stage 2 改进中 (混合检索)
作者: HiveMemory Team
版本: 0.3.0
"""

# 接口定义
from hivememory.retrieval.interfaces import (
    RetrievalRouter,
    QueryProcessor as QueryProcessorInterface,
    MemorySearcher,
    ContextRenderer as ContextRendererInterface,
    RetrievalEngine as RetrievalEngineInterface,
    BaseReranker
)

# 查询处理
from hivememory.retrieval.query import (
    QueryProcessor,
    TimeExpressionParser,
    MemoryTypeDetector,
)

# 路由器
from hivememory.retrieval.router import (
    SimpleRouter,
    LLMRouter,
    AlwaysRetrieveRouter,
    NeverRetrieveRouter,
)

# 数据模型 (统一从 models.py 导入)
from hivememory.retrieval.models import (
    QueryFilters,
    ProcessedQuery,
    SearchResult,
    SearchResults,
    RenderFormat,
)

# 检索器
from hivememory.retrieval.searcher import (
    HybridSearcher,
)

# 混合检索组件
from hivememory.retrieval.dense_retriever import (
    DenseRetriever,
)
from hivememory.retrieval.sparse_retriever import (
    SparseRetriever,
)
from hivememory.retrieval.fusion import (
    ReciprocalRankFusion,
)
from hivememory.retrieval.reranker import (
    NoopReranker,
    CrossEncoderReranker,
)

# 渲染器
from hivememory.retrieval.renderer import (
    RenderFormat,
    ContextRenderer,
    MinimalRenderer,
    render_memories_for_context,
)

# 引擎门面
from hivememory.retrieval.engine import (
    RetrievalResult,
    RetrievalEngine,
    create_retrieval_engine,
)


__all__ = [
    # 接口
    "RetrievalRouter",
    "QueryProcessorInterface",
    "MemorySearcher",
    "ContextRendererInterface",
    "RetrievalEngineInterface",
    "BaseReranker",

    # 数据模型
    "QueryFilters",
    "ProcessedQuery",
    "SearchResult",
    "SearchResults",
    "RenderFormat",

    # 查询
    "QueryProcessor",
    "TimeExpressionParser",
    "MemoryTypeDetector",

    # 路由
    "SimpleRouter",
    "LLMRouter",
    "AlwaysRetrieveRouter",
    "NeverRetrieveRouter",

    # 检索
    "DenseRetriever",
    "SparseRetriever",
    "ReciprocalRankFusion",
    "NoopReranker",
    "HybridSearcher",

    # 混合检索组件
    "DenseRetriever",
    "SparseRetriever",
    "ReciprocalRankFusion",
    "BaseReranker",
    "NoopReranker",
    "CrossEncoderReranker",

    # 渲染
    "ContextRenderer",
    "MinimalRenderer",
    "render_memories_for_context",

    # 引擎
    "RetrievalResult",
    "RetrievalEngine",
    "create_retrieval_engine",
]
