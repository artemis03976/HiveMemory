"""
HiveMemory - 记忆检索模块 (MemoryRetrieval)

该模块负责智能检索相关记忆并注入到对话上下文中。

核心组件:
- RetrievalRouter: 判断查询是否需要记忆
- QueryProcessor: 查询预处理与重写
- HybridSearcher: 混合检索引擎 (向量 + 结构化过滤)
- ContextRenderer: 上下文渲染器
- RetrievalEngine: 统一入口门面

对应设计文档: PROJECT.md 第 5 章

状态: Stage 2 实现完成
作者: HiveMemory Team
版本: 0.2.0
"""

# 接口定义
from hivememory.retrieval.interfaces import (
    RetrievalRouter,
    QueryProcessor as QueryProcessorInterface,
    MemorySearcher,
    ContextRenderer as ContextRendererInterface,
    RetrievalEngine as RetrievalEngineInterface,
    RenderFormat,
)

# 查询处理
from hivememory.retrieval.query import (
    QueryFilters,
    ProcessedQuery,
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

# 检索器
from hivememory.retrieval.searcher import (
    SearchResult,
    SearchResults,
    HybridSearcher,
    CachedSearcher,
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
    "RenderFormat",

    # 查询
    "QueryFilters",
    "ProcessedQuery",
    "QueryProcessor",
    "TimeExpressionParser",
    "MemoryTypeDetector",

    # 路由
    "SimpleRouter",
    "LLMRouter",
    "AlwaysRetrieveRouter",
    "NeverRetrieveRouter",

    # 检索
    "SearchResult",
    "SearchResults",
    "HybridSearcher",
    "CachedSearcher",

    # 渲染
    "ContextRenderer",
    "MinimalRenderer",
    "render_memories_for_context",

    # 引擎
    "RetrievalResult",
    "RetrievalEngine",
    "create_retrieval_engine",
]
