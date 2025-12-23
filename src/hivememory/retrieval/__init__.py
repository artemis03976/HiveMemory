"""
HiveMemory - 记忆检索模块 (MemoryRetrieval)

该模块负责智能检索相关记忆并注入到对话上下文中。

核心组件 (待实现):
- RetrievalRouter: 判断查询是否需要记忆
- QueryProcessor: 查询预处理与重写
- HybridSearcher: 混合检索引擎 (向量 + BM25 + 结构化)
- Reranker: 检索结果重排序
- ContextRenderer: 上下文渲染器

对应设计文档: PROJECT.md 第 5 章

状态: 骨架接口（Stage 2 实现）
作者: HiveMemory Team
版本: 0.1.0
"""

from hivememory.retrieval.interfaces import (
    RetrievalRouter,
    QueryProcessor,
    MemorySearcher,
)

__all__ = [
    "RetrievalRouter",
    "QueryProcessor",
    "MemorySearcher",
]
