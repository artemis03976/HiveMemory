"""
HiveMemory - 记忆检索模块 (MemoryRetrieval)

该模块负责智能检索相关记忆并注入到对话上下文中。

核心组件:
- RetrievalRouter: 判断查询是否需要记忆
- QueryProcessor: 查询预处理与重写
- DenseRetriever: 稠密向量检索器 (语义匹配)
- SparseRetriever: 稀疏向量检索器 (BGE-M3/BM25, 精准实体匹配)
- HybridRetriever: 混合检索引擎 (稠密 + 稀疏 + RRF 融合)
- ReciprocalRankFusion: RRF 结果融合
- ContextRenderer: 上下文渲染器
- RetrievalEngine: 统一入口门面

对应设计文档: PROJECT.md 第 5 章

状态: Stage 2 改进中 (混合检索)
作者: HiveMemory Team
版本: 0.4.0
"""

import logging
from typing import Optional, TYPE_CHECKING, Union
from hivememory.memory.storage import QdrantMemoryStore

if TYPE_CHECKING:
    from hivememory.core.config import MemoryRetrievalConfig

logger = logging.getLogger(__name__)


# 接口定义
from hivememory.retrieval.interfaces import (
    RetrievalRouter,
    QueryProcessor as QueryProcessorInterface,
    MemoryRetriever,
    ContextRenderer as ContextRendererInterface,
    RetrievalEngine,
    BaseReranker
)

# 数据模型
from hivememory.retrieval.models import (
    QueryFilters,
    ProcessedQuery,
    SearchResult,
    SearchResults,
    RenderFormat,
    RetrievalResult,
)

# 查询处理
from hivememory.retrieval.query import (
    QueryProcessor,
    TimeExpressionParser,
    MemoryTypeDetector,
    create_default_processor,
)

# 路由器
from hivememory.retrieval.router import (
    SimpleRouter,
    LLMRouter,
    AlwaysRetrieveRouter,
    NeverRetrieveRouter,
    create_default_router,
)

# 混合检索组件
from hivememory.retrieval.retriever import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    CachedRetriever,
    create_default_retriever,
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
    ContextRenderer,
    MinimalRenderer,
    create_default_renderer,
)

# 引擎门面
from hivememory.retrieval.engine import (
    MemoryRetrievalEngine,
)


def create_default_retrieval_engine(
    storage: QdrantMemoryStore,
    config: Optional["MemoryRetrievalConfig"] = None,
) -> MemoryRetrievalEngine:
    """
    创建默认配置的记忆检索引擎

    根据配置自动创建并配置所有子组件：
    - RetrievalRouter: 检索路由器
    - QueryProcessor: 查询处理器
    - HybridRetriever: 混合检索器
    - ContextRenderer: 上下文渲染器

    Args:
        storage: QdrantMemoryStore 实例
        config: 检索配置（可选，使用默认配置）

    Returns:
        RetrievalEngine: 检索引擎实例

    Examples:
        >>> from hivememory.retrieval import create_default_retrieval_engine
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> storage = QdrantMemoryStore()
        >>> engine = create_default_retrieval_engine(storage=storage)
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.core.config import MemoryRetrievalConfig
        >>> config = MemoryRetrievalConfig()
        >>> config.router.router_type = "llm"
        >>> engine = create_default_retrieval_engine(storage=storage, config=config)
    """
    if config is None:
        from hivememory.core.config import MemoryRetrievalConfig
        config = MemoryRetrievalConfig()

    # 使用工厂函数创建各组件
    router = create_default_router(config.router)
    processor = create_default_processor(config.processor)
    renderer = create_default_renderer(config.renderer)

    # 创建引擎
    engine = MemoryRetrievalEngine(
        storage=storage,
        router=router,
        processor=processor,
        renderer=renderer,
        config=config,
    )

    logger.info("MemoryRetrievalEngine created with default config")
    return engine


__all__ = [
    # 接口
    "RetrievalRouter",
    "QueryProcessorInterface",
    "MemoryRetriever",
    "ContextRendererInterface",
    "RetrievalEngine",
    "BaseReranker",

    # 数据模型
    "QueryFilters",
    "ProcessedQuery",
    "SearchResult",
    "SearchResults",
    "RenderFormat",

    # 查询处理
    "QueryProcessor",
    "TimeExpressionParser",
    "MemoryTypeDetector",
    "create_default_processor",

    # 路由
    "SimpleRouter",
    "LLMRouter",
    "AlwaysRetrieveRouter",
    "NeverRetrieveRouter",
    "create_default_router",

    # 混合检索组件
    "DenseRetriever",
    "SparseRetriever",
    "ReciprocalRankFusion",
    "BaseReranker",
    "NoopReranker",
    "CrossEncoderReranker",
    "HybridRetriever",
    "CachedRetriever",
    "create_default_retriever",

    # 渲染
    "ContextRenderer",
    "MinimalRenderer",
    "create_default_renderer",

    # 引擎
    "RetrievalResult",
    "MemoryRetrievalEngine",
    "create_default_retrieval_engine",
]
