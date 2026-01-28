"""
HiveMemory - 记忆检索模块 (MemoryRetrieval)

该模块负责智能检索相关记忆并注入到对话上下文中。

核心组件:
- DenseRetriever: 稠密向量检索器 (语义匹配)
- SparseRetriever: 稀疏向量检索器 (BGE-M3/BM25, 精准实体匹配)
- HybridRetriever: 混合检索引擎 (稠密 + 稀疏 + RRF 融合)
- ReciprocalRankFusion: RRF 结果融合
- AdaptiveWeightedFusion: 自适应加权融合 (Phase 4)
- FullContextRenderer: 完整上下文渲染器
- CascadeContextRenderer: 瀑布式上下文渲染器
- CompactContextRenderer: 紧凑上下文渲染器
- RetrievalEngine: 统一入口门面 (接口)

对应设计文档: PROJECT.md 第 5 章

状态: Stage 2 改进中 (混合检索)
作者: HiveMemory Team
版本: 0.6.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hivememory.patchouli.config import MemoryRetrievalConfig

logger = logging.getLogger(__name__)


# 接口定义
from hivememory.engines.retrieval.interfaces import (
    BaseMemoryRetriever,
    BaseContextRenderer,
    BaseReranker,
    BaseFusion,
)

# 数据模型
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    RetrievalResult,
    SearchResult,
    SearchResults,
    RenderFormat,
)

# Engine
from hivememory.engines.retrieval.engine import RetrievalEngine

# 过滤器适配器
from hivememory.engines.retrieval.filter_adapter import (
    FilterConverter,
    QdrantFilterConverter,
)

# 混合检索组件
from hivememory.engines.retrieval.retriever import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    CachedRetriever,
    create_retriever,
)
from hivememory.engines.retrieval.fusion import (
    ReciprocalRankFusion,
    AdaptiveWeightedFusion,
)
from hivememory.engines.retrieval.reranker import (
    NoopReranker,
    CrossEncoderReranker,
)

# 渲染器
from hivememory.engines.retrieval.renderer import (
    FullContextRenderer,
    CascadeContextRenderer,
    CompactContextRenderer,
    create_renderer,
)

if TYPE_CHECKING:
    from hivememory.patchouli.config import MemoryRetrievalConfig


__all__ = [
    # 主类
    "RetrievalEngine",
    # 数据模型
    "RetrievalQuery",
    "RetrievalResult",
    "SearchResult",
    "SearchResults",
    "RenderFormat",
    # 接口
    "BaseMemoryRetriever",
    "BaseContextRenderer",
    "BaseReranker",
    "BaseFusion",
    # 过滤器适配器
    "FilterConverter",
    "QdrantFilterConverter",
    # 混合检索组件
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CachedRetriever",
    "ReciprocalRankFusion",
    "AdaptiveWeightedFusion",
    "NoopReranker",
    "CrossEncoderReranker",
    "create_retriever",
    # 渲染
    "FullContextRenderer",
    "CascadeContextRenderer",
    "CompactContextRenderer",
    "create_renderer",
]
