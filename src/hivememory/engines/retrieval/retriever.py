"""
记忆检索器模块 (MemoryRetrievers)

包含各类记忆检索器的实现：
1. DenseRetriever: 稠密向量检索
2. SparseRetriever: 稀疏向量检索
3. HybridRetriever: 混合检索 (Dense + Sparse + RRF)
4. CachedRetriever: 带缓存的检索装饰器
"""

import logging
import math
import time
from datetime import datetime
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from hivememory.patchouli.config import (
    DenseRetrieverConfig,
    SparseRetrieverConfig,
)
from hivememory.engines.retrieval.interfaces import BaseMemoryRetriever
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    SearchResult,
    SearchResults,
    QueryFilters,
)
from hivememory.engines.retrieval.fusion import ReciprocalRankFusion
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.reranker import NoopReranker, CrossEncoderReranker, create_reranker
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.infrastructure.rerank.base import BaseRerankService

if TYPE_CHECKING:
    from hivememory.patchouli.config import HybridRetrieverConfig

logger = logging.getLogger(__name__)

# Qdrant 过滤器转换器单例
_qdrant_filter_converter = QdrantFilterConverter()


class DenseRetriever(BaseMemoryRetriever):
    """
    稠密向量检索器

    使用 Qdrant 的稠密向量进行语义检索，捕获模糊语义匹配。
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        config: Optional[DenseRetrieverConfig] = None
    ):
        """
        初始化稠密检索器

        Args:
            storage: QdrantMemoryStore 实例
            config: 检索器配置
        """
        self.storage = storage
        self.config = config or DenseRetrieverConfig()

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行稠密向量检索

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        start_time = time.time()

        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold

        # 构建��滤条件
        filters = _qdrant_filter_converter.convert(query.filters) if query.filters else None

        # 获取搜索文本
        search_text = query.get_search_text()
        logger.debug(f"Dense检索: '{search_text[:50]}...', filters={filters}")

        try:
            # 执行稠密向量检索
            raw_results = self.storage.search_memories(
                query_text=search_text,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
                mode="dense"
            )
        except Exception as e:
            logger.error(f"Dense检索失败: {e}")
            return SearchResults(latency_ms=(time.time() - start_time) * 1000)

        # 转换结果并应用加权
        search_results = []
        for hit in raw_results:
            memory = hit["memory"]
            vector_score = hit["score"]

            # 计算最终分数
            final_score = vector_score
            boost = 0.0

            # 时间衰减
            if self.config.enable_time_decay:
                decay = self._calculate_time_decay(memory.meta.updated_at)
                boost = (1 - decay) * 0.1  # 最多 10% 的时间惩罚
                final_score = vector_score * (1 - boost)

            # 置信度加权
            if self.config.enable_confidence_boost:
                confidence_boost = memory.meta.confidence_score * 0.05
                final_score += confidence_boost

            search_results.append(SearchResult(
                memory=memory,
                score=final_score,
                vector_score=vector_score,
                boost_applied=boost,
                match_reason=f"Dense: {vector_score:.3f}"
            ))

        latency = (time.time() - start_time) * 1000
        logger.info(f"Dense检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")

        return SearchResults(
            results=search_results,
            total_candidates=len(raw_results),
            latency_ms=latency
        )

    def _calculate_time_decay(self, updated_at: datetime) -> float:
        """
        计算时间衰减系数

        使用指数衰减: decay = e^(-λt)
        其中 λ = ln(2) / half_life_days

        Args:
            updated_at: 更新时间

        Returns:
            衰减系数 (0-1)，越新越接近 1
        """
        now = datetime.now()
        delta = now - updated_at
        days_elapsed = delta.total_seconds() / (24 * 3600)

        # 指数衰减
        lambda_val = math.log(2) / self.config.time_decay_days
        decay = math.exp(-lambda_val * days_elapsed)

        return decay


class SparseRetriever(BaseMemoryRetriever):
    """
    稀疏向量检索器

    使用 BGE-M3 的稀疏向量进行词汇级精准匹配。
    捕获精准实体匹配，如特定函数名、错误码、专有名词等。
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        config: Optional[SparseRetrieverConfig] = None
    ):
        """
        初始化稀疏检索器

        Args:
            storage: QdrantMemoryStore 实例
            config: 检索器配置
        """
        self.storage = storage
        self.config = config or SparseRetrieverConfig()

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行稀疏向量检索

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        start_time = time.time()

        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold

        # 构建��滤条件
        filters = _qdrant_filter_converter.convert(query.filters) if query.filters else None

        # 获取搜索文本
        search_text = query.get_search_text()
        logger.debug(f"Sparse检索: '{search_text[:50]}...', filters={filters}")

        try:
            # 执行稀疏向量检索
            raw_results = self.storage.search_memories(
                query_text=search_text,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
                mode="sparse"
            )
        except Exception as e:
            logger.error(f"Sparse检索失败: {e}")
            return SearchResults(latency_ms=(time.time() - start_time) * 1000)

        # 转换结果
        search_results = []
        for hit in raw_results:
            memory = hit["memory"]
            sparse_score = hit["score"]

            # 稀疏检索不应用时间衰减，直接使用原始分数
            search_results.append(SearchResult(
                memory=memory,
                score=sparse_score,
                match_reason=f"Sparse: {sparse_score:.3f}"
            ))

        latency = (time.time() - start_time) * 1000
        logger.info(f"Sparse检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")

        return SearchResults(
            results=search_results,
            total_candidates=len(raw_results),
            latency_ms=latency
        )


class HybridRetriever(BaseMemoryRetriever):
    """
    混合检索器

    结合稠密向量检索、稀疏向量检索和 RRF 融合，提供高质量的检索结果。

    架构:
        1. Parallel Recall: DenseRetriever || SparseRetriever
        2. Fusion: RRF 合并结果
        3. Rerank: 可选的重排序
        4. Return: Top-K 最终结果
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        config: Optional["HybridRetrieverConfig"] = None,
        reranker_service: Optional[BaseRerankService] = None,
    ):
        """
        初始化混合检索器

        Args:
            storage: QdrantMemoryStore 实例
            config: 统一混合检索配置（HybridRetrieverConfig）
            reranker_service: 重排序服务实例 (可选)
        """
        self.storage = storage

        if config is None:
            from hivememory.patchouli.config import HybridRetrieverConfig
            config = HybridRetrieverConfig()

        self.enable_parallel = config.enable_parallel
        self.enable_hybrid_search = config.enable_hybrid_search
        self.default_top_k = config.top_k
        self.default_threshold = config.score_threshold

        # 初始化稠密检索器
        if config.dense.enabled:
            self.dense_retriever = DenseRetriever(storage, config.dense)
        else:
            self.dense_retriever = None

        # 初始化稀疏检索器
        if config.sparse.enabled:
            self.sparse_retriever = SparseRetriever(storage, config.sparse)
        else:
            self.sparse_retriever = None

        # 初始化融合器
        self.fusion = ReciprocalRankFusion(config.fusion)

        # 初始化重排序器
        if config.reranker.enabled:
            if reranker_service is None:
                logger.warning("HybridRetriever: Reranker enabled but no service provided. Disabling reranker.")
                self.reranker = None
            else:
                self.reranker = CrossEncoderReranker(service=reranker_service, config=config.reranker)
        else:
            self.reranker = None

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行混合检索

        如果 enable_hybrid_search=True，执行稠密+稀疏并行召回和 RRF 融合。
        否则，仅执行稠密向量检索。

        Args:
            query: 处理后的查询
            top_k: 返回数量（可选）
            score_threshold: 相似度阈值（可选）

        Returns:
            检索结果集合
        """
        top_k = top_k or self.default_top_k
        score_threshold = score_threshold or self.default_threshold

        if self.enable_hybrid_search:
            return self._search_hybrid(query, top_k, score_threshold)
        else:
            return self._search_dense(query, top_k, score_threshold)

    def _search_hybrid(
        self,
        query: RetrievalQuery,
        top_k: int,
        score_threshold: float
    ) -> SearchResults:
        """
        执行混合检索 (稠密 + 稀疏 + RRF)

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        start_time = time.time()

        # 并行召回
        if self.enable_parallel:
            dense_results, sparse_results = self._parallel_recall(query)
        else:
            dense_results, sparse_results = self._sequential_recall(query)

        # RRF 融合
        fused_results = self.fusion.fuse(dense_results, sparse_results)

        # 可选重排序
        if self.reranker:
            fused_results = self.reranker.rerank(fused_results, query)

        # 应用分数阈值
        # 注意: 仅当使用了非 NoopReranker (即提供了有意义的绝对分数) 时才应用阈值
        if score_threshold > 0:
            if isinstance(self.reranker, NoopReranker):
                fused_results.results = [
                    r for r in fused_results.results
                    if r.score >= score_threshold
                ]
            else:
                logger.debug(f"跳过阈值过滤 (threshold={score_threshold}): 当前使用的是 RRF 分数，不适用 Cosine 阈值")

        # 截取 top_k
        fused_results.results = fused_results.results[:top_k]

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"混合检索完成: 返回 {len(fused_results)} 条结果, "
            f"耗时 {latency:.1f}ms"
        )

        return fused_results

    def _parallel_recall(
        self,
        query: RetrievalQuery
    ) -> Tuple[SearchResults, SearchResults]:
        """并行执行稠密和稀疏检索"""
        dense_results = None
        sparse_results = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.dense_retriever.retrieve, query): "dense",
                executor.submit(self.sparse_retriever.retrieve, query): "sparse"
            }

            for future in as_completed(futures):
                result_type = futures[future]
                try:
                    if result_type == "dense":
                        dense_results = future.result()
                    else:
                        sparse_results = future.result()
                except Exception as e:
                    logger.error(f"{result_type.capitalize()} 检索失败: {e}")

        # 返回空结果作为 fallback
        if dense_results is None:
            dense_results = SearchResults()
        if sparse_results is None:
            sparse_results = SearchResults()

        return dense_results, sparse_results

    def _sequential_recall(
        self,
        query: RetrievalQuery
    ) -> Tuple[SearchResults, SearchResults]:
        """顺序执行稠密和稀疏检索"""
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)
        return dense_results, sparse_results

    def _search_dense(
        self,
        query: RetrievalQuery,
        top_k: int,
        score_threshold: float
    ) -> SearchResults:
        """
        仅执行稠密向量检索

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索结果集合
        """
        return self.dense_retriever.retrieve(query, top_k, score_threshold)


class CachedRetriever(BaseMemoryRetriever):
    """
    带缓存的检索器
    
    简单的内存缓存装饰器，用于减少重复检索的开销。
    """
    
    def __init__(
        self,
        retriever: BaseMemoryRetriever,
        cache_ttl_seconds: int = 60,
        max_cache_size: int = 100
    ):
        """
        初始化缓存检索器
        
        Args:
            retriever: 被装饰的检索器
            cache_ttl_seconds: 缓存过期时间 (秒)
            max_cache_size: 最大缓存条目数
        """
        self.retriever = retriever
        self.ttl = cache_ttl_seconds
        self.max_size = max_cache_size
        self._cache: Dict[str, Tuple[float, SearchResults]] = {}
        
    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行检索 (带缓存)
        """
        # 生成缓存键
        cache_key = f"{query.semantic_query}_{top_k}_{score_threshold}"
        
        now = time.time()
        
        # 检查缓存
        if cache_key in self._cache:
            timestamp, result = self._cache[cache_key]
            if now - timestamp < self.ttl:
                logger.debug(f"缓存命中: {cache_key}")
                return result
            else:
                del self._cache[cache_key]
        
        # 执行检索
        result = self.retriever.retrieve(query, top_k, score_threshold)
        
        # 更新缓存
        if len(self._cache) >= self.max_size:
            # 简单清理: FIFO
            first_key = next(iter(self._cache))
            del self._cache[first_key]
            
        self._cache[cache_key] = (now, result)
        
        return result


def create_default_retriever(
    storage: QdrantMemoryStore, 
    config: Optional["HybridRetrieverConfig"] = None,
    reranker_service: Optional[BaseRerankService] = None
) -> BaseMemoryRetriever:
    """
    创建默认混合检索器

    Args:
        storage: QdrantMemoryStore 实例
        config: 混合检索配置
        reranker_service: Rerank 服务实例

    Returns:
        HybridRetriever 实例
    """
    if config is None:
        from hivememory.patchouli.config import HybridRetrieverConfig
        config = HybridRetrieverConfig()
    
    return HybridRetriever(storage=storage, config=config, reranker_service=reranker_service)


__all__ = [
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CachedRetriever",
    "create_default_retriever",
]