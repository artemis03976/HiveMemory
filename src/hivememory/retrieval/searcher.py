"""
混合检索引擎

职责:
    执行混合检索（稠密 + 稀疏向量 + RRF 融合）并返回排序结果

实现策略:
    - 稠密检索 (DenseRetriever): 使用 Qdrant Dense Vector，捕获模糊语义
    - 稀疏检索 (SparseRetriever): 使用 BGE-M3 Sparse Vector，捕获精准实体
    - 并行召回 (Parallel Recall): 同时执行两路检索
    - RRF 融合: 合并两路结果
    - 重排序 (Reranker): 可选的精排

对应设计文档: PROJECT.md 5.1 节
"""

from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

if TYPE_CHECKING:
    from hivememory.core.config import HybridSearchConfig

from hivememory.core.models import MemoryAtom
from hivememory.core.config import (
    DenseRetrieverConfig,
    SparseRetrieverConfig,
    FusionConfig,
)
from hivememory.retrieval.interfaces import MemorySearcher
from hivememory.retrieval.retriever import DenseRetriever, SparseRetriever
from hivememory.retrieval.fusion import ReciprocalRankFusion
from hivememory.retrieval.reranker import NoopReranker, CrossEncoderReranker
from hivememory.retrieval.models import (
    ProcessedQuery,
    QueryFilters,
    SearchResult,
    SearchResults,
)

logger = logging.getLogger(__name__)


class HybridSearcher(MemorySearcher):
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
        storage,  # QdrantMemoryStore
        # 统一配置（优先）
        config: Optional["HybridSearchConfig"] = None,
        # 新式混合搜索配置（单独指定，向后兼容）
        dense_config: Optional["DenseRetrieverConfig"] = None,
        sparse_config: Optional["SparseRetrieverConfig"] = None,
        fusion_config: Optional["FusionConfig"] = None,
        reranker: Optional["BaseReranker"] = None,
        enable_parallel: bool = True,
        # 启用混合搜索模式
        enable_hybrid_search: bool = False,
    ):
        """
        初始化混合检索器

        Args:
            storage: QdrantMemoryStore 实例
            config: 统一混合检索配置（HybridSearchConfig，优先使用）
            dense_config: 稠密检索器配置（当 config=None 时使用）
            sparse_config: 稀疏检索器配置（当 config=None 时使用）
            fusion_config: RRF 融合配置（当 config=None 时使用）
            reranker: 重排序器 (可选)
            enable_parallel: 是否启用并行召回（当 config=None 时使用）
            enable_hybrid_search: 启用混合搜索模式 (稠密+稀疏+RRF)（当 config=None 时使用）

        Examples:
            >>> # 使用统一配置
            >>> from hivememory.core.config import HybridSearchConfig
            >>> config = HybridSearchConfig()
            >>> searcher = HybridSearcher(storage, config=config)
            >>>
            >>> # 使用单独配置（向后兼容）
            >>> searcher = HybridSearcher(storage, enable_hybrid_search=True)
        """
        self.storage = storage

        # 如果提供了统一配置，使用它
        if config is not None:
            self.enable_parallel = config.enable_parallel
            self.enable_hybrid_search = config.enable_hybrid_search

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
                self.reranker = CrossEncoderReranker(
                    model_name=config.reranker.model_name,
                    device=config.reranker.device,
                    use_fp16=config.reranker.use_fp16,
                    batch_size=config.reranker.batch_size,
                    top_k=config.reranker.top_k,
                    normalize_scores=config.reranker.normalize_scores,
                )
            else:
                self.reranker = None

            # 默认参数
            self.default_top_k = config.fusion.final_top_k
            self.default_threshold = config.score_threshold
        else:
            # 向后兼容：使用单独的参数
            self.enable_parallel = enable_parallel
            self.enable_hybrid_search = enable_hybrid_search

            # 初始化稠密检索器
            dense_config = dense_config or DenseRetrieverConfig(top_k=50)
            self.dense_retriever = DenseRetriever(storage, dense_config)

            # 初始化稀疏检索器
            sparse_config = sparse_config or SparseRetrieverConfig(top_k=50)
            self.sparse_retriever = SparseRetriever(storage, sparse_config)

            # 初始化融合器
            fusion_config = fusion_config or FusionConfig()
            self.fusion = ReciprocalRankFusion(fusion_config)

            # 初始化重排序器
            self.reranker = reranker or CrossEncoderReranker()

            # 默认参数
            self.default_top_k = 5
            self.default_threshold = 0.0

    def search(
        self,
        query: ProcessedQuery,
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
        query: ProcessedQuery,
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
        import time
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
        # 原始 RRF 分数 (通常 < 0.1) 不应使用针对 Cosine 相似度 (通常 > 0.7) 的阈值过滤
        is_noop_reranker = isinstance(self.reranker, NoopReranker)
        
        if score_threshold > 0 and not is_noop_reranker:
            fused_results.results = [
                r for r in fused_results.results
                if r.score >= score_threshold
            ]
        elif score_threshold > 0 and is_noop_reranker:
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
        query: ProcessedQuery
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
        query: ProcessedQuery
    ) -> Tuple[SearchResults, SearchResults]:
        """顺序执行稠密和稀疏检索"""
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)
        return dense_results, sparse_results

    def _search_dense(
        self,
        query: ProcessedQuery,
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
    
    def _calculate_time_decay(self, updated_at: datetime) -> float:
        """
        计算时间衰减系数
        
        使用指数衰减: decay = e^(-λt)
        其中 λ = ln(2) / half_life_days
        
        Returns:
            衰减系数 (0-1)，越新越接近 1
        """
        import math
        
        now = datetime.now()
        delta = now - updated_at
        days_elapsed = delta.total_seconds() / (24 * 3600)
        
        # 指数衰减
        lambda_val = math.log(2) / self.time_decay_days
        decay = math.exp(-lambda_val * days_elapsed)
        
        return decay


class CachedSearcher(MemorySearcher):
    """
    带缓存的检索器
    
    简单的内存缓存装饰器，用于减少重复检索的开销。
    """
    
    def __init__(
        self,
        searcher: MemorySearcher,
        cache_ttl_seconds: int = 60,
        max_cache_size: int = 100
    ):
        """
        初始化缓存检索器
        
        Args:
            searcher: 被装饰的检索器
            cache_ttl_seconds: 缓存过期时间 (秒)
            max_cache_size: 最大缓存条目数
        """
        self.searcher = searcher
        self.ttl = cache_ttl_seconds
        self.max_size = max_cache_size
        self._cache: Dict[str, Tuple[float, SearchResults]] = {}
        
    def search(
        self,
        query: ProcessedQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行检索 (带缓存)
        """
        import time
        
        # 生成缓存键
        cache_key = f"{query.original_query}_{top_k}_{score_threshold}"
        
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
        result = self.searcher.search(query, top_k, score_threshold)
        
        # 更新缓存
        if len(self._cache) >= self.max_size:
            # 简单清理: FIFO
            first_key = next(iter(self._cache))
            del self._cache[first_key]
            
        self._cache[cache_key] = (now, result)
        
        return result


def create_default_searcher(storage, config: Optional["HybridSearchConfig"] = None) -> HybridSearcher:
    """
    创建默认混合检索器

    Args:
        storage: QdrantMemoryStore 实例
        config: 混合检索配置

    Returns:
        HybridSearcher 实例
    """
    if config is None:
        from hivememory.core.config import HybridSearchConfig
        config = HybridSearchConfig()
    
    return HybridSearcher(storage=storage, config=config)


__all__ = [
    "HybridSearcher",
    "CachedSearcher",
    "create_default_searcher",
]
