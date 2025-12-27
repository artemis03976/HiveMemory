"""
混合检索引擎

职责:
    执行混合检索（向量 + 元数据过滤）并返回排序结果

实现策略:
    - 向量检索: 使用 Qdrant Dense Vector
    - 元数据过滤: 类型、标签、时间、置信度等
    - 结果排序: 基于相似度和时间衰减

对应设计文档: PROJECT.md 5.1 节
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from hivememory.core.models import MemoryAtom
from hivememory.retrieval.interfaces import MemorySearcher
from hivememory.retrieval.query import ProcessedQuery, QueryFilters

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
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
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.match_reason:
            self.match_reason = f"语义匹配 (score: {self.score:.2f})"


@dataclass
class SearchResults:
    """
    检索结果集合
    
    包含:
    - 结果列表
    - 检索元信息
    """
    results: List[SearchResult] = field(default_factory=list)
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


class HybridSearcher(MemorySearcher):
    """
    混合检索器
    
    结合向量检索和元数据过滤，提供高质量的检索结果
    """
    
    def __init__(
        self,
        storage,  # QdrantMemoryStore
        default_top_k: int = 5,
        default_threshold: float = 0.3,
        enable_time_decay: bool = True,
        time_decay_days: int = 30
    ):
        """
        初始化检索器
        
        Args:
            storage: QdrantMemoryStore 实例
            default_top_k: 默认返回数量
            default_threshold: 默认相似度阈值
            enable_time_decay: 是否启用时间衰减
            time_decay_days: 时间衰减的半衰期（天）
        """
        self.storage = storage
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        self.enable_time_decay = enable_time_decay
        self.time_decay_days = time_decay_days
    
    def search(
        self,
        query: ProcessedQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """
        执行混合检索
        
        Args:
            query: 处理后的查询
            top_k: 返回数量（可选）
            score_threshold: 相似度阈值（可选）
            
        Returns:
            检索结果集合
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.default_top_k
        score_threshold = score_threshold or self.default_threshold
        
        # 1. 构建过滤条件
        filters = query.filters.to_qdrant_filter() if query.filters else {}
        
        # 2. 执行向量检索
        search_text = query.get_search_text()
        logger.debug(f"执行检索: '{search_text[:50]}...', filters={filters}")
        
        try:
            raw_results = self.storage.search_memories(
                query_text=search_text,
                top_k=top_k * 2,  # 取更多结果以便后续过滤
                score_threshold=score_threshold,
                filters=filters if filters else None
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return SearchResults(latency_ms=(time.time() - start_time) * 1000)
        
        # 3. 转换结果
        search_results = []
        for hit in raw_results:
            memory = hit["memory"]
            vector_score = hit["score"]
            
            # 计算最终分数（可选时间衰减）
            final_score = vector_score
            boost = 0.0
            
            if self.enable_time_decay:
                decay = self._calculate_time_decay(memory.meta.updated_at)
                boost = (1 - decay) * 0.1  # 最多 10% 的时间惩罚
                final_score = vector_score * (1 - boost)
            
            # 置信度加权
            confidence_boost = memory.meta.confidence_score * 0.05
            final_score += confidence_boost
            
            search_results.append(SearchResult(
                memory=memory,
                score=final_score,
                vector_score=vector_score,
                boost_applied=boost,
                match_reason=self._generate_match_reason(memory, query)
            ))
        
        # 4. 重新排序
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        # 5. 截取 top_k
        search_results = search_results[:top_k]
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"检索完成: 返回 {len(search_results)} 条结果, 耗时 {latency:.1f}ms")
        
        return SearchResults(
            results=search_results,
            total_candidates=len(raw_results),
            latency_ms=latency,
            query_used=search_text
        )
    
    def search_by_text(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.6,
        memory_type: Optional[str] = None
    ) -> SearchResults:
        """
        简化的文本检索接口
        
        Args:
            query_text: 查询文本
            user_id: 用户 ID（过滤）
            top_k: 返回数量
            score_threshold: 相似度阈值
            memory_type: 记忆类型过滤
            
        Returns:
            检索结果
        """
        from hivememory.core.models import MemoryType
        
        # 构建过滤条件
        filters = QueryFilters(user_id=user_id)
        if memory_type:
            try:
                filters.memory_type = MemoryType(memory_type)
            except ValueError:
                pass
        
        # 构建查询
        query = ProcessedQuery(
            semantic_query=query_text,
            original_query=query_text,
            filters=filters
        )
        
        return self.search(query, top_k, score_threshold)
    
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
    
    def _generate_match_reason(self, memory: MemoryAtom, query: ProcessedQuery) -> str:
        """
        生成匹配原因说明
        
        用于向用户解释为什么召回了这条记忆
        """
        reasons = []
        
        # 类型匹配
        if query.filters.memory_type:
            if memory.index.memory_type == query.filters.memory_type:
                reasons.append(f"类型匹配: {memory.index.memory_type.value}")
        
        # 标签匹配
        if query.keywords:
            matched_tags = set(query.keywords) & set(memory.index.tags)
            if matched_tags:
                reasons.append(f"标签命中: {', '.join(matched_tags)}")
        
        # 语义匹配（默认）
        if not reasons:
            reasons.append("语义相关")
        
        return "; ".join(reasons)


class CachedSearcher:
    """
    带缓存的检索器
    
    对相同查询的结果进行短期缓存，减少重复检索
    """
    
    def __init__(
        self,
        searcher: HybridSearcher,
        cache_ttl_seconds: int = 60,
        max_cache_size: int = 100
    ):
        """
        Args:
            searcher: 底层检索器
            cache_ttl_seconds: 缓存过期时间（秒）
            max_cache_size: 最大缓存数量
        """
        self.searcher = searcher
        self.cache_ttl = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, tuple] = {}  # query_hash -> (results, timestamp)
    
    def search(
        self,
        query: ProcessedQuery,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResults:
        """执行带缓存的检索"""
        import time
        
        # 生成缓存键
        cache_key = self._make_cache_key(query, top_k, score_threshold)
        
        # 检查缓存
        if cache_key in self._cache:
            results, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"命中缓存: {cache_key[:20]}...")
                return results
        
        # 执行检索
        results = self.searcher.search(query, top_k, score_threshold)
        
        # 更新缓存
        self._cache[cache_key] = (results, time.time())
        self._evict_if_needed()
        
        return results
    
    def _make_cache_key(
        self,
        query: ProcessedQuery,
        top_k: Optional[int],
        score_threshold: Optional[float]
    ) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{query.semantic_query}|{top_k}|{score_threshold}"
        if query.filters.user_id:
            key_str += f"|{query.filters.user_id}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _evict_if_needed(self):
        """如果缓存过大，清理最旧的条目"""
        if len(self._cache) > self.max_cache_size:
            # 按时间排序，删除最旧的一半
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1][1]
            )
            entries_to_remove = len(self._cache) // 2
            for key, _ in sorted_items[:entries_to_remove]:
                del self._cache[key]
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


__all__ = [
    "SearchResult",
    "SearchResults",
    "HybridSearcher",
    "CachedSearcher",
]
