"""
MemoryRetriever 单元测试

测试覆盖:
- HybridRetriever (混合检索)
- CachedRetriever (缓存检索)
- 结果排序和打分
- 时间衰减逻辑
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import time

from hivememory.core.models import MemoryAtom, MemoryType, IndexLayer, PayloadLayer, MetaData
from hivememory.patchouli.config import (
    DenseRetrieverConfig,
    SparseRetrieverConfig,
    ReciprocalRankFusionConfig,
    HybridRetrieverConfig,
)
from hivememory.engines.retrieval.retriever import HybridRetriever, CachedRetriever, DenseRetriever, SearchResults
from hivememory.engines.retrieval.models import RetrievalQuery, QueryFilters, SearchResult

class TestDenseRetriever:
    """测试稠密检索器"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.config = DenseRetrieverConfig()
        self.retriever = DenseRetriever(
            storage=self.mock_storage,
            config=self.config
        )
        
        # 准备一些测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(title="M1", summary="Summary of M1 content", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C1"),
            meta=MetaData(source_agent_id="a1", user_id="u1", updated_at=datetime.now(), confidence_score=0.9)
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(title="M2", summary="Summary of M2 content", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C2"),
            meta=MetaData(source_agent_id="a1", user_id="u1", updated_at=datetime.now() - timedelta(days=60), confidence_score=0.8)
        )

    def test_search_basic(self):
        """测试基本检索"""
        # 模拟存储返回
        self.mock_storage.search_memories.return_value = [
            {"memory": self.memory1, "score": 0.9},
            {"memory": self.memory2, "score": 0.8}
        ]

        query = RetrievalQuery(semantic_query="test")
        results = self.retriever.retrieve(query, top_k=2)

        assert len(results) == 2
        assert results.results[0].memory.index.title == "M1"
        self.mock_storage.search_memories.assert_called_once()

    def test_search_with_filters(self):
        """测试带过滤条件的检索"""
        self.mock_storage.search_memories.return_value = []
        
        filters = QueryFilters(memory_type=MemoryType.FACT, user_id="u1")
        query = RetrievalQuery(semantic_query="test", filters=filters)
        
        self.retriever.retrieve(query)
        
        # 验证过滤条件传递
        call_args = self.mock_storage.search_memories.call_args
        assert call_args.kwargs["filters"]["index.memory_type"] == "FACT"
        assert call_args.kwargs["filters"]["meta.user_id"] == "u1"

    def test_time_decay(self):
        """测试时间衰减"""
        # M1: 新, 原始分 0.84
        # M2: 旧(180天前), 原始分 0.85

        # 更新 M2 时间为 180 天前
        self.memory2.meta.updated_at = datetime.now() - timedelta(days=180)

        self.mock_storage.search_memories.return_value = [
            {"memory": self.memory1, "score": 0.84},
            {"memory": self.memory2, "score": 0.85}
        ]
        
        query = RetrievalQuery(semantic_query="test")
        results = self.retriever.retrieve(query)
        
        # M1 虽然原始分低，但因为 M2 时间久远衰减，M1 应该排在前面
        # 或者至少验证 M2 的分数被降低了
        # DenseRetriever._calculate_time_decay 逻辑：
        # decay = exp(-lambda * days)
        # boost = (1 - decay) * 0.1
        # final = score * (1 - boost)
        # 180天，decay 应该很小，boost 接近 0.1，score * 0.9 -> 0.85 * 0.9 = 0.765
        # M1: 0天，decay=1，boost=0，score=0.84
        # 所以 M1 > M2
        
        assert results.results[0].memory.index.title == "M1"

    def test_match_reason(self):
        """测试匹配原因生成"""
        self.mock_storage.search_memories.return_value = [
            {"memory": self.memory1, "score": 0.9}
        ]

        query = RetrievalQuery(
            semantic_query="test",
            keywords=["t1"],
            filters={}
        )
        results = self.retriever.retrieve(query)
        
        assert "Dense" in results.results[0].match_reason


class TestHybridRetriever:
    """测试混合检索器"""

    def setup_method(self):
        self.config = HybridRetrieverConfig(
            enable_parallel=False,
            reranker={"enabled": False}
        )
        
        self.mock_dense = Mock()
        self.mock_sparse = Mock()
        self.mock_fusion = Mock()
        
        self.searcher = HybridRetriever(
            config=self.config,
            dense_retriever=self.mock_dense,
            sparse_retriever=self.mock_sparse,
            fusion=self.mock_fusion
        )
        
        # 准备一些测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(title="M1", summary="Summary of M1 content is long enough", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C1"),
            meta=MetaData(source_agent_id="a1", user_id="u1", updated_at=datetime.now())
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(title="M2", summary="Summary of M2 content is long enough", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C2"),
            meta=MetaData(source_agent_id="a1", user_id="u1", updated_at=datetime.now())
        )

    def test_search_hybrid(self):
        """测试混合检索 (Dense + Sparse + RRF)"""
        # 模拟 Dense 返回
        self.mock_dense.retrieve.return_value = SearchResults(results=[
            SearchResult(memory=self.memory1, score=0.9)
        ])
        
        # 模拟 Sparse 返回
        self.mock_sparse.retrieve.return_value = SearchResults(results=[
            SearchResult(memory=self.memory2, score=0.85)
        ])
        
        # 模拟 Fusion 返回
        self.mock_fusion.fuse.return_value = SearchResults(results=[
            SearchResult(memory=self.memory1, score=0.9, match_reason="RRF"),
            SearchResult(memory=self.memory2, score=0.8, match_reason="RRF")
        ])

        query = RetrievalQuery(semantic_query="test")
        results = self.searcher.retrieve(query, top_k=2)

        # 验证调用
        self.mock_dense.retrieve.assert_called_once()
        self.mock_sparse.retrieve.assert_called_once()
        self.mock_fusion.fuse.assert_called_once()
        
        assert len(results.results) == 2



class TestCachedRetriever:
    """测试带缓存的检索器"""

    def setup_method(self):
        self.mock_retriever = Mock(spec=HybridRetriever)
        self.cached_retriever = CachedRetriever(retriever=self.mock_retriever, cache_ttl_seconds=1)
        
        self.query = RetrievalQuery(semantic_query="test")
        self.results = SearchResults(results=[])

    def test_cache_hit(self):
        """测试缓存命中"""
        self.mock_retriever.retrieve.return_value = self.results
        
        # 第一次调用
        self.cached_retriever.retrieve(self.query)
        self.mock_retriever.retrieve.assert_called_once()
        
        # 第二次调用（应该命中缓存）
        self.cached_retriever.retrieve(self.query)
        self.mock_retriever.retrieve.assert_called_once()  # 调用次数不变

    def test_cache_expiration(self):
        """测试缓存过期"""
        self.mock_retriever.retrieve.return_value = self.results
        
        # 第一次调用
        self.cached_retriever.retrieve(self.query)
        
        # 等待过期
        time.sleep(1.1)
        
        # 第二次调用（应该重新检索）
        self.cached_retriever.retrieve(self.query)
        assert self.mock_retriever.retrieve.call_count == 2

    def test_cache_eviction(self):
        """测试缓存清理"""
        retriever = CachedRetriever(self.mock_retriever, max_cache_size=1)
        
        q1 = RetrievalQuery(semantic_query="q1")
        q2 = RetrievalQuery(semantic_query="q2")
        
        retriever.retrieve(q1)
        retriever.retrieve(q2)  # 这应该触发清理 q1
        
        # 再次搜索 q1，应该重新调用底层
        retriever.retrieve(q1)
        assert self.mock_retriever.retrieve.call_count == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
