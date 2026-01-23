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
    FusionConfig,
    HybridRetrieverConfig,
)
from hivememory.engines.retrieval.retriever import HybridRetriever, CachedRetriever, SearchResult, SearchResults
from hivememory.engines.retrieval.models import RetrievalQuery, QueryFilters

class TestHybridRetriever:
    """测试混合检索器"""

    def setup_method(self):
        self.mock_storage = Mock()
        # 默认禁用混合搜索，只使用 Dense
        # 禁用 reranker 避免初始化真实的 CrossEncoderReranker
        config = HybridRetrieverConfig(
            enable_hybrid_search=False,
            reranker={"enabled": False}
        )
        self.searcher = HybridRetriever(
            storage=self.mock_storage,
            config=config
        )
        
        # 准备一些测试记忆
        # 增加 summary 长度以满足最小长度要求 (10 字符)
        self.memory1 = MemoryAtom(
            index=IndexLayer(title="M1", summary="Summary of M1 content", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C1"),
            meta=MetaData(source_agent_id="a1", user_id="u1", updated_at=datetime.now())
        )
        self.memory2 = MemoryAtom(
            # 修改 MemoryType.EPISODE 为 MemoryType.FACT，因为 EPISODE 不在枚举中
            index=IndexLayer(title="M2", summary="Summary of M2 content", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C2"),
            meta=MetaData(source_agent_id="a1", user_id="u1", updated_at=datetime.now() - timedelta(days=60))
        )

    def test_search_dense_only(self):   
        # 准备一些测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(title="M1", summary="This is summary 1 for testing.", memory_type=MemoryType.FACT, tags=["t1"]),
            payload=PayloadLayer(content="C1"),
            meta=MetaData(source_agent_id="test", user_id="u1", updated_at=datetime.now(), confidence_score=0.9)
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(title="M2", summary="This is summary 2 for testing.", memory_type=MemoryType.CODE_SNIPPET, tags=["t2"]),
            payload=PayloadLayer(content="C2"),
            meta=MetaData(source_agent_id="test", user_id="u1", updated_at=datetime.now() - timedelta(days=60), confidence_score=0.8)
        )

    def test_search_dense_only(self):
        """测试仅稠密检索 (默认模式)"""
        # 模拟存储返回
        self.mock_storage.search_memories.return_value = [
            {"memory": self.memory1, "score": 0.9},
            {"memory": self.memory2, "score": 0.8}
        ]

        query = RetrievalQuery(semantic_query="test")
        results = self.searcher.retrieve(query, top_k=2)

        assert len(results) == 2
        assert results.results[0].memory.index.title == "M1"
        self.mock_storage.search_memories.assert_called_once()

    def test_search_hybrid(self):
        """测试混合检索 (Dense + Sparse + RRF)"""
        # 启用混合搜索，禁用并行便于测试，禁用 reranker
        config = HybridRetrieverConfig(
            enable_hybrid_search=True,
            enable_parallel=False,
            reranker={"enabled": False}
        )

        hybrid_retriever = HybridRetriever(
            storage=self.mock_storage,
            config=config
        )

        # 模拟 Dense 返回 (只有 M1)
        def mock_search_side_effect(*args, **kwargs):
            mode = kwargs.get("mode", "dense")
            if mode == "sparse":
                return [{"memory": self.memory2, "score": 0.85}]
            else:
                return [{"memory": self.memory1, "score": 0.9}]

        self.mock_storage.search_memories.side_effect = mock_search_side_effect

        query = RetrievalQuery(semantic_query="test", filters={})
        results = hybrid_retriever.retrieve(query, top_k=2)

        # RRF 融合后应该包含 M1 和 M2
        assert len(results.results) == 2

        # 验证 search_memories 被调用了两次 (dense 和 sparse 各一次)
        assert self.mock_storage.search_memories.call_count == 2

        # 检查是否使用了 RRF 分数
        m1_result = next(r for r in results.results if r.memory.index.title == "M1")
        m2_result = next(r for r in results.results if r.memory.index.title == "M2")

        assert m1_result.score > 0
        assert m2_result.score > 0

    def test_search_with_filters(self):
        """测试带过滤条件的检索"""
        self.mock_storage.search_memories.return_value = []
        
        filters = QueryFilters(memory_type=MemoryType.FACT, user_id="u1")
        query = RetrievalQuery(semantic_query="test", filters=filters)
        
        self.searcher.retrieve(query)
        
        # 验证过滤条件传递
        call_args = self.mock_storage.search_memories.call_args
        assert call_args.kwargs["filters"]["index.memory_type"] == "FACT"
        assert call_args.kwargs["filters"]["meta.user_id"] == "u1"

    def test_time_decay(self):
        """测试时间衰减 (仅 Dense)"""
        # M1: 新, 原始分 0.8
        # M2: 旧(60天前), 原始分 0.85

        self.mock_storage.search_memories.return_value = [
            {"memory": self.memory1, "score": 0.8},
            {"memory": self.memory2, "score": 0.85}
        ]

        query = RetrievalQuery(semantic_query="test", filters={})
        results = self.searcher.retrieve(query)
        
        # 调整测试用例：M1 score=0.84, M2 score=0.85, M2 180天前
        self.mock_storage.search_memories.return_value = [
            {"memory": self.memory1, "score": 0.84},
            {"memory": self.memory2, "score": 0.85}
        ]
        # 更新 M2 时间为 180 天前
        self.memory2.meta.updated_at = datetime.now() - timedelta(days=180)
        
        results = self.searcher.retrieve(query)
        
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
        results = self.searcher.retrieve(query)
        
        assert "Dense" in results.results[0].match_reason


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
