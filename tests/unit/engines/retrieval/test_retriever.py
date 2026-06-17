"""
MemoryRetriever 单元测试

测试覆盖:
- HybridRetriever (混合检索)
- DenseRetriever (稠密检索)
- 结果排序和打分
- 时间衰减逻辑
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from hivememory.core.models import Identity, MemoryAtom, MemoryType, IndexLayer, PayloadLayer, MetaData
from hivememory.system.config import (
    DenseRetrieverConfig,
    SparseRetrieverConfig,
    ReciprocalRankFusionConfig,
    HybridRetrieverConfig,
)
from hivememory.engines.retrieval.retriever import HybridRetriever, DenseRetriever, SearchResults
from hivememory.engines.retrieval.models import RetrievalQuery, QueryFilters, SearchResult


class TestDenseRetriever:
    """测试稠密检索器"""

    def setup_method(self):
        self.mock_storage = AsyncMock()
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

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """测试基本检索"""
        # 模拟存储返回
        self.mock_storage.search_memories = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.9},
            {"memory": self.memory2, "score": 0.8}
        ])

        query = RetrievalQuery(semantic_query="test")
        results = await self.retriever.retrieve(query, top_k=2)

        assert len(results) == 2
        assert results.results[0].memory.index.title == "M1"
        self.mock_storage.search_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """测试带过滤条件的检索"""
        self.mock_storage.search_memories = AsyncMock(return_value=[])
        
        filters = QueryFilters(memory_type=MemoryType.FACT, identity=Identity(user_id="u1"))
        query = RetrievalQuery(semantic_query="test", filters=filters)
        
        await self.retriever.retrieve(query)
        
        # 验证过滤条件传递 (现在返回 qdrant Filter 对象)
        call_args = self.mock_storage.search_memories.call_args
        qdrant_filter = call_args.kwargs["filters"]
        # Filter 对象的 must 条件中应包含 user_id 和 memory_type
        must_conditions = qdrant_filter.must
        field_keys = []
        for cond in must_conditions:
            if hasattr(cond, 'key'):
                field_keys.append(cond.key)
        assert "meta.user_id" in field_keys
        assert "index.memory_type" in field_keys

    @pytest.mark.asyncio
    async def test_time_decay(self):
        """测试时间衰减"""
        # M1: 新, 原始分 0.84
        # M2: 旧(180天前), 原始分 0.85

        # 更新 M2 时间为 180 天前
        self.memory2.meta.updated_at = datetime.now() - timedelta(days=180)

        self.mock_storage.search_memories = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.84},
            {"memory": self.memory2, "score": 0.85}
        ])
        
        query = RetrievalQuery(semantic_query="test")
        results = await self.retriever.retrieve(query)
        
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

    @pytest.mark.asyncio
    async def test_match_reason(self):
        """测试匹配原因生成"""
        self.mock_storage.search_memories = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.9}
        ])

        query = RetrievalQuery(
            semantic_query="test",
            keywords=["t1"],
            filters={}
        )
        results = await self.retriever.retrieve(query)
        
        assert "Dense" in results.results[0].match_reason

    @pytest.mark.asyncio
    async def test_time_decay_with_aware_datetime(self):
        """测试 aware datetime 时间衰减不抛异常"""
        self.memory1.meta.updated_at = datetime.now(ZoneInfo("UTC")) - timedelta(days=1)
        self.mock_storage.search_memories = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.9}
        ])

        query = RetrievalQuery(semantic_query="test")
        results = await self.retriever.retrieve(query)

        assert len(results.results) == 1
        assert results.results[0].score > 0


class TestHybridRetriever:
    """测试混合检索器"""

    def setup_method(self):
        self.config = HybridRetrieverConfig(
            enable_parallel=False,
            reranker={"enabled": False}
        )
        
        self.mock_dense = AsyncMock()
        self.mock_sparse = AsyncMock()
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

    @pytest.mark.asyncio
    async def test_search_hybrid(self):
        """测试混合检索 (Dense + Sparse + RRF)"""
        # 模拟 Dense 返回
        self.mock_dense.retrieve = AsyncMock(return_value=SearchResults(results=[
            SearchResult(memory=self.memory1, score=0.9)
        ]))
        
        # 模拟 Sparse 返回
        self.mock_sparse.retrieve = AsyncMock(return_value=SearchResults(results=[
            SearchResult(memory=self.memory2, score=0.85)
        ]))
        
        # 模拟 Fusion 返回
        self.mock_fusion.fuse.return_value = SearchResults(results=[
            SearchResult(memory=self.memory1, score=0.9, match_reason="RRF"),
            SearchResult(memory=self.memory2, score=0.8, match_reason="RRF")
        ])

        query = RetrievalQuery(semantic_query="test")
        results = await self.searcher.retrieve(query, top_k=2)

        # 验证调用
        self.mock_dense.retrieve.assert_called_once()
        self.mock_sparse.retrieve.assert_called_once()
        self.mock_fusion.fuse.assert_called_once()
        
        assert len(results.results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
