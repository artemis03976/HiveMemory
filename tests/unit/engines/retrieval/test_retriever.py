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
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


def _make_access_context():
    return make_access_context(user_id="u1", agent_id="a1")


class TestDenseRetriever:
    """测试稠密检索器"""

    def setup_method(self):
        self.mock_storage = AsyncMock()
        self.config = DenseRetrieverConfig()
        self.retriever = DenseRetriever(
            mid_term=self.mock_storage,
            config=self.config
        )
        
        # 准备一些测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(title="M1", summary="Summary of M1 content", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C1"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(),
                confidence_score=0.9,
            )
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(title="M2", summary="Summary of M2 content", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C2"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now() - timedelta(days=60),
                confidence_score=0.8,
            )
        )

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """测试基本检索"""
        # 模拟存储返回
        self.mock_storage.search = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.9},
            {"memory": self.memory2, "score": 0.8}
        ])

        query = RetrievalQuery(semantic_query="test", access_context=_make_access_context())
        results = await self.retriever.retrieve(query, top_k=2)

        assert len(results) == 2
        assert results.results[0].memory.index.title == "M1"
        self.mock_storage.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """测试带过滤条件的检索"""
        self.mock_storage.search = AsyncMock(return_value=[])
        
        filters = QueryFilters(memory_type=MemoryType.FACT)
        query = RetrievalQuery(
            semantic_query="test",
            filters=filters,
            access_context=_make_access_context(),
        )
        
        await self.retriever.retrieve(query)
        
        # 过滤条件属于业务查询；授权 scope 由独立 AccessContext 传递。
        call_args = self.mock_storage.search.call_args
        assert call_args.args[0] == query.access_context
        assert call_args.kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_time_decay(self):
        """测试时间衰减"""
        # M1: 新, 原始分 0.84
        # M2: 旧(180天前), 原始分 0.85

        # 更新 M2 时间为 180 天前
        self.memory2.meta.updated_at = datetime.now() - timedelta(days=180)

        self.mock_storage.search = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.84},
            {"memory": self.memory2, "score": 0.85}
        ])
        
        query = RetrievalQuery(semantic_query="test", access_context=_make_access_context())
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
        self.mock_storage.search = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.9}
        ])

        query = RetrievalQuery(
            semantic_query="test",
            keywords=["t1"],
            filters={},
            access_context=_make_access_context(),
        )
        results = await self.retriever.retrieve(query)
        
        assert "Dense" in results.results[0].match_reason

    @pytest.mark.asyncio
    async def test_time_decay_with_aware_datetime(self):
        """测试 aware datetime 时间衰减不抛异常"""
        self.memory1.meta.updated_at = datetime.now(ZoneInfo("UTC")) - timedelta(days=1)
        self.mock_storage.search = AsyncMock(return_value=[
            {"memory": self.memory1, "score": 0.9}
        ])

        query = RetrievalQuery(semantic_query="test", access_context=_make_access_context())
        results = await self.retriever.retrieve(query)

        # 1 天前的新记忆几乎无衰减，分数接近原始值
        assert len(results.results) == 1
        assert results.results[0].score == pytest.approx(0.9, abs=0.05)


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
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(),
            )
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(title="M2", summary="Summary of M2 content is long enough", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C2"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(),
            )
        )

    @pytest.mark.asyncio
    async def test_search_hybrid(self):
        """测试混合检索 (Dense + Sparse + 真实 RRF 融合)"""
        from hivememory.engines.retrieval.fusion import ReciprocalRankFusion
        from hivememory.system.config import ReciprocalRankFusionConfig

        self.searcher.fusion = ReciprocalRankFusion(config=ReciprocalRankFusionConfig())
        self.mock_dense.retrieve = AsyncMock(return_value=SearchResults(results=[
            SearchResult(memory=self.memory1, score=0.9)
        ]))
        self.mock_sparse.retrieve = AsyncMock(return_value=SearchResults(results=[
            SearchResult(memory=self.memory2, score=0.85)
        ]))

        query = RetrievalQuery(semantic_query="test", access_context=_make_access_context())
        results = await self.searcher.retrieve(query, top_k=2)

        # 两个检索通道的结果经真实 RRF 融合后都被保留
        assert len(results.results) == 2
        assert {r.memory.id for r in results.results} == {self.memory1.id, self.memory2.id}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
