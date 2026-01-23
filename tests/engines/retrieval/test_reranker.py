"""
测试 CrossEncoderReranker

测试 BGE-Reranker-v2-m3 重排序器的功能。
"""

import pytest
from unittest.mock import Mock, patch
from hivememory.engines.retrieval.reranker import (
    CrossEncoderReranker,
    NoopReranker,
    create_reranker,
)
from hivememory.engines.retrieval.models import SearchResults, SearchResult, RetrievalQuery
from hivememory.core.models import MemoryAtom, IndexLayer, MetaData, PayloadLayer, MemoryType, MemoryVisibility
from hivememory.patchouli.config import RerankerConfig
from hivememory.infrastructure.rerank.base import BaseRerankService
from datetime import datetime
from uuid import uuid4


@pytest.fixture
def mock_memory():
    """创建测试用的 MemoryAtom"""
    index = IndexLayer(
        title="Test Memory Title",
        summary="This is a test summary for the memory fixture with enough characters",
        memory_type=MemoryType.FACT,
    )
    meta = MetaData(
        source_agent_id="test_agent",
        user_id="test_user",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    payload = PayloadLayer(content="Test content for memory")
    return MemoryAtom(id=uuid4(), index=index, meta=meta, payload=payload)


@pytest.fixture
def sample_results(mock_memory):
    """创建测试用的 SearchResults"""
    return SearchResults(
        results=[
            SearchResult(memory=mock_memory, score=0.9, match_reason="dense"),
            SearchResult(memory=mock_memory, score=0.7, match_reason="sparse"),
            SearchResult(memory=mock_memory, score=0.5, match_reason="rrf"),
        ],
        total_candidates=3,
    )


@pytest.fixture
def sample_query():
    """创建测试用的 RetrievalQuery"""
    return RetrievalQuery(
        semantic_query="测试查询",
    )


class TestCrossEncoderReranker:
    """测试 CrossEncoderReranker"""

    def test_rerank_basic(self, sample_results, sample_query):
        """测试基本重排序功能"""
        mock_service = Mock(spec=BaseRerankService)
        # 返回的分数顺序与原始结果不同，测试重排序
        mock_service.compute_score.return_value = [0.5, 0.9, 0.3]

        config = RerankerConfig(top_k=10)
        reranker = CrossEncoderReranker(service=mock_service, config=config)

        result = reranker.rerank(sample_results, sample_query)

        # 验证结果数量
        assert len(result.results) == 3

        # 验证重排序后: 第二个结果 (0.9 -> sigmoid=0.711) 排第一
        scores = [r.score for r in result.results]
        # sigmoid(0.5)=0.622, sigmoid(0.9)=0.711, sigmoid(0.3)=0.574
        assert scores[0] == pytest.approx(0.711, rel=0.01)  # 原始 0.9
        assert scores[1] == pytest.approx(0.622, rel=0.01)  # 原始 0.5
        assert scores[2] == pytest.approx(0.574, rel=0.01)  # 原始 0.3

        # 验证 match_reason 被更新
        assert "Rerank" in result.results[0].match_reason
        
        mock_service.compute_score.assert_called_once()

    def test_rerank_empty(self, sample_query):
        """测试空结果处理"""
        mock_service = Mock(spec=BaseRerankService)
        reranker = CrossEncoderReranker(service=mock_service)
        empty_results = SearchResults()

        result = reranker.rerank(empty_results, sample_query)

        assert result.is_empty()
        mock_service.compute_score.assert_not_called()

    def test_rerank_top_k_filtering(self, sample_results, sample_query):
        """测试 top_k 过滤功能"""
        mock_service = Mock(spec=BaseRerankService)
        mock_service.compute_score.return_value = [0.5, 0.9]  # 只返回2个分数

        config = RerankerConfig(top_k=2)  # 只重排序前2个
        reranker = CrossEncoderReranker(service=mock_service, config=config)

        result = reranker.rerank(sample_results, sample_query)

        # 只有 top_k=2 个结果被重排序
        assert len(result.results) == 2
        mock_service.compute_score.assert_called_once()

    def test_rerank_no_normalization(self, sample_results, sample_query):
        """测试不标准化分数"""
        mock_service = Mock(spec=BaseRerankService)
        mock_service.compute_score.return_value = [0.5, 0.9, 0.3]

        config = RerankerConfig(top_k=10, normalize_scores=False)
        reranker = CrossEncoderReranker(service=mock_service, config=config)

        result = reranker.rerank(sample_results, sample_query)

        # 验证使用原始分数 (未经过 sigmoid)
        scores = [r.score for r in result.results]
        # 注意：这里期望的顺序是降序排列后的
        # 0.9 排第一，0.5 排第二，0.3 排第三
        assert scores == [0.9, 0.5, 0.3]  

    def test_rerank_error_fallback(self, sample_results, sample_query):
        """测试计算失败时的降级处理"""
        mock_service = Mock(spec=BaseRerankService)
        mock_service.compute_score.side_effect = Exception("CUDA OOM")

        reranker = CrossEncoderReranker(service=mock_service)

        result = reranker.rerank(sample_results, sample_query)

        # 返回原始结果
        assert len(result.results) == len(sample_results.results)
        # 验证分数未改变
        original_scores = [r.score for r in sample_results.results]
        result_scores = [r.score for r in result.results]
        assert original_scores == result_scores

    def test_normalize_score(self):
        """测试分数标准化函数"""
        mock_service = Mock(spec=BaseRerankService)
        reranker = CrossEncoderReranker(service=mock_service)

        # 测试 sigmoid 函数
        assert reranker._normalize_score(0) == pytest.approx(0.5, rel=0.01)
        assert reranker._normalize_score(5) == pytest.approx(0.993, rel=0.01)
        assert reranker._normalize_score(-5) == pytest.approx(0.0067, rel=0.01)


class TestNoopReranker:
    """测试 NoopReranker"""

    def test_passthrough(self, sample_results, sample_query):
        """测试透传行为"""
        reranker = NoopReranker()
        result = reranker.rerank(sample_results, sample_query)

        # 返回相同的结果
        assert len(result.results) == len(sample_results.results)
        assert result.results == sample_results.results


class TestCreateReranker:
    """测试工厂函数"""

    def test_create_noop_reranker_when_disabled(self):
        """测试 enabled=False 时创建 NoopReranker"""
        config = RerankerConfig(enabled=False)
        reranker = create_reranker(config)
        assert isinstance(reranker, NoopReranker)

    def test_create_cross_encoder_reranker(self):
        """测试创建 CrossEncoderReranker"""
        mock_service = Mock(spec=BaseRerankService)
        config = RerankerConfig(enabled=True, type="cross_encoder")
        
        # 必须提供 service
        reranker = create_reranker(config, service=mock_service)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_default_config(self):
        """测试默认配置创建 CrossEncoderReranker"""
        # 如果不提供 service，应该降级为 NoopReranker (根据 create_reranker 逻辑)
        reranker = create_reranker()
        assert isinstance(reranker, NoopReranker)
        
        # 提供 service 则创建 CrossEncoderReranker
        mock_service = Mock(spec=BaseRerankService)
        reranker = create_reranker(service=mock_service)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_with_none_config(self):
        """测试 None 配置使用默认值"""
        mock_service = Mock(spec=BaseRerankService)
        reranker = create_reranker(None, service=mock_service)
        assert isinstance(reranker, CrossEncoderReranker)
