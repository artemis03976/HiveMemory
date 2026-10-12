"""
RetrievalEngine 单元测试 (Phase B — renderer 解耦后)

测试覆盖:
- 有结果时正确返回 memories
- 空结果时返回空列表
- 参数传递 (top_k, score_threshold)
- 返回结果字段完整性
- 延迟测量
"""

from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.config.patchouli import (
    DenseRetrieverConfig,
    RerankerConfig,
    SparseRetrieverConfig,
)
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.query import QueryFilters
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    SearchResult,
    SearchResults,
)
from hivememory.engines.retrieval.reranker import NoopReranker, create_reranker
from hivememory.engines.retrieval.retriever import DenseRetriever, SparseRetriever
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _make_memory(title="测试记忆") -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1", session_id="s1"),
        index=IndexLayer(
            title=title,
            summary="这是一段足够长的测试摘要用于通过验证",
            tags=["t1"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="内容"),
    )


def _make_query(text="测试查询") -> RetrievalQuery:
    return RetrievalQuery(
        semantic_query=text,
        keywords=[],
        filters=QueryFilters(),
        belong_to=make_identity_scope(user_id="u1", agent_id="a1").workspace_identity,
        from_actor=make_identity_scope(user_id="u1", agent_id="a1").actor_identity,
    )


class TestRetrievalEngine:

    def setup_method(self):
        self.mock_retriever = AsyncMock()
        self.engine = RetrievalEngine(retriever=self.mock_retriever, reranker=NoopReranker())

    @pytest.mark.asyncio
    async def test_retrieve_with_results(self):
        mem = _make_memory()
        self.mock_retriever.retrieve.return_value = SearchResults(
            results=[SearchResult(memory=mem, score=0.9)]
        )

        result = await self.engine.retrieve(_make_query())

        assert result.memories_count == 1
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        self.mock_retriever.retrieve.return_value = SearchResults(results=[])

        result = await self.engine.retrieve(_make_query())

        assert result.memories_count == 0

    @pytest.mark.asyncio
    async def test_retrieve_passes_parameters(self):
        self.mock_retriever.retrieve.return_value = SearchResults(results=[])

        await self.engine.retrieve(_make_query(), top_k=10, score_threshold=0.5)

        call_kwargs = self.mock_retriever.retrieve.call_args
        assert call_kwargs[1]["top_k"] == 10
        assert call_kwargs[1]["score_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_result_fields(self):
        mem = _make_memory()
        search_results = SearchResults(results=[SearchResult(memory=mem, score=0.9)])
        self.mock_retriever.retrieve.return_value = search_results

        result = await self.engine.retrieve(_make_query())

        assert result.memories == [mem]
        assert result.memories_count == 1


class TestRetrievalEngineRerankStage:
    """重排是引擎中召回之后的独立阶段，接收任意检索器的输出。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("retriever_cls", "config"),
        [
            (
                DenseRetriever,
                DenseRetrieverConfig(enable_time_decay=False, enable_confidence_boost=False),
            ),
            (SparseRetriever, SparseRetrieverConfig()),
        ],
        ids=["dense", "sparse"],
    )
    async def test_recall_output_is_reranked_from_candidate_pool(self, retriever_cls, config):
        first, second = _make_memory("M1"), _make_memory("M2")
        storage = AsyncMock()
        storage.search = AsyncMock(
            return_value=[{"memory": first, "score": 0.9}, {"memory": second, "score": 0.8}]
        )
        rerank_service = Mock()
        # 候选按召回顺序 [M1, M2] 送入重排；重排分数把 M2 提到首位
        rerank_service.compute_score = Mock(return_value=[-5.0, 5.0])
        engine = RetrievalEngine(
            retriever=retriever_cls(storage, config),
            reranker=create_reranker(RerankerConfig(enabled=True, top_k=20), rerank_service),
        )

        result = await engine.retrieve(_make_query(), top_k=1)

        assert result.memories == [second]
        # 召回取 max(top_k, reranker.top_k) 条候选，而不是只取最终的 top_k
        assert storage.search.call_args.kwargs["top_k"] == 20
