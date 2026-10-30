"""
RetrievalEngine 单元测试 (Phase B — renderer 解耦后)

测试覆盖:
- 有结果时正确返回 memories
- 空结果时返回空列表
- 结果条数：请求的 top_k，缺省时为配置的 default_top_k；不向检索器传阈值
- 最终截取发生在重排阶段之后
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
from hivememory.infrastructure.model_service import ModelLoadState
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
        self.engine = RetrievalEngine(
            retriever=self.mock_retriever, reranker=NoopReranker(), default_top_k=3
        )

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
    async def test_request_top_k_sets_candidate_count_without_threshold(self):
        """请求给出 top_k 时据此取候选；分数阈值属于 Dense 配置，不由引擎传入。"""
        self.mock_retriever.retrieve.return_value = SearchResults(results=[])
        query = _make_query()

        await self.engine.retrieve(query, top_k=10)

        assert self.mock_retriever.retrieve.call_args.kwargs == {"query": query, "top_k": 10}

    @pytest.mark.asyncio
    async def test_missing_top_k_uses_configured_default_after_rerank(self):
        """请求未给出 top_k：按 default_top_k 取候选，并在重排位置之后截取。"""
        memories = [_make_memory(f"M{i}") for i in range(5)]
        # 检索器可能交出多于请求数的候选，最终条数只由重排之后的截取决定
        self.mock_retriever.retrieve.return_value = SearchResults(
            results=[SearchResult(memory=m, score=0.9 - i * 0.1) for i, m in enumerate(memories)]
        )

        result = await self.engine.retrieve(_make_query())

        assert self.mock_retriever.retrieve.call_args.kwargs["top_k"] == 3
        assert result.memories == memories[:3]

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
        rerank_service = Mock(state=ModelLoadState.LOADED)
        # 候选按召回顺序 [M1, M2] 送入重排；重排分数把 M2 提到首位
        rerank_service.compute_score = AsyncMock(return_value=[-5.0, 5.0])
        engine = RetrievalEngine(
            retriever=retriever_cls(storage, config),
            reranker=create_reranker(RerankerConfig(enabled=True, top_k=20), rerank_service),
            default_top_k=5,
        )

        result = await engine.retrieve(_make_query(), top_k=1)

        assert result.memories == [second]
        # 召回取 max(top_k, reranker.top_k) 条候选，而不是只取最终的 top_k
        assert storage.search.call_args.kwargs["top_k"] == 20

    @pytest.mark.asyncio
    async def test_missing_top_k_sizes_candidate_pool_from_default(self):
        """请求未给出 top_k：候选池按 max(default_top_k, reranker.top_k)，结果按 default_top_k。"""
        memories = [_make_memory(f"M{i}") for i in range(4)]
        storage = AsyncMock()
        storage.search = AsyncMock(
            return_value=[{"memory": m, "score": 0.9} for m in memories],
        )
        rerank_service = Mock(state=ModelLoadState.LOADED)
        # 重排把召回顺序完全反转
        rerank_service.compute_score = AsyncMock(return_value=[1.0, 2.0, 3.0, 4.0])
        engine = RetrievalEngine(
            retriever=SparseRetriever(storage, SparseRetrieverConfig()),
            reranker=create_reranker(RerankerConfig(enabled=True, top_k=20), rerank_service),
            default_top_k=2,
        )

        result = await engine.retrieve(_make_query())

        assert storage.search.call_args.kwargs["top_k"] == 20
        assert result.memories == [memories[3], memories[2]]
