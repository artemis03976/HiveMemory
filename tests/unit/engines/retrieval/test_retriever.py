"""
MemoryRetriever 单元测试

测试覆盖:
- HybridRetriever (混合检索)
- DenseRetriever (稠密检索)
- 结果排序和打分
- 时间衰减逻辑
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest

from hivememory.config.patchouli import (
    DenseRetrieverConfig,
    HybridRetrieverConfig,
    ReciprocalRankFusionConfig,
    SparseRetrieverConfig,
)
from hivememory.core.errors import StorageOfflineError
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.models.query import QueryFilters
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    SearchResult,
)
from hivememory.engines.retrieval.retriever import (
    DenseRetriever,
    HybridRetriever,
    SearchResults,
    create_retriever,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _make_identity_scope():
    return make_identity_scope(user_id="u1", agent_id="a1")


# 检索评分决策时刻：注入 DenseRetriever 的固定时钟，
# 衰减天数由 meta.lifecycle.decay_anchor_at 相对该时刻控制。
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


class TestDenseRetriever:
    """测试稠密检索器"""

    def setup_method(self):
        self.mock_storage = AsyncMock()
        self.config = DenseRetrieverConfig()
        self.retriever = DenseRetriever(
            mid_term=self.mock_storage, config=self.config, now=lambda: FIXED_NOW
        )

        # 准备一些测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(
                title="M1", summary="Summary of M1 content", memory_type=MemoryType.FACT
            ),
            payload=PayloadLayer(content="C1"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(UTC),
                confidence_score=0.9,
            ),
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(
                title="M2", summary="Summary of M2 content", memory_type=MemoryType.FACT
            ),
            payload=PayloadLayer(content="C2"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(UTC) - timedelta(days=60),
                confidence_score=0.8,
            ),
        )

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """测试基本检索"""
        # 模拟存储返回
        self.mock_storage.search = AsyncMock(
            return_value=[
                {"memory": self.memory1, "score": 0.9},
                {"memory": self.memory2, "score": 0.8},
            ]
        )

        query = RetrievalQuery(
            semantic_query="test",
            belong_to=_make_identity_scope().workspace_identity,
            from_actor=_make_identity_scope().actor_identity,
        )
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
            belong_to=_make_identity_scope().workspace_identity,
            from_actor=_make_identity_scope().actor_identity,
        )

        await self.retriever.retrieve(query)

        # 业务过滤与归属、读取策略主体分别传递。
        call_args = self.mock_storage.search.call_args
        assert call_args.args[0] == query.belong_to
        assert call_args.kwargs["from_actor"] == query.from_actor
        assert call_args.kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_time_decay(self):
        """测试时间衰减（衰减基准 decay_anchor_at，决策时刻用注入的 now）"""
        # M1: 新（衰减基准 = 决策时刻，无时间惩罚）, 原始分 0.84
        # M2: 旧(衰减基准 180 天前), 原始分 0.85

        # 衰减基准相对注入的 now 控制：M1 无衰减，M2 为 180 天前
        self.memory1.meta.lifecycle.decay_anchor_at = FIXED_NOW
        self.memory2.meta.lifecycle.decay_anchor_at = FIXED_NOW - timedelta(days=180)

        self.mock_storage.search = AsyncMock(
            return_value=[
                {"memory": self.memory1, "score": 0.84},
                {"memory": self.memory2, "score": 0.85},
            ]
        )

        query = RetrievalQuery(
            semantic_query="test",
            belong_to=_make_identity_scope().workspace_identity,
            from_actor=_make_identity_scope().actor_identity,
        )
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

        # 衰减确实作用在 M2 上（boost > 0），且 M1 无时间惩罚
        assert results.results[1].boost_applied > 0
        assert results.results[0].boost_applied == 0
        assert results.results[0].memory.index.title == "M1"

    @pytest.mark.asyncio
    async def test_match_reason(self):
        """测试匹配原因生成"""
        self.mock_storage.search = AsyncMock(return_value=[{"memory": self.memory1, "score": 0.9}])

        query = RetrievalQuery(
            semantic_query="test",
            keywords=["t1"],
            filters={},
            belong_to=_make_identity_scope().workspace_identity,
            from_actor=_make_identity_scope().actor_identity,
        )
        results = await self.retriever.retrieve(query)

        assert "Dense" in results.results[0].match_reason

    @pytest.mark.asyncio
    async def test_time_decay_with_aware_datetime(self):
        """测试 aware datetime 衰减基准不抛异常"""
        # 相对固定决策时刻 1 天前，保留非 datetime.UTC 的 aware tzinfo
        anchor = (FIXED_NOW - timedelta(days=1)).astimezone(ZoneInfo("UTC"))
        self.memory1.meta.lifecycle.decay_anchor_at = anchor
        self.mock_storage.search = AsyncMock(return_value=[{"memory": self.memory1, "score": 0.9}])

        query = RetrievalQuery(
            semantic_query="test",
            belong_to=_make_identity_scope().workspace_identity,
            from_actor=_make_identity_scope().actor_identity,
        )
        results = await self.retriever.retrieve(query)

        # 1 天前的记忆几乎无衰减，分数接近原始值
        assert len(results.results) == 1
        assert results.results[0].score == pytest.approx(0.9, abs=0.05)


class TestHybridRetriever:
    """测试混合检索器"""

    def setup_method(self):
        self.config = HybridRetrieverConfig(enable_parallel=False)

        self.mock_dense = AsyncMock()
        self.mock_sparse = AsyncMock()
        self.mock_fusion = Mock()

        self.searcher = HybridRetriever(
            config=self.config,
            dense_retriever=self.mock_dense,
            sparse_retriever=self.mock_sparse,
            fusion=self.mock_fusion,
        )

        # 准备一些测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(
                title="M1",
                summary="Summary of M1 content is long enough",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="C1"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(UTC),
            ),
        )
        self.memory2 = MemoryAtom(
            index=IndexLayer(
                title="M2",
                summary="Summary of M2 content is long enough",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="C2"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(UTC),
            ),
        )

    @pytest.mark.asyncio
    async def test_search_hybrid(self):
        """测试混合检索 (Dense + Sparse + 真实 RRF 融合)"""
        from hivememory.engines.retrieval.fusion import ReciprocalRankFusion

        self.searcher.fusion = ReciprocalRankFusion(config=ReciprocalRankFusionConfig())
        self.mock_dense.retrieve = AsyncMock(
            return_value=SearchResults(results=[SearchResult(memory=self.memory1, score=0.9)])
        )
        self.mock_sparse.retrieve = AsyncMock(
            return_value=SearchResults(results=[SearchResult(memory=self.memory2, score=0.85)])
        )

        query = RetrievalQuery(
            semantic_query="test",
            belong_to=_make_identity_scope().workspace_identity,
            from_actor=_make_identity_scope().actor_identity,
        )
        results = await self.searcher.retrieve(query, top_k=2)

        # 两个检索通道的结果经真实 RRF 融合后都被保留
        assert len(results.results) == 2
        assert {r.memory.id for r in results.results} == {self.memory1.id, self.memory2.id}


class TestHybridRetrieverParallelRecall:
    """并行召回：单路普通失败降级为空，存储失败与取消向上传播。"""

    def setup_method(self):
        from hivememory.engines.retrieval.fusion import ReciprocalRankFusion

        self.mock_dense = AsyncMock()
        self.mock_sparse = AsyncMock()
        self.searcher = HybridRetriever(
            config=HybridRetrieverConfig(enable_parallel=True),
            dense_retriever=self.mock_dense,
            sparse_retriever=self.mock_sparse,
            fusion=ReciprocalRankFusion(config=ReciprocalRankFusionConfig()),
        )
        self.memory = MemoryAtom(
            index=IndexLayer(
                title="M1",
                summary="Summary of M1 content is long enough",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="C1"),
            meta=make_memory_metadata(
                source_agent_id="a1",
                user_id="u1",
                updated_at=datetime.now(UTC),
            ),
        )
        scope = _make_identity_scope()
        self.query = RetrievalQuery(
            semantic_query="test",
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
        )

    def _sparse_returns_memory(self) -> None:
        self.mock_sparse.retrieve = AsyncMock(
            return_value=SearchResults(results=[SearchResult(memory=self.memory, score=0.8)])
        )

    @pytest.mark.asyncio
    async def test_programming_failure_on_one_path_propagates(self):
        self.mock_dense.retrieve = AsyncMock(side_effect=RuntimeError("dense index broken"))
        self._sparse_returns_memory()

        with pytest.raises(RuntimeError, match="dense index broken"):
            await self.searcher.retrieve(self.query, top_k=5)

    @pytest.mark.asyncio
    async def test_storage_offline_on_one_path_propagates(self):
        self.mock_dense.retrieve = AsyncMock(
            side_effect=StorageOfflineError(details={"operation": "search_memories"})
        )
        self._sparse_returns_memory()

        with pytest.raises(StorageOfflineError):
            await self.searcher.retrieve(self.query, top_k=5)

    @pytest.mark.asyncio
    async def test_cancellation_on_one_path_propagates_instead_of_fusing(self):
        self.mock_dense.retrieve = AsyncMock(side_effect=asyncio.CancelledError())
        self._sparse_returns_memory()

        with pytest.raises(asyncio.CancelledError):
            await self.searcher.retrieve(self.query, top_k=5)


class TestHybridRetrieverDisabledPath:
    """混合检索关闭其中一路召回时，另一路照常返回。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enable_parallel", [True, False], ids=["parallel", "sequential"])
    async def test_disabled_dense_path_recalls_from_sparse_only(self, enable_parallel):
        memory = MemoryAtom(
            index=IndexLayer(
                title="M1",
                summary="Summary of M1 content is long enough",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="C1"),
            meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        )
        storage = AsyncMock()
        storage.search = AsyncMock(return_value=[{"memory": memory, "score": 0.7}])
        retriever = create_retriever(
            storage,
            HybridRetrieverConfig(enable_parallel=enable_parallel, dense={"enabled": False}),
        )
        scope = _make_identity_scope()
        query = RetrievalQuery(
            semantic_query="test",
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
        )

        results = await retriever.retrieve(query, top_k=5)

        assert [r.memory.id for r in results.results] == [memory.id]
        assert [call.kwargs["mode"] for call in storage.search.call_args_list] == ["sparse"]


def _recall_memory(title: str) -> MemoryAtom:
    return MemoryAtom(
        index=IndexLayer(
            title=title,
            summary="Summary of recall candidate is long enough",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=title),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
    )


def _recall_query() -> RetrievalQuery:
    scope = _make_identity_scope()
    return RetrievalQuery(
        semantic_query="test",
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
    )


class TestRecallThresholdAndCandidateCount:
    """分数阈值只作用于 Dense 召回；融合不截取，Hybrid 按请求的候选数交出结果。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("config", "expected"),
        [(DenseRetrieverConfig(), 0.5), (DenseRetrieverConfig(score_threshold=0.42), 0.42)],
        ids=["default", "configured"],
    )
    async def test_dense_recall_filters_by_configured_cosine_threshold(self, config, expected):
        storage = AsyncMock()
        storage.search = AsyncMock(return_value=[])

        await create_retriever(storage, config).retrieve(_recall_query(), top_k=5)

        assert storage.search.call_args.kwargs["score_threshold"] == expected

    @pytest.mark.asyncio
    async def test_sparse_recall_passes_no_score_threshold(self):
        storage = AsyncMock()
        storage.search = AsyncMock(return_value=[])

        await create_retriever(storage, SparseRetrieverConfig()).retrieve(_recall_query(), top_k=5)

        assert "score_threshold" not in storage.search.call_args.kwargs

    @pytest.mark.asyncio
    async def test_hybrid_hands_out_requested_candidates_beyond_former_fusion_cut(self):
        """融合不再截到 5 条：引擎为重排请求 20 条时，两路共 12 条候选全部交出。"""
        by_mode = {
            "dense": [_recall_memory(f"D{i}") for i in range(6)],
            "sparse": [_recall_memory(f"S{i}") for i in range(6)],
        }

        async def search(*_args, mode, **_kwargs):
            return [{"memory": m, "score": 0.9 - i * 0.01} for i, m in enumerate(by_mode[mode])]

        storage = AsyncMock()
        storage.search = AsyncMock(side_effect=search)
        retriever = create_retriever(storage, HybridRetrieverConfig(enable_parallel=False))

        results = await retriever.retrieve(_recall_query(), top_k=20)

        assert {r.memory.id for r in results.results} == {
            m.id for memories in by_mode.values() for m in memories
        }
        thresholds = {
            call.kwargs["mode"]: call.kwargs.get("score_threshold")
            for call in storage.search.call_args_list
        }
        assert thresholds == {"dense": 0.5, "sparse": None}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_recall_programming_error_is_not_an_empty_result(mode):
    """两种召回器都不能把代码缺陷转换成无命中，避免上层继续执行错误的空上下文。"""
    from hivememory.engines.retrieval.retriever import SparseRetriever

    mid_term = AsyncMock()
    mid_term.search.side_effect = KeyError("invalid index")
    retriever = (
        DenseRetriever(mid_term=mid_term, config=DenseRetrieverConfig())
        if mode == "dense"
        else SparseRetriever(mid_term=mid_term, config=SparseRetrieverConfig())
    )
    scope = _make_identity_scope()
    query = RetrievalQuery(
        semantic_query="test",
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
    )

    with pytest.raises(KeyError, match="invalid index"):
        await retriever.retrieve(query)
