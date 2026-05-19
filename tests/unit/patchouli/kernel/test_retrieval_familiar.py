"""
RetrievalFamiliar 单元测试

测试覆盖:
- retrieve: 基础流程 / user_id 过滤 / MTP filter 合并 / mode 分支
- update_access_stats: 正常 / 单条失败隔离 / 空列表
"""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.retrieval.models import QueryFilters, SearchResult, SearchResults
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse


def _make_memory(title="测试记忆") -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(source_agent_id="a1", user_id="u1", session_id="s1"),
        index=IndexLayer(
            title=title,
            summary="这是一段足够长的测试摘要用于通过验证",
            tags=["t1"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="内容"),
    )


def _make_engine_result(memories=None, rendered="<ctx>", is_empty=False):
    """构建 mock engine.retrieve 返回值"""
    result = Mock()
    result.memories = memories or []
    result.memories_count = len(result.memories)
    result.latency_ms = 10.0
    result.rendered_context = rendered
    if is_empty:
        result.search_results = SearchResults(results=[])
    else:
        sr = [SearchResult(memory=m, score=0.9) for m in (memories or [])]
        result.search_results = SearchResults(results=sr)
    return result


def _make_request(query="测试查询", user_id="u1", filters=None):
    return RetrievalRequest(semantic_query=query, identity=Identity(user_id=user_id), filters=filters)

class TestRetrievalFamiliarRetrieve:
    """retrieve() 方法测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_engine = Mock()
        self.mock_passive_renderer = Mock()
        self.familiar = RetrievalFamiliar(
            storage=self.mock_storage,
            engine=self.mock_engine,
            passive_renderer=self.mock_passive_renderer,
        )

    def test_retrieve_basic(self):
        """基础检索流程"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])

        response = self.familiar.retrieve(_make_request())

        self.mock_engine.retrieve.assert_called_once()
        assert response.memories_count == 1

    def test_retrieve_user_id_filter(self):
        """user_id 作为安全基线过滤"""
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(_make_request(user_id="user_abc"))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.user_id == "user_abc"

    def test_retrieve_no_mtp_filters(self):
        """request.filters=None 时只有 user_id 过滤"""
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(_make_request(filters=None))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.user_id == "u1"
        assert query.filters.memory_type is None

    def test_retrieve_with_memory_type_filter(self):
        """合并 memory_type 过滤"""
        filters = QueryFilters(memory_type="CODE_SNIPPET")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.memory_type == "CODE_SNIPPET"

    def test_retrieve_with_tags_filter(self):
        """合并 tags 过滤"""
        filters = QueryFilters(tags=["python", "docker"])
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.tags == ["python", "docker"]

    def test_retrieve_with_min_confidence_filter(self):
        """min_confidence > 0 时合并"""
        filters = QueryFilters(min_confidence=0.8)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.min_confidence == 0.8

    def test_retrieve_min_confidence_zero_ignored(self):
        """min_confidence=0 不合并"""
        filters = QueryFilters(min_confidence=0)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        # 默认值应保持不变
        assert query.filters.min_confidence == 0

    def test_retrieve_active_mode_uses_engine_context(self):
        """active 模式使用 engine 默认渲染"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [mem], rendered="engine_rendered"
        )

        response = self.familiar.retrieve(_make_request(), mode="active")

        assert response.rendered_context == "engine_rendered"
        self.mock_passive_renderer.render.assert_not_called()

    def test_retrieve_passive_mode_uses_passive_renderer(self):
        """passive 模式使用 passive_renderer"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        self.mock_passive_renderer.render.return_value = "passive_rendered"

        response = self.familiar.retrieve(_make_request(), mode="passive")

        self.mock_passive_renderer.render.assert_called_once()
        assert response.rendered_context == "passive_rendered"

    def test_retrieve_passive_no_renderer_fallback(self):
        """passive 模式无 renderer 时 fallback 到 engine 渲染"""
        familiar = RetrievalFamiliar(
            storage=self.mock_storage,
            engine=self.mock_engine,
            passive_renderer=None,
        )
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [mem], rendered="engine_ctx"
        )

        response = familiar.retrieve(_make_request(), mode="passive")

        assert response.rendered_context == "engine_ctx"

    def test_retrieve_passive_empty_results_fallback(self):
        """passive 模式空结果时 fallback"""
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [], rendered="", is_empty=True
        )

        response = self.familiar.retrieve(_make_request(), mode="passive")

        self.mock_passive_renderer.render.assert_not_called()

    def test_retrieve_exception_returns_empty(self):
        """engine 抛异常时返回空 response"""
        self.mock_engine.retrieve.side_effect = RuntimeError("engine error")

        response = self.familiar.retrieve(_make_request())

        assert response.memories_count == 0
        assert response.latency_ms >= 0


class TestRetrievalFamiliarIdentityPropagation:
    """§3.3 identity 完整传播测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_engine = Mock()
        self.familiar = RetrievalFamiliar(
            storage=self.mock_storage,
            engine=self.mock_engine,
        )

    def _get_query_filters(self) -> QueryFilters:
        return self.mock_engine.retrieve.call_args[1]["query"].filters

    def test_identity_propagated_to_engine(self):
        """完整 identity 对象传播到 engine 的 QueryFilters"""
        identity = Identity(user_id="u1", agent_id="coder_doll", team_id="team_a")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(RetrievalRequest(
            semantic_query="test", identity=identity,
        ))

        qf = self._get_query_filters()
        assert qf.identity.user_id == "u1"
        assert qf.identity.agent_id == "coder_doll"
        assert qf.identity.team_id == "team_a"

    def test_team_id_none_propagated(self):
        """team_id=None 时 identity 仍正确传播"""
        identity = Identity(user_id="u1", agent_id="default", team_id=None)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(RetrievalRequest(
            semantic_query="test", identity=identity,
        ))

        qf = self._get_query_filters()
        assert qf.identity.team_id is None

    def test_mtp_filter_cannot_override_identity(self):
        """MTP filter 不能覆盖 identity 安全基线"""
        identity = Identity(user_id="u1", agent_id="coder_doll")
        mtp_filters = QueryFilters(
            identity=Identity(user_id="hacker", agent_id="evil"),
            memory_type=MemoryType.CODE_SNIPPET,
        )
        self.mock_engine.retrieve.return_value = _make_engine_result()

        self.familiar.retrieve(RetrievalRequest(
            semantic_query="test", identity=identity, filters=mtp_filters,
        ))

        qf = self._get_query_filters()
        # identity 来自 request，不被 mtp_filters 覆盖
        assert qf.identity.user_id == "u1"
        assert qf.identity.agent_id == "coder_doll"
        # 但 memory_type 被合并
        assert qf.memory_type == MemoryType.CODE_SNIPPET


class TestRetrievalFamiliarAccessStats:
    """update_access_stats() 测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.familiar = RetrievalFamiliar(
            storage=self.mock_storage,
            engine=Mock(),
        )

    def test_update_access_stats(self):
        """逐条调用 storage.update_access_info"""
        m1 = _make_memory("m1")
        m2 = _make_memory("m2")

        self.familiar.update_access_stats([m1, m2])

        assert self.mock_storage.update_access_info.call_count == 2

    def test_update_access_stats_per_item_failure(self):
        """单条失败不影响其他"""
        m1 = _make_memory("m1")
        m2 = _make_memory("m2")
        self.mock_storage.update_access_info.side_effect = [
            RuntimeError("fail"), None
        ]

        self.familiar.update_access_stats([m1, m2])

        assert self.mock_storage.update_access_info.call_count == 2

    def test_update_access_stats_empty_list(self):
        """空列表不报错"""
        self.familiar.update_access_stats([])
        self.mock_storage.update_access_info.assert_not_called()
