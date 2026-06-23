"""
RetrievalFamiliar 单元测试

测试覆盖:
- retrieve: 基础流程 / user_id 过滤 / MTP filter 合并 / mode 分支
- update_access_stats: 正常 / 单条失败隔离 / 空列表
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from uuid import uuid4

from hivememory.core.models import Identity, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.retrieval.models import QueryFilters, SearchResult, SearchResults
from hivememory.engines.perception.models import LogicalBlock
from hivememory.patchouli.memory_library.models import TopicData
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
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


def _make_memory_library():
    library = Mock()
    library.short_term = Mock()
    library.mid_term = Mock()
    library.long_term = Mock()
    library.mid_term.get_by_alias = AsyncMock()
    library.mid_term.update_access_info = AsyncMock()
    library.long_term.query = AsyncMock()
    library.long_term.is_archived = AsyncMock()
    return library


def _make_topic_data(
    topic_id="topic_1",
    user_id="u1",
    blocks=None,
    last_accessed_at=1.0,
):
    return TopicData(
        topic_id=topic_id,
        user_id=user_id,
        topic_title=f"title-{topic_id}",
        topic_summary=f"summary-{topic_id}",
        state_summary=f"state-{topic_id}",
        blocks=tuple(blocks or []),
        last_update=last_accessed_at,
        last_accessed_at=last_accessed_at,
        total_tokens=10,
    )


class TestRetrievalFamiliarRetrieve:
    """retrieve() 方法测试"""

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.mock_engine.retrieve = AsyncMock()
        self.mock_passive_renderer = Mock()
        self.familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            passive_renderer=self.mock_passive_renderer,
        )

    @pytest.mark.asyncio
    async def test_retrieve_basic(self):
        """基础检索流程"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])

        response = await self.familiar.retrieve(_make_request())

        self.mock_engine.retrieve.assert_awaited_once()
        assert response.memories_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_user_id_filter(self):
        """user_id 作为安全基线过滤"""
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(user_id="user_abc"))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.user_id == "user_abc"

    @pytest.mark.asyncio
    async def test_retrieve_no_mtp_filters(self):
        """request.filters=None 时只有 user_id 过滤"""
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=None))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.user_id == "u1"
        assert query.filters.memory_type is None

    @pytest.mark.asyncio
    async def test_retrieve_with_memory_type_filter(self):
        """合并 memory_type 过滤"""
        filters = QueryFilters(memory_type="CODE_SNIPPET")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.memory_type == "CODE_SNIPPET"

    @pytest.mark.asyncio
    async def test_retrieve_with_tags_filter(self):
        """合并 tags 过滤"""
        filters = QueryFilters(tags=["python", "docker"])
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.tags == ["python", "docker"]

    @pytest.mark.asyncio
    async def test_retrieve_with_min_confidence_filter(self):
        """min_confidence > 0 时合并"""
        filters = QueryFilters(min_confidence=0.8)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        assert query.filters.min_confidence == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_min_confidence_zero_ignored(self):
        """min_confidence=0 不合并"""
        filters = QueryFilters(min_confidence=0)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        call_args = self.mock_engine.retrieve.call_args
        query = call_args[1]["query"]
        # 默认值应保持不变
        assert query.filters.min_confidence == 0

    @pytest.mark.asyncio
    async def test_retrieve_active_mode_uses_engine_context(self):
        """active 模式使用 engine 默认渲染"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [mem], rendered="engine_rendered"
        )
        self.mock_engine.renderer.render.return_value = "engine_rendered"

        response = await self.familiar.retrieve(_make_request(), mode="active")

        assert response.rendered_context == "engine_rendered"
        self.mock_passive_renderer.render.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_async_refreshes_vitality_through_local_bus_before_render(self):
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        self.mock_engine.renderer.render.return_value = "stale_rendered"
        self.mock_engine.render_memories.return_value = "fresh_rendered"
        bus = PatchouliBus()

        async def _refresh(memories, persist=False):
            memories[0].meta.vitality_score = 42.0
            return [(memories[0].id, 42.0)]

        refresh = AsyncMock(side_effect=_refresh)
        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, refresh)
        self.familiar._local_bus = bus

        response = await self.familiar.retrieve_async(_make_request(), mode="active")

        refresh.assert_awaited_once_with([mem], persist=False)
        self.mock_engine.render_memories.assert_called_once_with([mem])
        assert response.memories[0].meta.vitality_score == 42.0
        assert response.rendered_context == "fresh_rendered"

    @pytest.mark.asyncio
    async def test_retrieve_async_vitality_refresh_failure_keeps_response(self):
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [mem], rendered="engine_rendered"
        )
        self.mock_engine.renderer.render.return_value = "engine_rendered"
        self.mock_engine.render_memories.return_value = "rerendered"
        bus = PatchouliBus()
        refresh = AsyncMock(side_effect=RuntimeError("refresh failed"))
        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, refresh)
        self.familiar._local_bus = bus

        response = await self.familiar.retrieve_async(_make_request(), mode="active")

        refresh.assert_awaited_once_with([mem], persist=False)
        assert response.memories == [mem]
        assert response.rendered_context == "rerendered"

    @pytest.mark.asyncio
    async def test_retrieve_passive_mode_uses_passive_renderer(self):
        """passive 模式使用 passive_renderer"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        self.mock_passive_renderer.render.return_value = "passive_rendered"

        response = await self.familiar.retrieve(_make_request(), mode="passive")

        self.mock_passive_renderer.render.assert_called_once()
        assert response.rendered_context == "passive_rendered"

    @pytest.mark.asyncio
    async def test_retrieve_passive_no_renderer_fallback(self):
        """passive 模式无 renderer 时 fallback 到 engine 渲染"""
        familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            passive_renderer=None,
        )
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [mem], rendered="engine_ctx"
        )
        self.mock_engine.renderer.render.return_value = "engine_ctx"

        response = await familiar.retrieve(_make_request(), mode="passive")

        assert response.rendered_context == "engine_ctx"

    @pytest.mark.asyncio
    async def test_retrieve_passive_empty_results_fallback(self):
        """passive 模式空结果时 fallback"""
        self.mock_engine.retrieve.return_value = _make_engine_result(
            [], rendered="", is_empty=True
        )

        response = await self.familiar.retrieve(_make_request(), mode="passive")

        self.mock_passive_renderer.render.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_exception_returns_empty(self):
        """engine 抛异常时返回空 response"""
        self.mock_engine.retrieve.side_effect = RuntimeError("engine error")

        response = await self.familiar.retrieve(_make_request())

        assert response.memories_count == 0
        assert response.latency_ms >= 0


class TestRetrievalFamiliarIdentityPropagation:
    """§3.3 identity 完整传播测试"""

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = AsyncMock()
        self.familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
        )

    def _get_query_filters(self) -> QueryFilters:
        return self.mock_engine.retrieve.call_args[1]["query"].filters

    @pytest.mark.asyncio
    async def test_identity_propagated_to_engine(self):
        """完整 identity 对象传播到 engine 的 QueryFilters"""
        identity = Identity(user_id="u1", agent_id="coder_doll", team_id="team_a")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(RetrievalRequest(
            semantic_query="test", identity=identity,
        ))

        qf = self._get_query_filters()
        assert qf.identity.user_id == "u1"
        assert qf.identity.agent_id == "coder_doll"
        assert qf.identity.team_id == "team_a"

    @pytest.mark.asyncio
    async def test_team_id_none_propagated(self):
        """team_id=None 时 identity 仍正确传播"""
        identity = Identity(user_id="u1", agent_id="default", team_id=None)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(RetrievalRequest(
            semantic_query="test", identity=identity,
        ))

        qf = self._get_query_filters()
        assert qf.identity.team_id is None

    @pytest.mark.asyncio
    async def test_mtp_filter_cannot_override_identity(self):
        """MTP filter 不能覆盖 identity 安全基线"""
        identity = Identity(user_id="u1", agent_id="coder_doll")
        mtp_filters = QueryFilters(
            identity=Identity(user_id="hacker", agent_id="evil"),
            memory_type=MemoryType.CODE_SNIPPET,
        )
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(RetrievalRequest(
            semantic_query="test", identity=identity, filters=mtp_filters,
        ))

        qf = self._get_query_filters()
        # identity 来自 request，不被 mtp_filters 覆盖
        assert qf.identity.user_id == "u1"
        assert qf.identity.agent_id == "coder_doll"
        # 但 memory_type 被合并
        assert qf.memory_type == MemoryType.CODE_SNIPPET


class TestRetrievalFamiliarRetrieveByAliases:
    """retrieve_by_aliases() 精确取回并复用统一 renderer。"""

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.mock_engine.retrieve = AsyncMock()
        self.mock_engine.retrieve_by_aliases = AsyncMock()
        self.mock_passive_renderer = Mock()
        self.familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            passive_renderer=self.mock_passive_renderer,
        )

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_renders_with_engine(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.return_value = mem
        self.mock_engine.render_memories.return_value = "rendered aliases"

        response = await self.familiar.retrieve_by_aliases(
            aliases=["fact_a"],
            identity=Identity(user_id="u1"),
        )

        self.mock_library.mid_term.get_by_alias.assert_awaited_once_with("fact_a", "u1")
        self.mock_engine.render_memories.assert_called_once_with([mem])
        assert response.memories == [mem]
        assert response.memories_count == 1
        assert response.rendered_context == "rendered aliases"

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_async_refreshes_vitality_through_local_bus(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.return_value = mem
        self.mock_engine.render_memories.side_effect = ["stale", "fresh aliases"]
        bus = PatchouliBus()
        refresh = AsyncMock(return_value=[(mem.id, 41.0)])
        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, refresh)
        self.familiar._local_bus = bus

        response = await self.familiar.retrieve_by_aliases_async(
            aliases=["fact_a"],
            identity=Identity(user_id="u1"),
        )

        refresh.assert_awaited_once_with([mem], persist=False)
        assert self.mock_engine.render_memories.call_count == 2
        assert response.rendered_context == "fresh aliases"

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_deduplicates_and_skips_missing(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.side_effect = [mem, None]
        self.mock_engine.render_memories.return_value = "rendered"

        response = await self.familiar.retrieve_by_aliases(
            aliases=["fact_a", "fact_a", "", "fact_missing"],
            identity=Identity(user_id="u1"),
        )

        assert self.mock_library.mid_term.get_by_alias.call_count == 2
        self.mock_engine.render_memories.assert_called_once_with([mem])
        assert response.memories == [mem]
        assert response.rendered_context == "rendered"

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_passive_uses_passive_renderer(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.return_value = mem
        self.mock_passive_renderer.render.return_value = "passive rendered"

        response = await self.familiar.retrieve_by_aliases(
            aliases=["fact_a"],
            identity=Identity(user_id="u1"),
            mode="passive",
        )

        self.mock_passive_renderer.render.assert_called_once_with([mem])
        self.mock_engine.render_memories.assert_not_called()
        assert response.rendered_context == "passive rendered"


class TestRetrievalFamiliarAccessStats:
    """update_access_stats() 测试"""

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
        )

    @pytest.mark.asyncio
    async def test_update_access_stats(self):
        """逐条调用 mid_term.update_access_info"""
        m1 = _make_memory("m1")
        m2 = _make_memory("m2")

        await self.familiar.update_access_stats([m1, m2])

        assert self.mock_library.mid_term.update_access_info.call_count == 2

    @pytest.mark.asyncio
    async def test_update_access_stats_per_item_failure(self):
        """单条失败不影响其他"""
        m1 = _make_memory("m1")
        m2 = _make_memory("m2")
        self.mock_library.mid_term.update_access_info.side_effect = [
            RuntimeError("fail"), None
        ]

        await self.familiar.update_access_stats([m1, m2])

        assert self.mock_library.mid_term.update_access_info.call_count == 2

    @pytest.mark.asyncio
    async def test_update_access_stats_empty_list(self):
        """空列表不报错"""
        await self.familiar.update_access_stats([])
        self.mock_library.mid_term.update_access_info.assert_not_called()


class TestRetrievalFamiliarShortTermTopics:
    """短期话题读入口测试"""

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
        )

    def test_list_active_topics_excludes_empty_and_sorts_by_access(self):
        block = LogicalBlock()
        old_topic = _make_topic_data("old", blocks=[block], last_accessed_at=1.0)
        empty_topic = _make_topic_data("empty", blocks=[], last_accessed_at=3.0)
        new_topic = _make_topic_data("new", blocks=[block], last_accessed_at=2.0)
        self.mock_library.short_term.list_topic_data.return_value = [
            old_topic,
            empty_topic,
            new_topic,
        ]

        snapshots = self.familiar.list_active_topics(Identity(user_id="u1"))

        self.mock_library.short_term.list_topic_data.assert_called_once_with(
            user_id="u1",
            include_empty=False,
            deep_copy=False,
        )
        assert [snapshot.topic_id for snapshot in snapshots] == ["new", "old"]
        assert snapshots[0].block_count == 1

    def test_list_active_topics_include_empty_passthrough(self):
        self.mock_library.short_term.list_topic_data.return_value = []

        self.familiar.list_active_topics(Identity(user_id="u1"), include_empty=True)

        self.mock_library.short_term.list_topic_data.assert_called_once_with(
            user_id="u1",
            include_empty=True,
            deep_copy=False,
        )

    def test_get_topic_defaults_to_deep_copy(self):
        topic_data = _make_topic_data()
        self.mock_library.short_term.get_topic_data.return_value = topic_data

        result = self.familiar.get_topic("topic_1")

        self.mock_library.short_term.get_topic_data.assert_called_once_with(
            "topic_1",
            touch=True,
            deep_copy=True,
        )
        assert result is topic_data

    def test_get_topic_can_skip_deep_copy(self):
        self.familiar.get_topic("topic_1", touch=False, deep_copy=False)

        self.mock_library.short_term.get_topic_data.assert_called_once_with(
            "topic_1",
            touch=False,
            deep_copy=False,
        )
