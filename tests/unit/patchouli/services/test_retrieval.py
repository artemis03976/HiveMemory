"""
RetrievalFamiliar 单元测试 (Phase C — renderer 解耦后)

测试覆盖:
- retrieve: 基础流程 / user_id 过滤 / MTP filter 合并
- retrieve_by_aliases: 精确取回 / 去重 / 跳过缺失
- update_access_stats, archive queries, topics
"""

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TopicData,
    TopicFold,
)
from hivememory.core.models.query import QueryFilters
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    SearchResult,
    SearchResults,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.memory import make_memory_metadata
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope


def _make_memory(title="测试记忆") -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary="这是一段足够长的测试摘要用于通过验证",
            tags=["t1"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="内容"),
    )


def _make_engine_result(memories=None, is_empty=False):
    result = Mock()
    result.memories = memories or []
    result.memories_count = len(result.memories)
    result.latency_ms = 10.0
    if is_empty:
        result.search_results = SearchResults(results=[])
    else:
        sr = [SearchResult(memory=m, score=0.9) for m in (memories or [])]
        result.search_results = SearchResults(results=sr)
    return result


def _make_request(query="测试查询", user_id="u1", filters=None):
    return RetrievalQuery(
        semantic_query=query,
        belong_to=make_identity_scope(user_id=user_id).workspace_identity,
        from_actor=make_identity_scope(user_id=user_id).actor_identity,
        filters=filters or QueryFilters(),
    )


def _make_memory_library():
    library = Mock()
    library.short_term = Mock()
    library.mid_term = Mock()
    library.long_term = Mock()
    library.mid_term.get_by_alias = AsyncMock()
    library.mid_term.patch_payload = AsyncMock()
    library.long_term.query = AsyncMock()
    library.long_term.is_archived = AsyncMock()
    return library


def _make_topic_data(topic_id="topic_1", user_id="u1", interactions=(), folds=(), last_update=1.0):
    identity_scope = make_identity_scope(user_id=user_id)
    return TopicData(
        topic_id=topic_id,
        workspace_identity=identity_scope.workspace_identity,
        topic_title=f"title-{topic_id}",
        topic_summary=f"summary-{topic_id}",
        interactions=tuple(interactions),
        folds=tuple(folds),
        last_update=last_update,
    )


class TestRetrievalFamiliarRetrieve:
    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.mock_engine.retrieve = AsyncMock()
        self.familiar = RetrievalFamiliar(engine=self.mock_engine, memory_library=self.mock_library)

    @pytest.mark.asyncio
    async def test_retrieve_propagates_identity_scope(self):
        self.mock_engine.retrieve.return_value = _make_engine_result()

        request = _make_request(user_id="user_abc")
        await self.familiar.retrieve(request)

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.belong_to == request.belong_to

    @pytest.mark.asyncio
    async def test_retrieve_no_mtp_filters(self):
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=None))

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.filters.is_empty()
        assert query.filters.memory_type is None

    @pytest.mark.asyncio
    async def test_retrieve_with_memory_type_filter(self):
        filters = QueryFilters(memory_type="CODE_SNIPPET")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.filters.memory_type == "CODE_SNIPPET"

    @pytest.mark.asyncio
    async def test_retrieve_with_tags_filter(self):
        filters = QueryFilters(tags=["python", "docker"])
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.filters.tags == ["python", "docker"]

    @pytest.mark.asyncio
    async def test_retrieve_with_min_confidence_filter(self):
        filters = QueryFilters(min_confidence=0.8)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.filters.min_confidence == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_min_confidence_zero_ignored(self):
        filters = QueryFilters(min_confidence=0)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(_make_request(filters=filters))

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.filters.min_confidence == 0

    @pytest.mark.asyncio
    async def test_retrieve_does_not_compile_context(self):
        """检索服务直接返回引擎排序后的完整原子列表，不产出 Agent 可读文本"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])

        memories = await self.familiar.retrieve(_make_request())

        assert memories == [mem]

    @pytest.mark.asyncio
    async def test_retrieve_async_refreshes_vitality_through_local_bus(self):
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        bus = PatchouliBus()

        async def _refresh(memories, persist=False):
            memories[0].meta.lifecycle.vitality_score = 42.0
            return [(memories[0].id, 42.0)]

        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, AsyncMock(side_effect=_refresh))
        familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            local_bus=bus,
        )

        memories = await familiar.retrieve_async(_make_request())

        assert memories[0].meta.lifecycle.vitality_score == 42.0

    @pytest.mark.asyncio
    async def test_retrieve_async_vitality_refresh_failure_keeps_response(self):
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        bus = PatchouliBus()
        bus.register(
            PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
            AsyncMock(side_effect=RuntimeError("fail")),
        )
        familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            local_bus=bus,
        )

        memories = await familiar.retrieve_async(_make_request())

        assert memories == [mem]

    @pytest.mark.asyncio
    async def test_retrieve_engine_failure_propagates_instead_of_empty_list(self):
        """引擎异常按原错误传播，不伪装为"没有检索到记忆"。"""
        self.mock_engine.retrieve.side_effect = RuntimeError("engine error")

        with pytest.raises(RuntimeError, match="engine error"):
            await self.familiar.retrieve(_make_request())


class TestRetrievalFamiliarIdentityPropagation:
    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = AsyncMock()
        self.familiar = RetrievalFamiliar(engine=self.mock_engine, memory_library=self.mock_library)

    def _get_query(self):
        return self.mock_engine.retrieve.call_args[1]["query"]

    @pytest.mark.asyncio
    async def test_identity_propagated_to_engine(self):
        identity = ActorIdentity(user_id="u1", agent_id="coder_doll", team_id="team_a")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(
            RetrievalQuery(
                semantic_query="test",
                belong_to=make_identity_scope(actor_identity=identity).workspace_identity,
                from_actor=identity,
            )
        )

        query = self._get_query()
        assert query.from_actor == identity

    @pytest.mark.asyncio
    async def test_team_id_none_propagated(self):
        identity = ActorIdentity(user_id="u1", agent_id="default", team_id=None)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(
            RetrievalQuery(
                semantic_query="test",
                belong_to=make_identity_scope(actor_identity=identity).workspace_identity,
                from_actor=identity,
            )
        )

        assert self._get_query().from_actor.team_id is None

    @pytest.mark.asyncio
    async def test_mtp_filter_cannot_carry_identity(self):
        """防止业务过滤条件重新引入第二套授权身份。"""
        with pytest.raises(ValidationError, match="identity"):
            QueryFilters.model_validate(
                {
                    "identity": {"user_id": "hacker", "agent_id": "evil"},
                    "memory_type": MemoryType.CODE_SNIPPET,
                }
            )


class TestRetrievalFamiliarRetrieveByAliases:
    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.familiar = RetrievalFamiliar(engine=self.mock_engine, memory_library=self.mock_library)

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_async_refreshes_vitality(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.return_value = mem
        bus = PatchouliBus()
        refresh = AsyncMock(return_value=[(mem.id, 41.0)])
        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, refresh)
        familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            local_bus=bus,
        )

        response = await familiar.retrieve_by_aliases_async(
            aliases=["fact_a"],
            belong_to=make_identity_scope(user_id="u1").workspace_identity,
            from_actor=make_identity_scope(user_id="u1").actor_identity,
        )

        refresh.assert_awaited_once_with([mem], persist=False)
        assert response == [mem]

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_deduplicates_and_skips_missing(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.side_effect = [mem, None]

        response = await self.familiar.retrieve_by_aliases(
            aliases=["fact_a", "fact_a", "", "fact_missing"],
            belong_to=make_identity_scope(user_id="u1").workspace_identity,
            from_actor=make_identity_scope(user_id="u1").actor_identity,
        )

        assert self.mock_library.mid_term.get_by_alias.call_count == 2
        assert response == [mem]


class TestRetrievalFamiliarAccessStats:
    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.familiar = RetrievalFamiliar(engine=Mock(), memory_library=self.mock_library)

    @pytest.mark.asyncio
    async def test_update_access_stats_per_item_failure(self):
        m1, m2 = _make_memory("m1"), _make_memory("m2")
        self.mock_library.mid_term.patch_payload.side_effect = [RuntimeError("fail"), None]
        await self.familiar.update_access_stats(
            make_identity_scope(user_id="u1").workspace_identity, [m1, m2]
        )
        # 逐条 patch_payload，单条失败 warning 吞掉，不中断后续条目
        assert self.mock_library.mid_term.patch_payload.call_count == 2
        first_call = self.mock_library.mid_term.patch_payload.call_args_list[0]
        key, patch = first_call.args
        assert key.memory_id == m1.id
        assert key.workspace_identity == make_identity_scope(user_id="u1").workspace_identity
        assert set(patch) == {
            "meta.lifecycle.access_count",
            "meta.lifecycle.last_accessed_at",
        }
        assert patch["meta.lifecycle.access_count"] == m1.meta.lifecycle.access_count + 1

    @pytest.mark.asyncio
    async def test_update_access_stats_empty_list(self):
        await self.familiar.update_access_stats(
            make_identity_scope(user_id="u1").workspace_identity, []
        )
        self.mock_library.mid_term.patch_payload.assert_not_called()


class TestRetrievalFamiliarShortTermTopics:
    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.familiar = RetrievalFamiliar(engine=Mock(), memory_library=self.mock_library)

    def test_list_active_topics_excludes_empty_and_sorts_by_recency(self):
        block = make_logical_block()
        old_topic = _make_topic_data("old", interactions=[block], last_update=1.0)
        empty_topic = _make_topic_data("empty", last_update=3.0)
        folded = make_logical_block()
        fully_folded = _make_topic_data(
            "fully-folded",
            interactions=[folded],
            folds=[TopicFold(folded_block_ids=(folded.block_id,), summary="s", folded_at=4.0)],
            last_update=4.0,
        )
        new_topic = _make_topic_data("new", interactions=[block], last_update=2.0)
        self.mock_library.short_term.list_by_workspace.return_value = [
            old_topic,
            empty_topic,
            fully_folded,
            new_topic,
        ]

        identity_scope = make_identity_scope(user_id="u1")
        snapshots = self.familiar.list_active_topics(belong_to=identity_scope.workspace_identity)

        self.mock_library.short_term.list_by_workspace.assert_called_once_with(
            identity_scope.workspace_identity, include_empty=False
        )
        # 没有接收过交互的 Topic 被排除；交互全部折叠的 Topic 与有 blocks 的 Topic 保留，
        # 按最近写入排序，快照携带排序所依据的写入时间供话题池显示
        assert [(s.topic_id, s.last_update) for s in snapshots] == [
            ("fully-folded", 4.0),
            ("new", 2.0),
            ("old", 1.0),
        ]
