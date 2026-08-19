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

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    Artifacts,
    Identity,
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
    TopicData,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
    StorageReadError,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.engines.retrieval.models import QueryFilters, SearchResult, SearchResults
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.workspace import make_access_context
from tests.helpers.memory import make_memory_metadata


def _make_memory(title="测试记忆") -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1", session_id="s1"),
        index=IndexLayer(title=title, summary="这是一段足够长的测试摘要用于通过验证", tags=["t1"], memory_type=MemoryType.FACT),
        payload=PayloadLayer(content="内容"),
    )


def _make_profile_memory(
    *,
    alias: str = "coder_doll",
    user_id: str = "u1",
    source_agent_id: str = "omni_doll",
    team_id: str | None = None,
    visibility: MemoryVisibility = MemoryVisibility.PUBLIC,
    agent_config: dict | None = None,
) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(
            source_agent_id=source_agent_id,
            user_id=user_id,
            team_id=team_id,
            visibility=visibility,
        ),
        index=IndexLayer(
            title="Coder Doll",
            summary="A specialized coding agent profile",
            tags=["agent"],
            memory_type=MemoryType.AGENT_PROFILE,
            alias=alias,
        ),
        payload=PayloadLayer(
            content="You are a coding specialist.",
            artifacts=Artifacts(agent_config=agent_config or {"model_name": "default"}),
        ),
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
    return RetrievalRequest(
        semantic_query=query,
        access_context=make_access_context(user_id=user_id),
        filters=filters,
    )


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


def _make_topic_data(topic_id="topic_1", user_id="u1", blocks=None, last_accessed_at=1.0):
    access_context = make_access_context(user_id=user_id)
    return TopicData(
        topic_id=topic_id, workspace_identity=access_context.workspace_identity,
        topic_title=f"title-{topic_id}", topic_summary=f"summary-{topic_id}",
        state_summary=f"state-{topic_id}",
        blocks=tuple(blocks or []),
        last_update=last_accessed_at, last_accessed_at=last_accessed_at, total_tokens=10,
    )


class TestRetrievalFamiliarAgentProfiles:

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.familiar = RetrievalFamiliar(
            engine=Mock(),
            memory_library=self.mock_library,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias", [None, "", "   ", "default", "omni_doll"])
    async def test_unspecified_or_builtin_profile_uses_explicit_fallback(self, alias):
        result = await self.familiar.get_agent_profile(
            alias,
            access_context=make_access_context(),
        )

        assert result is OMNI_DOLL_PROFILE
        self.mock_library.mid_term.get_by_alias.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_visible_profile_loads_for_identity(self):
        atom = _make_profile_memory()
        self.mock_library.mid_term.get_by_alias.return_value = atom
        identity = Identity(user_id="u1", agent_id="omni_doll")

        result = await self.familiar.get_agent_profile(
            "coder_doll",
            access_context=make_access_context(actor_identity=identity),
        )

        assert result.persona == "You are a coding specialist."
        self.mock_library.mid_term.get_by_alias.assert_awaited_once_with(
            make_access_context(actor_identity=identity),
            "coder_doll",
        )

    @pytest.mark.asyncio
    async def test_explicit_missing_profile_does_not_fallback(self):
        self.mock_library.mid_term.get_by_alias.return_value = None

        with pytest.raises(AliasNotFoundError) as exc_info:
            await self.familiar.get_agent_profile(
                "missing_doll",
                access_context=make_access_context(user_id="u1"),
            )

        assert exc_info.value.message_key == "mtp.call.profile_not_found"

    @pytest.mark.asyncio
    async def test_custom_profile_without_scope_is_denied_before_storage(self):
        """防止 profile 资源读取在缺 scope 时访问存储。"""
        with pytest.raises(ScopeRequiredError):
            await self.familiar.get_agent_profile(
                "private_doll",
                access_context=None,  # type: ignore[arg-type]
            )

        self.mock_library.mid_term.get_by_alias.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_profile_alias_is_type_mismatch(self):
        self.mock_library.mid_term.get_by_alias.return_value = _make_memory()

        with pytest.raises(MemoryTypeMismatchError) as exc_info:
            await self.familiar.get_agent_profile(
                "fact_alias",
                access_context=make_access_context(user_id="u1"),
            )

        assert exc_info.value.message_key == "mtp.call.profile_type_mismatch"

    @pytest.mark.asyncio
    async def test_invalid_profile_config_is_rejected(self):
        self.mock_library.mid_term.get_by_alias.return_value = _make_profile_memory(
            agent_config={"top_p": 2.0},
        )

        with pytest.raises(InvalidArgumentError) as exc_info:
            await self.familiar.get_agent_profile(
                "broken_doll",
                access_context=make_access_context(user_id="u1"),
            )

        assert exc_info.value.message_key == "mtp.call.profile_invalid"

    @pytest.mark.asyncio
    async def test_storage_failure_is_not_converted_to_fallback(self):
        failure = StorageReadError(cause=RuntimeError("storage failure"))
        self.mock_library.mid_term.get_by_alias.side_effect = failure

        with pytest.raises(StorageReadError) as exc_info:
            await self.familiar.get_agent_profile(
                "coder_doll",
                access_context=make_access_context(user_id="u1"),
            )

        assert exc_info.value is failure


class TestRetrievalFamiliarRetrieve:

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = Mock()
        self.mock_engine.retrieve = AsyncMock()
        self.familiar = RetrievalFamiliar(engine=self.mock_engine, memory_library=self.mock_library)

    @pytest.mark.asyncio
    async def test_retrieve_propagates_workspace_access_context(self):
        self.mock_engine.retrieve.return_value = _make_engine_result()

        request = _make_request(user_id="user_abc")
        await self.familiar.retrieve(request)

        query = self.mock_engine.retrieve.call_args[1]["query"]
        assert query.access_context == request.access_context

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
        """检索服务只返回记忆原子，不产出 Agent 可读文本"""
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])

        response = await self.familiar.retrieve(_make_request())

        assert not hasattr(response, "rendered_context")

    @pytest.mark.asyncio
    async def test_retrieve_async_refreshes_vitality_through_local_bus(self):
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        bus = PatchouliBus()

        async def _refresh(memories, persist=False):
            memories[0].meta.vitality_score = 42.0
            return [(memories[0].id, 42.0)]

        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, AsyncMock(side_effect=_refresh))
        familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            local_bus=bus,
        )

        response = await familiar.retrieve_async(_make_request())

        assert response.memories[0].meta.vitality_score == 42.0

    @pytest.mark.asyncio
    async def test_retrieve_async_vitality_refresh_failure_keeps_response(self):
        mem = _make_memory()
        self.mock_engine.retrieve.return_value = _make_engine_result([mem])
        bus = PatchouliBus()
        bus.register(PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, AsyncMock(side_effect=RuntimeError("fail")))
        familiar = RetrievalFamiliar(
            engine=self.mock_engine,
            memory_library=self.mock_library,
            local_bus=bus,
        )

        response = await familiar.retrieve_async(_make_request())

        assert response.memories == [mem]

    @pytest.mark.asyncio
    async def test_retrieve_exception_returns_empty(self):
        self.mock_engine.retrieve.side_effect = RuntimeError("engine error")

        response = await self.familiar.retrieve(_make_request())

        assert response.memories_count == 0
        assert response.latency_ms >= 0


class TestRetrievalFamiliarIdentityPropagation:

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.mock_engine = AsyncMock()
        self.familiar = RetrievalFamiliar(engine=self.mock_engine, memory_library=self.mock_library)

    def _get_query(self):
        return self.mock_engine.retrieve.call_args[1]["query"]

    @pytest.mark.asyncio
    async def test_identity_propagated_to_engine(self):
        identity = Identity(user_id="u1", agent_id="coder_doll", team_id="team_a")
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(
            RetrievalRequest(
                semantic_query="test",
                access_context=make_access_context(actor_identity=identity),
            )
        )

        query = self._get_query()
        assert query.access_context.actor_identity == identity

    @pytest.mark.asyncio
    async def test_team_id_none_propagated(self):
        identity = Identity(user_id="u1", agent_id="default", team_id=None)
        self.mock_engine.retrieve.return_value = _make_engine_result()

        await self.familiar.retrieve(
            RetrievalRequest(
                semantic_query="test",
                access_context=make_access_context(actor_identity=identity),
            )
        )

        assert self._get_query().access_context.actor_identity.team_id is None

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
            access_context=make_access_context(user_id="u1"),
        )

        refresh.assert_awaited_once_with([mem], persist=False)
        assert response.memories == [mem]

    @pytest.mark.asyncio
    async def test_retrieve_by_aliases_deduplicates_and_skips_missing(self):
        mem = _make_memory("alias memory")
        self.mock_library.mid_term.get_by_alias.side_effect = [mem, None]

        response = await self.familiar.retrieve_by_aliases(
            aliases=["fact_a", "fact_a", "", "fact_missing"],
            access_context=make_access_context(user_id="u1"),
        )

        assert self.mock_library.mid_term.get_by_alias.call_count == 2
        assert response.memories == [mem]


class TestRetrievalFamiliarAccessStats:

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.familiar = RetrievalFamiliar(engine=Mock(), memory_library=self.mock_library)

    @pytest.mark.asyncio
    async def test_update_access_stats_per_item_failure(self):
        m1, m2 = _make_memory("m1"), _make_memory("m2")
        self.mock_library.mid_term.update_access_info.side_effect = [RuntimeError("fail"), None]
        await self.familiar.update_access_stats(make_access_context(user_id="u1"), [m1, m2])
        assert self.mock_library.mid_term.update_access_info.call_count == 2

    @pytest.mark.asyncio
    async def test_update_access_stats_empty_list(self):
        await self.familiar.update_access_stats(make_access_context(user_id="u1"), [])
        self.mock_library.mid_term.update_access_info.assert_not_called()


class TestRetrievalFamiliarShortTermTopics:

    def setup_method(self):
        self.mock_library = _make_memory_library()
        self.familiar = RetrievalFamiliar(engine=Mock(), memory_library=self.mock_library)

    def test_list_active_topics_excludes_empty_and_sorts_by_access(self):
        block = LogicalBlock()
        old_topic = _make_topic_data("old", blocks=[block], last_accessed_at=1.0)
        empty_topic = _make_topic_data("empty", blocks=[], last_accessed_at=3.0)
        new_topic = _make_topic_data("new", blocks=[block], last_accessed_at=2.0)
        self.mock_library.short_term.list_topic_data.return_value = [old_topic, empty_topic, new_topic]

        access_context = make_access_context(user_id="u1")
        snapshots = self.familiar.list_active_topics(access_context=access_context)

        self.mock_library.short_term.list_topic_data.assert_called_once_with(
            access_context, include_empty=False
        )
        assert [s.topic_id for s in snapshots] == ["new", "old"]

