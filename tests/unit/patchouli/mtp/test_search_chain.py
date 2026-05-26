"""
SEARCH 指令执行链路测试

验证 MTP SEARCH 指令从 Koakuma._handle_search 的完整链路。

测试覆盖:
    1. _parse_mtp_filter 过滤器解析
    2. SEARCH → RetrievalFamiliar.retrieve() 调用参数
    3. RetrievalResponse.rendered_context 结果返回
    4. 别名注册到 KoakumaAtomCache
    5. Koakuma SEARCH E2E
    6. 参数校验

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import pytest
from uuid import uuid4
from unittest.mock import MagicMock, patch

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, Identity,
)
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.alice.runtime.models import MTPExecutionContext
from hivememory.system.config import KoakumaConfig
from hivememory.core.mtp import MTPResponseStatus


# ========== Helpers ==========

def _make_memory(
    title: str = "Test Memory",
    summary: str = "A test memory for unit testing",
    memory_type: MemoryType = MemoryType.FACT,
    alias: str = None,
    content: str = "test content",
) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title=title, summary=summary, tags=["test"],
            memory_type=memory_type, alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


def _make_retrieval_response(memories=None, rendered_context=None) -> RetrievalResponse:
    memories = memories or []
    if rendered_context is None and memories:
        rendered_context = "\n".join(
            f"[{memory.index.alias}]: {memory.index.summary}"
            for memory in memories
        )
    return RetrievalResponse(
        memories=memories,
        rendered_context=rendered_context or "",
        memories_count=len(memories),
    )


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    from .conftest import make_koakuma_runtime, make_mock_bus
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = _make_retrieval_response()
    bus = make_mock_bus(mock_retrieval=mock_retrieval)
    return make_koakuma_runtime(bus, KoakumaConfig())


def _execute_mtp(koakuma: KoakumaRuntime, text: str, context=None):
    return asyncio.run(koakuma.execute_mtp(text, context=context))


def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    return asyncio.run(koakuma.intercept_and_execute(assistant_text, context=context))


# ========== Test 1: _parse_mtp_filter ==========

from hivememory.core.mtp import MTPFilterParser

class TestParseFilter:
    """测试 _parse_mtp_filter 方法"""

    @pytest.fixture
    def koakuma(self):
        # 此处使用 MTPFilterParser 代替 koakuma._parse_mtp_filter
        return MTPFilterParser()

    def test_type_code(self, koakuma):
        filters, warnings = koakuma.parse("type:code")
        assert filters.memory_type == MemoryType.CODE_SNIPPET
        assert not warnings

    def test_type_fact(self, koakuma):
        filters, warnings = koakuma.parse("type:fact")
        assert filters.memory_type == MemoryType.FACT
        assert not warnings

    def test_type_url(self, koakuma):
        filters, warnings = koakuma.parse("type:url_resource")
        assert filters.memory_type == MemoryType.URL_RESOURCE
        assert not warnings

    def test_type_reflection(self, koakuma):
        filters, warnings = koakuma.parse("type:reflection")
        assert filters.memory_type == MemoryType.REFLECTION
        assert not warnings

    def test_type_profile(self, koakuma):
        filters, warnings = koakuma.parse("type:user_profile")
        assert filters.memory_type == MemoryType.USER_PROFILE
        assert not warnings

    def test_type_wip(self, koakuma):
        filters, warnings = koakuma.parse("type:wip")
        assert filters.memory_type == MemoryType.WORK_IN_PROGRESS
        assert not warnings

    def test_tag_single(self, koakuma):
        filters, warnings = koakuma.parse("tag:python")
        assert filters.tags == ["python"]
        assert not warnings

    def test_tag_multiple(self, koakuma):
        filters, warnings = koakuma.parse("tag:python tag:bug")
        assert filters.tags == ["python", "bug"]
        assert not warnings

    def test_agent_filter(self, koakuma):
        filters, warnings = koakuma.parse("agent:agent_123")
        assert filters.source_agent_id == "agent_123"
        assert not warnings

    def test_confidence_filter(self, koakuma):
        filters, warnings = koakuma.parse("confidence:0.8")
        assert filters.min_confidence == 0.8
        assert not warnings

    def test_confidence_out_of_range(self, koakuma):
        filters, warnings = koakuma.parse("confidence:1.5")
        # should ignore out of range and fallback to 0.0
        assert filters is None
        assert len(warnings) == 1
        assert "out of range" in warnings[0]

    def test_multi_token_combination(self, koakuma):
        filters, warnings = koakuma.parse("type:code tag:api agent:bot1 confidence:0.5")
        assert filters.memory_type == MemoryType.CODE_SNIPPET
        assert filters.tags == ["api"]
        assert filters.source_agent_id == "bot1"
        assert filters.min_confidence == 0.5
        assert not warnings

    def test_unknown_type_ignored(self, koakuma):
        filters, warnings = koakuma.parse("type:unknown_type tag:test")
        assert filters.memory_type is None
        assert filters.tags == ["test"]
        assert len(warnings) == 1
        assert "Unknown filter type" in warnings[0]

    def test_unknown_key_ignored(self, koakuma):
        filters, warnings = koakuma.parse("unknown:value tag:test")
        assert filters.tags == ["test"]
        assert len(warnings) == 1
        assert "Unknown filter key" in warnings[0]

    def test_invalid_token_no_colon(self, koakuma):
        filters, warnings = koakuma.parse("invalid_token tag:test")
        assert filters.tags == ["test"]
        assert len(warnings) == 1
        assert "missing ':' separator" in warnings[0]

    def test_empty_string(self, koakuma):
        filters, warnings = koakuma.parse("")
        assert filters is None
        assert not warnings

    def test_none_input(self, koakuma):
        filters, warnings = koakuma.parse(None)
        assert filters is None
        assert not warnings

    def test_whitespace_only(self, koakuma):
        filters, warnings = koakuma.parse("   \t  ")
        assert filters is None
        assert not warnings


# ========== Test 2: SEARCH → RetrievalRequest ==========

class TestSearchRetrievalRequest:
    """验证 SEARCH → RetrievalFamiliar.retrieve() 调用参数"""

    def test_query_passed_to_retrieval(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])
        context = MTPExecutionContext(identity=Identity(user_id="user_42"))

        _execute_mtp(koakuma, '⟪ SEARCH | * | query="python decorators" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.semantic_query == "python decorators"

    def test_user_id_injected(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])
        context = MTPExecutionContext(identity=Identity(user_id="user_42"))

        _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫', context=context)

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.user_id == "user_42"

    def test_filter_passed_to_retrieval(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" filter="type:code" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is not None
        assert call_args.filters.memory_type == MemoryType.CODE_SNIPPET

    def test_no_filter_passes_none(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is None


# ========== Test 3: Search Result Rendering ==========

class TestSearchResultRendering:
    """SEARCH uses RetrievalResponse.rendered_context from RetrievalFamiliar."""

    def test_single_result_rendered_context(self, koakuma):
        mem = _make_memory(title="API Spec", summary="REST API specification", alias="fact_api_spec")
        rendered = "<memory_ref alias='fact_api_spec'>REST API specification</memory_ref>"
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response(
            [mem],
            rendered_context=rendered,
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="api" ⟫')

        assert result.success
        assert result.response_content == rendered

    def test_multiple_results_rendered_context(self, koakuma):
        mems = [
            _make_memory(title="API Spec", summary="REST API spec", alias="fact_api_spec"),
            _make_memory(title="DB Config", summary="Database configuration", alias="fact_db_config"),
        ]
        rendered = (
            "<memory_ref alias='fact_api_spec'>REST API spec</memory_ref>\n"
            "<memory_ref alias='fact_db_config'>Database configuration</memory_ref>"
        )
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response(
            mems,
            rendered_context=rendered,
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="api db" ⟫')

        assert result.success
        assert result.response_content == rendered

    def test_filter_warning_appended_to_rendered_context(self, koakuma):
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response(
            [mem],
            rendered_context="<memory_ref alias='fact_test'>Test</memory_ref>",
        )

        result = _execute_mtp(
            koakuma,
            '⟪ SEARCH | * | query="test" filter="unknown:value" ⟫',
        )

        assert result.success
        assert "<memory_ref alias='fact_test'>Test</memory_ref>" in result.response_content
        assert "Unknown filter key" in result.response_content


# ========== Test 4: Alias Registration ==========

class TestSearchAliasRegistration:
    """SEARCH 后别名注册到 KoakumaAtomCache"""

    def test_aliases_registered_after_search(self, koakuma):
        mem = _make_memory(alias="fact_api_spec")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        _execute_mtp(koakuma, '⟪ SEARCH | * | query="api spec" ⟫')

        assert koakuma.atom_cache.has_alias("fact_api_spec")
        atom = koakuma.atom_cache.get_atom_by_alias("fact_api_spec")
        assert atom is not None
        assert str(atom.id) == str(mem.id)

    def test_multiple_aliases_registered(self, koakuma):
        mems = [
            _make_memory(alias="fact_a"),
            _make_memory(alias="fact_b"),
        ]
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response(mems)

        _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        assert koakuma.atom_cache.has_alias("fact_a")
        assert koakuma.atom_cache.has_alias("fact_b")

    def test_registered_alias_resolvable_by_read(self, koakuma):
        """SEARCH 注册的 alias 可被 READ 解析"""
        mem = _make_memory(alias="fact_api", content="API documentation content")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])
        koakuma._bus._mock_storage.get_memory.return_value = mem

        # SEARCH 注册 alias
        _execute_mtp(koakuma, '⟪ SEARCH | * | query="api" ⟫')

        # READ 使用注册的 alias
        result = _execute_mtp(koakuma, '⟪ READ | fact_api | ⟫')
        assert result.success
        assert "API documentation content" in result.response_content


# ========== Test 5: Koakuma SEARCH E2E ==========

class TestKoakumaSearchE2E:
    """通过 execute_mtp 端到端测试 SEARCH"""

    def test_search_returns_rendered_context(self, koakuma):
        mem = _make_memory(alias="fact_test", summary="Test summary")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response(
            [mem],
            rendered_context="<memory_ref alias='fact_test'>Test summary</memory_ref>",
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        assert result.success
        assert "<memory_ref" in result.response_content
        assert "fact_test" in result.response_content

    def test_search_with_filter(self, koakuma):
        mem = _make_memory(alias="code_sort", memory_type=MemoryType.CODE_SNIPPET)
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="sort" filter="type:code" ⟫')

        assert result.success
        # 验证 filter 被传递
        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is not None

    def test_search_empty_result(self, koakuma):
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([])

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="nonexistent" ⟫')

        assert result.success
        assert "No memories found" in result.response_content

    def test_search_retrieval_exception(self, koakuma):
        koakuma._bus._mock_retrieval.retrieve.side_effect = Exception("Connection error")

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        assert not result.success
        assert "An unexpected error occurred" in result.response_content

    def test_search_formatted_response_contains_xml(self, koakuma):
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        assert '<mtp_response status="success"' in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_search_via_intercept(self, koakuma):
        """通过 intercept_and_execute 测试"""
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        agent_text = 'Let me search for that. ⟪ SEARCH | * | query="test"'
        result = _intercept_and_execute(koakuma, agent_text)

        assert result is not None
        assert result.success


# ========== Test 6: Koakuma SEARCH Validation ==========

class TestKoakumaSearchValidation:
    """SEARCH 参数校验"""

    def test_missing_query(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ SEARCH | * | ⟫')
        assert not result.success
        assert "query" in result.response_content.lower()

    def test_empty_query(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="" ⟫')
        assert not result.success

    def test_invalid_filter_degrades_gracefully(self, koakuma):
        """无效 filter 静默降级，不影响搜索"""
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" filter="invalid:garbage" ⟫')

        # 搜索仍应成功 (filter 被忽略)
        assert result.success

    def test_search_with_only_filter_no_query(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ SEARCH | * | filter="type:code" ⟫')
        assert not result.success
        assert "query" in result.response_content.lower()
