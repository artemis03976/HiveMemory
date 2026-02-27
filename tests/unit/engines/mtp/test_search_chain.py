"""
SEARCH 指令执行链路测试

验证 MTP SEARCH 指令从 Koakuma._handle_search 的完整链路。

测试覆盖:
    1. _parse_mtp_filter 过滤器解析
    2. SEARCH → RetrievalFamiliar.retrieve() 调用参数
    3. _render_search_menu 结果渲染
    4. 别名注册到 AliasResolver
    5. Koakuma SEARCH E2E
    6. 参数校验

作者: HiveMemory Team
版本: 1.0
"""

import pytest
from uuid import uuid4
from unittest.mock import MagicMock, patch

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, Identity,
)
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.patchouli.protocol.models import RetrievalResponse
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig
from hivememory.patchouli.protocol.mtp import MTPResponseStatus


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


def _make_retrieval_response(memories=None) -> RetrievalResponse:
    return RetrievalResponse(
        memories=memories or [],
        rendered_context="",
        memories_count=len(memories) if memories else 0,
    )


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    from tests.unit.engines.mtp.conftest import make_mock_bus
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = _make_retrieval_response()
    bus = make_mock_bus(mock_retrieval=mock_retrieval)
    return KoakumaRuntime(bus=bus, config=KoakumaConfig())


# ========== Test 1: _parse_mtp_filter ==========

class TestParseFilter:
    """_parse_mtp_filter 过滤器解析测试"""

    def test_type_code(self, koakuma):
        f = koakuma._parse_mtp_filter("type:code")
        assert f is not None
        assert f.memory_type == MemoryType.CODE_SNIPPET

    def test_type_fact(self, koakuma):
        f = koakuma._parse_mtp_filter("type:fact")
        assert f is not None
        assert f.memory_type == MemoryType.FACT

    def test_type_url(self, koakuma):
        f = koakuma._parse_mtp_filter("type:url")
        assert f is not None
        assert f.memory_type == MemoryType.URL_RESOURCE

    def test_type_reflection(self, koakuma):
        f = koakuma._parse_mtp_filter("type:reflection")
        assert f is not None
        assert f.memory_type == MemoryType.REFLECTION

    def test_type_profile(self, koakuma):
        f = koakuma._parse_mtp_filter("type:profile")
        assert f is not None
        assert f.memory_type == MemoryType.USER_PROFILE

    def test_type_wip(self, koakuma):
        f = koakuma._parse_mtp_filter("type:wip")
        assert f is not None
        assert f.memory_type == MemoryType.WORK_IN_PROGRESS

    def test_tag_single(self, koakuma):
        f = koakuma._parse_mtp_filter("tag:python")
        assert f is not None
        assert "python" in f.tags

    def test_tag_multiple(self, koakuma):
        f = koakuma._parse_mtp_filter("tag:python tag:async")
        assert f is not None
        assert "python" in f.tags
        assert "async" in f.tags

    def test_agent_filter(self, koakuma):
        f = koakuma._parse_mtp_filter("agent:coder_01")
        assert f is not None
        assert f.source_agent_id == "coder_01"

    def test_confidence_filter(self, koakuma):
        f = koakuma._parse_mtp_filter("confidence:0.8")
        assert f is not None
        assert f.min_confidence == 0.8

    def test_confidence_out_of_range(self, koakuma):
        """超出范围的 confidence 被忽略"""
        f = koakuma._parse_mtp_filter("confidence:1.5")
        assert f is None  # 全空 → None

    def test_multi_token_combination(self, koakuma):
        f = koakuma._parse_mtp_filter("type:code tag:python confidence:0.7")
        assert f is not None
        assert f.memory_type == MemoryType.CODE_SNIPPET
        assert "python" in f.tags
        assert f.min_confidence == 0.7

    def test_unknown_type_ignored(self, koakuma):
        """未知 type 值被忽略"""
        f = koakuma._parse_mtp_filter("type:UNKNOWN")
        assert f is None

    def test_unknown_key_ignored(self, koakuma):
        """未知 key 被忽略"""
        f = koakuma._parse_mtp_filter("foo:bar")
        assert f is None

    def test_invalid_token_no_colon(self, koakuma):
        """无冒号的 token 被忽略"""
        f = koakuma._parse_mtp_filter("garbage")
        assert f is None

    def test_empty_string(self, koakuma):
        assert koakuma._parse_mtp_filter("") is None

    def test_none_input(self, koakuma):
        assert koakuma._parse_mtp_filter(None) is None

    def test_whitespace_only(self, koakuma):
        assert koakuma._parse_mtp_filter("   ") is None


# ========== Test 2: SEARCH → RetrievalRequest ==========

class TestSearchRetrievalRequest:
    """验证 SEARCH → RetrievalFamiliar.retrieve() 调用参数"""

    def test_query_passed_to_retrieval(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])
        koakuma.set_current_user("user_42")

        koakuma.execute_mtp('⟪ SEARCH | * | query="python decorators" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.semantic_query == "python decorators"

    def test_user_id_injected(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])
        koakuma.set_current_user("user_42")

        koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.user_id == "user_42"

    def test_filter_passed_to_retrieval(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        koakuma.execute_mtp('⟪ SEARCH | * | query="test" filter="type:code" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is not None
        assert call_args.filters.memory_type == MemoryType.CODE_SNIPPET

    def test_no_filter_passes_none(self, koakuma):
        mem = _make_memory()
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is None


# ========== Test 3: Search Result Rendering ==========

class TestSearchResultRendering:
    """_render_search_menu 结果渲染测试"""

    def test_single_result_menu(self, koakuma):
        mem = _make_memory(title="API Spec", summary="REST API specification", alias="fact_api_spec")
        result = _make_retrieval_response([mem])

        menu = koakuma._render_search_menu(result)

        assert "[Menu]:" in menu
        assert "1." in menu
        assert "fact_api_spec" in menu
        assert "(Alias)" in menu

    def test_multiple_results_menu(self, koakuma):
        mems = [
            _make_memory(title="API Spec", summary="REST API spec", alias="fact_api_spec"),
            _make_memory(title="DB Config", summary="Database configuration", alias="fact_db_config"),
        ]
        result = _make_retrieval_response(mems)

        menu = koakuma._render_search_menu(result)

        assert "1." in menu
        assert "2." in menu
        assert "fact_api_spec" in menu
        assert "fact_db_config" in menu

    def test_alias_from_index_preferred(self, koakuma):
        """优先使用 index.alias"""
        mem = _make_memory(title="My Title", alias="custom_alias")
        result = _make_retrieval_response([mem])

        menu = koakuma._render_search_menu(result)
        assert "custom_alias" in menu

    def test_alias_fallback_generated(self, koakuma):
        """无 index.alias 时 fallback 生成"""
        mem = _make_memory(title="My Title", alias=None)
        result = _make_retrieval_response([mem])

        menu = koakuma._render_search_menu(result)
        # fallback: {type_prefix}_{title_slug}
        assert "fact_my_title" in menu

    def test_summary_truncated_at_80(self, koakuma):
        long_summary = "A" * 200
        mem = _make_memory(summary=long_summary, alias="test_alias")
        result = _make_retrieval_response([mem])

        menu = koakuma._render_search_menu(result)
        # summary 应被截断到 80 字符
        lines = menu.split("\n")
        detail_line = [l for l in lines if "test_alias" in l][0]
        # 引号内的 summary 不应超过 80 字符
        assert "A" * 81 not in detail_line


# ========== Test 4: Alias Registration ==========

class TestSearchAliasRegistration:
    """SEARCH 后别名注册到 AliasResolver"""

    def test_aliases_registered_after_search(self, koakuma):
        mem = _make_memory(alias="fact_api_spec")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        koakuma.execute_mtp('⟪ SEARCH | * | query="api spec" ⟫')

        assert koakuma._alias_resolver.has_alias("fact_api_spec")
        assert koakuma._alias_resolver.resolve("fact_api_spec") == str(mem.id)

    def test_multiple_aliases_registered(self, koakuma):
        mems = [
            _make_memory(alias="fact_a"),
            _make_memory(alias="fact_b"),
        ]
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response(mems)

        koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        assert koakuma._alias_resolver.has_alias("fact_a")
        assert koakuma._alias_resolver.has_alias("fact_b")

    def test_registered_alias_resolvable_by_read(self, koakuma):
        """SEARCH 注册的 alias 可被 READ 解析"""
        mem = _make_memory(alias="fact_api", content="API documentation content")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])
        koakuma._bus._mock_storage.get_memory.return_value = mem

        # SEARCH 注册 alias
        koakuma.execute_mtp('⟪ SEARCH | * | query="api" ⟫')

        # READ 使用注册的 alias
        result = koakuma.execute_mtp('⟪ READ | fact_api | ⟫')
        assert result.success
        assert "API documentation content" in result.response_content


# ========== Test 5: Koakuma SEARCH E2E ==========

class TestKoakumaSearchE2E:
    """通过 execute_mtp 端到端测试 SEARCH"""

    def test_search_returns_menu(self, koakuma):
        mem = _make_memory(alias="fact_test", summary="Test summary")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        assert result.success
        assert "[Menu]:" in result.response_content
        assert "fact_test" in result.response_content

    def test_search_with_filter(self, koakuma):
        mem = _make_memory(alias="code_sort", memory_type=MemoryType.CODE_SNIPPET)
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="sort" filter="type:code" ⟫')

        assert result.success
        # 验证 filter 被传递
        call_args = koakuma._bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is not None

    def test_search_empty_result(self, koakuma):
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([])

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="nonexistent" ⟫')

        assert result.success
        assert "No memories found" in result.response_content

    def test_search_retrieval_exception(self, koakuma):
        koakuma._bus._mock_retrieval.retrieve.side_effect = Exception("Connection error")

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        assert not result.success
        assert "Search failed" in result.response_content

    def test_search_formatted_response_contains_xml(self, koakuma):
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="test" ⟫')

        assert '<mtp_response status="success"' in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_search_via_intercept(self, koakuma):
        """通过 intercept_and_execute 测试"""
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        agent_text = 'Let me search for that. ⟪ SEARCH | * | query="test"'
        result = koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert result.success


# ========== Test 6: Koakuma SEARCH Validation ==========

class TestKoakumaSearchValidation:
    """SEARCH 参数校验"""

    def test_missing_query(self, koakuma):
        result = koakuma.execute_mtp('⟪ SEARCH | * | ⟫')
        assert not result.success
        assert "query" in result.response_content.lower()

    def test_empty_query(self, koakuma):
        result = koakuma.execute_mtp('⟪ SEARCH | * | query="" ⟫')
        assert not result.success

    def test_invalid_filter_degrades_gracefully(self, koakuma):
        """无效 filter 静默降级，不影响搜索"""
        mem = _make_memory(alias="fact_test")
        koakuma._bus._mock_retrieval.retrieve.return_value = _make_retrieval_response([mem])

        result = koakuma.execute_mtp('⟪ SEARCH | * | query="test" filter="invalid:garbage" ⟫')

        # 搜索仍应成功 (filter 被忽略)
        assert result.success

    def test_search_with_only_filter_no_query(self, koakuma):
        result = koakuma.execute_mtp('⟪ SEARCH | * | filter="type:code" ⟫')
        assert not result.success
        assert "query" in result.response_content.lower()
