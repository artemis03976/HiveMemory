"""
SEARCH 指令执行链路测试

验证 MTP SEARCH 经真实操作入口、workspace 读取视图与编译器的协作链路。

测试覆盖:
    1. _parse_mtp_filter 过滤器解析
    2. SEARCH → workspace 操作入口 → backing 的检索契约
    3. Koakuma 通过 MemoryCompiler 编译检索结果
    4. 结果预热 workspace 原子缓存，后续 READ 命中
    5. Koakuma SEARCH 执行入口
    6. 参数校验

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.config.alice import KoakumaConfig
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.mtp import (
    MTPFilterParser,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope, make_runtime_scope, make_workspace_identity

MAIN = make_workspace_identity()

# ========== 辅助函数 ==========


def _make_memory(
    title: str = "Test Memory",
    summary: str = "A test memory for unit testing",
    memory_type: MemoryType = MemoryType.FACT,
    alias: str = None,
    content: str = "test content",
    user_id: str = "test_user",
) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id=user_id, source_agent_id="test_agent"),
        index=IndexLayer(
            title=title,
            summary=summary,
            tags=["test"],
            memory_type=memory_type,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


def _make_retrieval_response(memories=None) -> list[MemoryAtom]:
    """检索路由的返回值：按领域排序的完整原子列表（A2 §2.1）。"""
    return list(memories or [])


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    from .conftest import make_koakuma_runtime, make_mock_bus

    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = _make_retrieval_response()
    bus = make_mock_bus(mock_retrieval=mock_retrieval)
    return make_koakuma_runtime(bus, KoakumaConfig())


def _execute_mtp(koakuma: KoakumaRuntime, text: str, context=None):
    return asyncio.run(
        koakuma.execute_mtp(
            text,
            context=context or MTPExecutionContext(runtime_scope=make_runtime_scope()),
        )
    )


def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    from .conftest import normalize_worker_agent_mtp_output

    return asyncio.run(
        koakuma.intercept_and_execute(
            normalize_worker_agent_mtp_output(assistant_text),
            context=context or MTPExecutionContext(runtime_scope=make_runtime_scope()),
        )
    )


# ========== Test 1：_parse_mtp_filter ==========


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
        # 越界值应被忽略并回退到 0.0
        assert filters is None
        assert len(warnings) == 1
        assert warnings[0].message_key == "mtp.filter.confidence_out_of_range"
        assert warnings[0].params == {"value": 1.5}

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
        assert warnings[0].message_key == "mtp.filter.unknown_type"
        assert warnings[0].params == {"value": "unknown_type"}

    def test_unknown_key_ignored(self, koakuma):
        filters, warnings = koakuma.parse("unknown:value tag:test")
        assert filters.tags == ["test"]
        assert len(warnings) == 1
        assert warnings[0].message_key == "mtp.filter.unknown_key"
        assert warnings[0].params == {"key": "unknown"}

    def test_warning_is_structured(self, koakuma):
        filters, warnings = koakuma.parse("unknown:value tag:test")
        assert filters.tags == ["test"]
        assert len(warnings) == 1
        assert warnings[0].message_key == "mtp.filter.unknown_key"
        assert warnings[0].params == {"key": "unknown"}

    def test_invalid_token_no_colon(self, koakuma):
        filters, warnings = koakuma.parse("invalid_token tag:test")
        assert filters.tags == ["test"]
        assert len(warnings) == 1
        assert warnings[0].message_key == "mtp.filter.token_missing_separator"
        assert warnings[0].params == {"token": "invalid_token"}

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


# ========== Test 2：SEARCH → RetrievalRequest ==========


class TestSearchRetrievalRequest:
    """验证协议参数经真实能力层到达存储边界，并产生相应检索输出。"""

    def test_query_passed_to_retrieval(self, koakuma):
        mem = _make_memory()
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="python decorators" ⟫')

        call_args = koakuma.harness.backing.bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.semantic_query == "python decorators"
        assert call_args.top_k == 5
        assert call_args.keywords == []
        assert result.response_status == "success"
        assert "Test Memory" in result.response_content

    def test_search_uses_credential_identity_when_observation_labels_differ(self, koakuma):
        """观测标签不能替代凭据授权声明，检索仍以显式认证主体读取。"""
        mem = _make_memory(user_id="user_42")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )
        context = MTPExecutionContext(
            runtime_scope=make_runtime_scope(
                agent_id="display_agent", workspace_id="display_workspace"
            ),
            submit_operation=asyncio.run(
                koakuma.harness.submitter(make_identity_scope(user_id="user_42"))
            ),
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫', context=context)

        request = koakuma.harness.backing.bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert request.identity_scope.actor_identity.user_id == "user_42"
        assert request.identity_scope.actor_identity.agent_id == "test_agent"
        assert request.identity_scope.workspace_identity.owner_user_id == "user_42"
        assert request.identity_scope.workspace_identity.workspace_id == "main_workspace"
        assert result.response_status == "success"
        assert "Test Memory" in result.response_content

    def test_filter_passed_to_retrieval(self, koakuma):
        mem = _make_memory(memory_type=MemoryType.CODE_SNIPPET)
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" filter="type:code" ⟫')

        call_args = koakuma.harness.backing.bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters.memory_type == MemoryType.CODE_SNIPPET
        assert result.response_status == "success"
        assert "Test Memory" in result.response_content

    def test_no_filter_passes_none(self, koakuma):
        mem = _make_memory()
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        call_args = koakuma.harness.backing.bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters is None
        assert result.response_status == "success"
        assert "Test Memory" in result.response_content


# ========== Test 3：搜索结果渲染 ==========


class TestSearchResultRendering:
    """SEARCH 通过 MemoryCompiler 编译检索返回的完整原子列表。"""

    def test_single_result_compiled_context(self, koakuma):
        mem = _make_memory(
            title="API Spec", summary="REST API specification", alias="fact_api_spec"
        )
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="api" ⟫')

        assert result.success
        assert "API Spec" in result.response_content

    def test_multiple_results_compiled_context(self, koakuma):
        mems = [
            _make_memory(title="API Spec", summary="REST API spec", alias="fact_api_spec"),
            _make_memory(
                title="DB Config", summary="Database configuration", alias="fact_db_config"
            ),
        ]
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response(mems)
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="api db" ⟫')

        assert result.success
        assert "API Spec" in result.response_content
        assert "DB Config" in result.response_content

    def test_filter_warning_added_to_response_warnings(self, koakuma):
        mem = _make_memory(alias="fact_test")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response(
                [mem],
            )
        )

        result = _execute_mtp(
            koakuma,
            '⟪ SEARCH | * | query="test" filter="unknown:value" ⟫',
            context=MTPExecutionContext(runtime_scope=make_runtime_scope(), language="en"),
        )

        assert result.response_status == "success"
        assert "Test Memory" in result.response_content
        assert "<warnings>" in result.formatted_response
        assert "<warning>Note: Unknown filter key 'unknown' was ignored.</warning>" in (
            result.formatted_response
        )


# ========== Test 4：Alias 注册 ==========


class TestSearchReferences:
    """SEARCH 预热 workspace 原子缓存，随后的 READ 无需存储冷读。"""

    def test_search_reference_resolves_by_read(self, koakuma):
        mem = _make_memory(alias="fact_api", content="API documentation content")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )
        # 冷读边界刻意不可用；若检索结果未预热，READ 必然失败。
        koakuma.harness.backing.bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "SEARCH 后不应再次冷读"
        )
        search = _execute_mtp(koakuma, '⟪ SEARCH | * | query="api" ⟫')
        assert "fact_api" in search.response_content
        assert koakuma.harness.runtime.stats()["atom_size"] == 1
        assert koakuma.harness.backing.bus._memory_citations == []
        result = _execute_mtp(koakuma, "⟪ READ | fact_api | ⟫")
        assert result.response_status == "success"
        assert "API documentation content" in result.response_content
        assert koakuma.harness.backing.bus._memory_citations == [
            {"memory_id": mem.id, "source": "workspace.reference_read"}
        ]


# ========== Test 5：Koakuma SEARCH E2E ==========


class TestKoakumaSearchExecution:
    """通过 execute_mtp 验证 SEARCH 的输出、警告和错误。"""

    def test_search_returns_compiled_context(self, koakuma):
        mem = _make_memory(alias="fact_test", summary="Test summary")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        assert result.success
        assert "Test Memory" in result.response_content
        assert "Test summary" in result.response_content

    def test_search_with_filter(self, koakuma):
        mem = _make_memory(alias="code_sort", memory_type=MemoryType.CODE_SNIPPET)
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="sort" filter="type:code" ⟫')

        assert result.success
        # 验证 filter 被传递
        call_args = koakuma.harness.backing.bus._mock_retrieval.retrieve.call_args[1]["request"]
        assert call_args.filters.memory_type == MemoryType.CODE_SNIPPET
        assert "code_sort" in result.response_content

    def test_search_empty_result(self, koakuma):
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([])
        )

        result = _execute_mtp(
            koakuma,
            '⟪ SEARCH | * | query="nonexistent" ⟫',
            context=MTPExecutionContext(
                runtime_scope=make_runtime_scope(),
                language="en",
            ),
        )

        assert result.success
        assert result.response_content == ""
        assert "No memories found" in result.formatted_response

    def test_search_empty_result_language_zh(self, koakuma):
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([])
        )
        context = MTPExecutionContext(
            runtime_scope=make_runtime_scope(),
            language="zh",
        )

        result = _execute_mtp(
            koakuma,
            '⟪ SEARCH | * | query="nonexistent" ⟫',
            context=context,
        )

        assert result.success
        assert result.response_content == ""
        assert "未找到相关记忆" in result.formatted_response

    def test_search_empty_result_warning_entry(self, koakuma):
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([])
        )

        result = _execute_mtp(
            koakuma,
            '⟪ SEARCH | * | query="nonexistent" ⟫',
            context=MTPExecutionContext(
                runtime_scope=make_runtime_scope(),
                language="en",
            ),
        )

        assert result.response_status == "success"
        assert result.response_content == ""
        assert "<warning>No memories found. Try a different query.</warning>" in (
            result.formatted_response
        )

    def test_search_retrieval_exception(self, koakuma):
        koakuma.harness.backing.bus._mock_retrieval.retrieve.side_effect = Exception(
            "Connection error"
        )

        result = _execute_mtp(
            koakuma,
            '⟪ SEARCH | * | query="test" ⟫',
            context=MTPExecutionContext(
                runtime_scope=make_runtime_scope(),
                language="en",
            ),
        )

        assert not result.success
        assert result.response_content == ""
        assert "An unexpected error occurred" in result.formatted_response

    def test_search_formatted_response_contains_xml(self, koakuma):
        mem = _make_memory(alias="fact_test")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" ⟫')

        assert '<mtp_response status="success"' in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_search_via_intercept(self, koakuma):
        """通过 intercept_and_execute 测试"""
        mem = _make_memory(alias="fact_test")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        agent_text = 'Let me search for that. ⟪ SEARCH | * | query="test"'
        result = _intercept_and_execute(koakuma, agent_text)

        assert result is not None
        assert result.success


# ========== Test 6：Koakuma SEARCH 校验 ==========


class TestKoakumaSearchValidation:
    """SEARCH 参数校验"""

    def test_missing_query(self, koakuma):
        result = _execute_mtp(koakuma, "⟪ SEARCH | * | ⟫")
        assert not result.success
        assert result.response_content == ""
        assert "query" in result.formatted_response.lower()

    def test_empty_query(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="" ⟫')
        assert not result.success

    def test_invalid_filter_degrades_gracefully(self, koakuma):
        """无效 filter 静默降级，不影响搜索"""
        mem = _make_memory(alias="fact_test")
        koakuma.harness.backing.bus._mock_retrieval.retrieve.return_value = (
            _make_retrieval_response([mem])
        )

        result = _execute_mtp(koakuma, '⟪ SEARCH | * | query="test" filter="invalid:garbage" ⟫')

        # 搜索仍应成功 (filter 被忽略)
        assert result.success

    def test_search_with_only_filter_no_query(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ SEARCH | * | filter="type:code" ⟫')
        assert not result.success
        assert result.response_content == ""
        assert "query" in result.formatted_response.lower()
