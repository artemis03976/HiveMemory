"""
MTP 别名系统测试 (Section 2.3)

测试覆盖:
- MemoryGenerationEngine._build_alias() 别名构建逻辑
- _draft_to_memory() 中别名的端到端生成
- IndexLayer.alias 的 Qdrant 持久化
- Koakuma._make_alias_from_memory() 的存储别名优先逻辑

对应设计文档: MemoryToolProtocol.md Section 2.3
"""

import pytest
from unittest.mock import MagicMock

from hivememory.engines.generation.engine import (
    MemoryGenerationEngine,
    MEMORY_TYPE_ALIAS_PREFIX,
)
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryType,
)
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime


# ========== _build_alias 单元测试 ==========

class TestBuildAlias:
    """测试 MemoryGenerationEngine._build_alias() 别名构建"""

    def test_code_snippet_with_suffix(self):
        """CODE_SNIPPET + LLM suffix → code_quicksort_impl"""
        result = MemoryGenerationEngine._build_alias(
            "CODE_SNIPPET", "quicksort_impl", "Quick Sort Algorithm"
        )
        assert result == "code_quicksort_impl"

    def test_fact_with_suffix(self):
        """FACT + LLM suffix → fact_project_env"""
        result = MemoryGenerationEngine._build_alias(
            "FACT", "project_env", "Project Environment"
        )
        assert result == "fact_project_env"

    def test_url_resource_with_suffix(self):
        """URL_RESOURCE + LLM suffix → url_python_datetime_docs"""
        result = MemoryGenerationEngine._build_alias(
            "URL_RESOURCE", "python_datetime_docs", "Python Docs"
        )
        assert result == "url_python_datetime_docs"

    def test_reflection_with_suffix(self):
        """REFLECTION → ref_ prefix"""
        result = MemoryGenerationEngine._build_alias(
            "REFLECTION", "avoid_global_state", "Avoid Global State"
        )
        assert result == "ref_avoid_global_state"

    def test_user_profile_with_suffix(self):
        """USER_PROFILE → user_ prefix"""
        result = MemoryGenerationEngine._build_alias(
            "USER_PROFILE", "prefers_typescript", "Prefers TypeScript"
        )
        assert result == "user_prefers_typescript"

    def test_wip_with_suffix(self):
        """WORK_IN_PROGRESS → wip_ prefix"""
        result = MemoryGenerationEngine._build_alias(
            "WORK_IN_PROGRESS", "refactor_auth", "Refactor Auth Module"
        )
        assert result == "wip_refactor_auth"

    def test_fallback_from_title(self):
        """alias_suffix 为空时从 title 派生"""
        result = MemoryGenerationEngine._build_alias(
            "FACT", "", "Project Environment"
        )
        assert result == "fact_project_environment"

    def test_fallback_from_title_with_special_chars(self):
        """title 含特殊字符时正确清洗"""
        result = MemoryGenerationEngine._build_alias(
            "CODE_SNIPPET", "", "Python utils: parse_date() 函数"
        )
        assert result == "code_python_utils_parse_date"

    def test_suffix_sanitization_uppercase(self):
        """suffix 含大写和特殊字符时清洗"""
        result = MemoryGenerationEngine._build_alias(
            "CODE_SNIPPET", "  UPPER Case!! ", "irrelevant"
        )
        # Space between "UPPER" and "Case" is removed by [^a-z0-9_] sanitization
        # after lowercasing: "upper case!!" → "uppercase"
        assert result == "code_uppercase"

    def test_unknown_type_uses_mem_prefix(self):
        """未知类型使用 mem_ 前缀"""
        result = MemoryGenerationEngine._build_alias(
            "UNKNOWN_TYPE", "test_thing", "Test"
        )
        assert result == "mem_test_thing"

    def test_empty_everything_returns_none(self):
        """全空时返回 None"""
        result = MemoryGenerationEngine._build_alias("FACT", "", "")
        assert result is None

    def test_suffix_truncation(self):
        """超长 suffix 截断至 40 字符"""
        long_suffix = "a" * 60
        result = MemoryGenerationEngine._build_alias(
            "FACT", long_suffix, "irrelevant"
        )
        assert result == f"fact_{'a' * 40}"
        assert len(result) == 5 + 40  # "fact_" + 40 chars

    def test_consecutive_underscores_collapsed(self):
        """连续下划线合并"""
        result = MemoryGenerationEngine._build_alias(
            "FACT", "hello___world", "irrelevant"
        )
        assert result == "fact_hello_world"

    def test_prefix_mapping_completeness(self):
        """验证所有 MemoryType 都有对应前缀"""
        for mem_type in MemoryType:
            assert mem_type.value in MEMORY_TYPE_ALIAS_PREFIX, (
                f"MemoryType {mem_type.value} missing from MEMORY_TYPE_ALIAS_PREFIX"
            )


# ========== _draft_to_memory 集成测试 ==========

class TestDraftToMemoryAlias:
    """测试 _draft_to_memory 中别名的端到端生成"""

    @pytest.fixture
    def engine(self):
        """提供 MemoryGenerationEngine 实例 (mock 依赖)"""
        return MemoryGenerationEngine(
            storage=MagicMock(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )

    @pytest.fixture
    def identity(self):
        return Identity(user_id="user1", agent_id="agent1", session_id="sess1")

    def test_alias_from_llm_suffix(self, engine, identity):
        """LLM 提供 alias_suffix 时正确拼接"""
        draft = ExtractedMemoryDraft(
            title="Quick Sort Algorithm",
            summary="A quicksort implementation in Python.",
            tags=["python", "algorithm"],
            memory_type="CODE_SNIPPET",
            content="def quicksort(arr): ...",
            confidence_score=0.9,
            has_value=True,
            alias_suffix="quicksort_impl",
        )
        memory = engine._draft_to_memory(draft, identity)
        assert memory.index.alias == "code_quicksort_impl"

    def test_alias_fallback_from_title(self, engine, identity):
        """alias_suffix 为空时从 title 派生"""
        draft = ExtractedMemoryDraft(
            title="API Rate Limit",
            summary="The API has a rate limit of 100 req/min.",
            tags=["api", "config"],
            memory_type="FACT",
            content="Rate limit: 100 req/min",
            confidence_score=0.8,
            has_value=True,
            alias_suffix="",
        )
        memory = engine._draft_to_memory(draft, identity)
        assert memory.index.alias == "fact_api_rate_limit"

    def test_alias_persists_to_qdrant_payload(self, engine, identity):
        """别名通过 to_qdrant_payload 持久化"""
        draft = ExtractedMemoryDraft(
            title="Test",
            summary="A test memory for alias persistence.",
            tags=["test"],
            memory_type="FACT",
            content="test content",
            confidence_score=0.5,
            has_value=True,
            alias_suffix="persistence_check",
        )
        memory = engine._draft_to_memory(draft, identity)
        payload = memory.to_qdrant_payload()
        assert payload["index"]["alias"] == "fact_persistence_check"

    def test_alias_none_when_no_suffix_no_title(self, engine, identity):
        """alias_suffix 和 title 都无法生成时为 None"""
        draft = ExtractedMemoryDraft(
            title="!@#$%",
            summary="Special chars only title.",
            tags=["test"],
            memory_type="FACT",
            content="test",
            confidence_score=0.5,
            has_value=True,
            alias_suffix="",
        )
        memory = engine._draft_to_memory(draft, identity)
        assert memory.index.alias is None


# ========== Koakuma 别名偏好测试 ==========

class TestKoakumaAliasPreference:
    """测试 Koakuma._make_alias_from_memory 的存储别名优先逻辑"""

    def test_prefer_stored_alias(self):
        """存储了正式别名时直接返回"""
        mem = MagicMock()
        mem.index.alias = "fact_api_spec"
        mem.index.memory_type.value = "FACT"
        mem.index.title = "API Specification"

        result = KoakumaRuntime._make_alias_from_memory(mem)
        assert result == "fact_api_spec"

    def test_fallback_when_alias_none(self):
        """alias 为 None 时 fallback 到运行时生成"""
        mem = MagicMock()
        mem.index.alias = None
        mem.index.memory_type.value = "FACT"
        mem.index.title = "API Specification"

        result = KoakumaRuntime._make_alias_from_memory(mem)
        assert result == "fact_api_specification"

    def test_fallback_when_alias_empty(self):
        """alias 为空字符串时 fallback 到运行时生成"""
        mem = MagicMock()
        mem.index.alias = ""
        mem.index.memory_type.value = "CODE_SNIPPET"
        mem.index.title = "Parse Date"

        result = KoakumaRuntime._make_alias_from_memory(mem)
        assert result == "code_parse_date"
