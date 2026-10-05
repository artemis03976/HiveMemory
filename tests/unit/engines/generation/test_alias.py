"""
AliasGenerator 单元测试（A2 §8 D-4 alias 唯一性第一层）

测试覆盖:
- build_candidate: 前缀映射、LLM 后缀优先、title 派生、清洗与截断
- generate: 空闲候选直接返回、占用时消歧重试、候选耗尽显式失败、
  唯一性按 Workspace 作用域判定
- MemoryGenerationEngine CREATE 路径使用生成器给出的空闲别名

占用查询端口以内存替身实现（按 Workspace + alias 记录占用者），属于被测
单元边界之外的中期存储。
"""

from uuid import uuid4

import pytest

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.core.models import MemoryType, WriteFocus
from hivememory.engines.generation.alias import MEMORY_TYPE_ALIAS_PREFIX, AliasGenerator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationContext,
    GenerationRequest,
)
from tests.helpers.memory import make_memory_identity_scope
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="other_workspace")


class _AliasOccupancy:
    """按 (Workspace, alias) 记录占用者的中期存储替身。"""

    def __init__(self) -> None:
        self._holders: dict[tuple, list] = {}
        self.queried: list[str] = []

    def occupy(self, workspace, alias: str) -> None:
        self._holders.setdefault((workspace, alias), []).append(uuid4())

    async def list_alias_holders(self, workspace_identity, alias, *, limit=2):
        self.queried.append(alias)
        return self._holders.get((workspace_identity, alias), [])[:limit]


# ========== build_candidate ==========


class TestBuildCandidate:
    """基础候选构造：纯规则，不查询唯一性。"""

    @pytest.mark.parametrize(
        ("memory_type", "suffix", "expected"),
        [
            ("CODE_SNIPPET", "quicksort_impl", "code_quicksort_impl"),
            ("FACT", "project_env", "fact_project_env"),
            ("URL_RESOURCE", "python_datetime_docs", "url_python_datetime_docs"),
            ("REFLECTION", "avoid_global_state", "ref_avoid_global_state"),
            ("USER_PROFILE", "prefers_typescript", "user_prefers_typescript"),
            ("WORK_IN_PROGRESS", "refactor_auth", "wip_refactor_auth"),
            ("AGENT_PROFILE", "coder_doll", "agent_coder_doll"),
        ],
    )
    def test_prefix_follows_memory_type(self, memory_type, suffix, expected):
        """记忆类型决定别名前缀，LLM 后缀原样拼接。"""
        assert AliasGenerator.build_candidate(memory_type, suffix, "irrelevant") == expected

    def test_unknown_type_uses_mem_prefix(self):
        """未知类型使用 mem_ 前缀。"""
        assert AliasGenerator.build_candidate("UNKNOWN_TYPE", "test_thing", "Test") == (
            "mem_test_thing"
        )

    def test_prefix_mapping_covers_every_memory_type(self):
        """每个 MemoryType 都有对应前缀，新增类型不会静默落到 mem_。"""
        missing = [t.value for t in MemoryType if t.value not in MEMORY_TYPE_ALIAS_PREFIX]
        assert missing == []

    def test_fallback_from_title(self):
        """alias_suffix 为空时从 title 派生。"""
        assert AliasGenerator.build_candidate("FACT", "", "Project Environment") == (
            "fact_project_environment"
        )

    def test_fallback_from_title_strips_special_chars(self):
        """title 含特殊字符与非 ASCII 字符时清洗。"""
        assert AliasGenerator.build_candidate(
            "CODE_SNIPPET", "", "Python utils: parse_date() 函数"
        ) == ("code_python_utils_parse_date")

    def test_suffix_sanitization(self):
        """后缀小写化并移除非 snake_case 字符。"""
        assert AliasGenerator.build_candidate("CODE_SNIPPET", "  UPPER Case!! ", "x") == (
            "code_uppercase"
        )

    def test_consecutive_underscores_collapsed(self):
        """连续下划线合并。"""
        assert AliasGenerator.build_candidate("FACT", "hello___world", "x") == ("fact_hello_world")

    def test_suffix_truncated_to_40_chars(self):
        """超长后缀截断至 40 字符。"""
        assert AliasGenerator.build_candidate("FACT", "a" * 60, "x") == "fact_" + "a" * 40

    def test_non_ascii_suffix_falls_back_to_title(self):
        """后缀清洗后为空（如纯中文）时回退 title，而不是产出 "fact_"。

        捕获只在清洗前判空、把退化别名交给唯一性消歧直至耗尽的缺陷。
        """
        assert AliasGenerator.build_candidate("FACT", "快速排序", "Quick Sort") == (
            "fact_quick_sort"
        )

    def test_no_candidate_when_suffix_and_title_sanitize_to_empty(self):
        """后缀与标题清洗后都为空时返回 None。"""
        assert AliasGenerator.build_candidate("FACT", "快速排序", "部署流程说明") is None

    def test_truncation_does_not_leave_trailing_underscore(self):
        """截断点落在下划线上时去除尾部下划线，消歧后缀不出现双下划线。"""
        assert AliasGenerator.build_candidate("FACT", "a" * 39 + "_tail", "x") == (
            "fact_" + "a" * 39
        )

    def test_no_candidate_when_suffix_and_title_empty(self):
        """后缀与标题都无法构造时返回 None。"""
        assert AliasGenerator.build_candidate("FACT", "", "") is None


# ========== generate ==========


class TestGenerate:
    """Workspace 内唯一别名生成。"""

    @pytest.mark.asyncio
    async def test_returns_base_candidate_when_free(self):
        """基础候选空闲时直接使用，不追加消歧后缀。"""
        generator = AliasGenerator(_AliasOccupancy())

        alias = await generator.generate(
            workspace_identity=MAIN,
            memory_type="FACT",
            alias_suffix="project_env",
            title="x",
        )

        assert alias == "fact_project_env"

    @pytest.mark.asyncio
    async def test_disambiguates_when_base_occupied(self):
        """基础候选被占用时追加 _2、_3……，返回第一个空闲候选。

        捕获生成器仍返回已占用基础候选、把冲突留给写入阶段的缺陷。
        """
        occupancy = _AliasOccupancy()
        occupancy.occupy(MAIN, "fact_project_env")
        occupancy.occupy(MAIN, "fact_project_env_2")
        generator = AliasGenerator(occupancy)

        alias = await generator.generate(
            workspace_identity=MAIN,
            memory_type="FACT",
            alias_suffix="project_env",
            title="x",
        )

        assert alias == "fact_project_env_3"

    @pytest.mark.asyncio
    async def test_occupancy_is_scoped_to_workspace(self):
        """其他 Workspace 的同名别名不构成占用。

        捕获占用查询丢失 Workspace 坐标、跨 Workspace 误判冲突的缺陷。
        """
        occupancy = _AliasOccupancy()
        occupancy.occupy(OTHER, "fact_project_env")
        generator = AliasGenerator(occupancy)

        alias = await generator.generate(
            workspace_identity=MAIN,
            memory_type="FACT",
            alias_suffix="project_env",
            title="x",
        )

        assert alias == "fact_project_env"

    @pytest.mark.asyncio
    async def test_raises_when_all_candidates_occupied(self):
        """消歧候选全部被占用时显式失败，不生成无别名或重名记忆。"""
        occupancy = _AliasOccupancy()
        occupancy.occupy(MAIN, "fact_busy")
        occupancy.occupy(MAIN, "fact_busy_2")
        occupancy.occupy(MAIN, "fact_busy_3")
        generator = AliasGenerator(occupancy, max_attempts=3)

        with pytest.raises(MemoryAliasConflictError) as exc_info:
            await generator.generate(
                workspace_identity=MAIN,
                memory_type="FACT",
                alias_suffix="busy",
                title="x",
            )

        assert exc_info.value.details["reason"] == "alias_candidates_exhausted"

    @pytest.mark.asyncio
    async def test_no_candidate_skips_occupancy_query(self):
        """无法构造候选时返回 None，不查询存储。"""
        occupancy = _AliasOccupancy()
        generator = AliasGenerator(occupancy)

        alias = await generator.generate(
            workspace_identity=MAIN,
            memory_type="FACT",
            alias_suffix="",
            title="!@#$%",
        )

        assert alias is None
        assert occupancy.queried == []


# ========== 引擎接线 ==========


class _SearchlessStore(_AliasOccupancy):
    """同时满足引擎查重检索的中期存储替身：查重永远无候选。"""

    async def search(self, *args, **kwargs):
        return []


class _Extractor:
    def __init__(self, draft: ExtractedMemoryDraft) -> None:
        self._draft = draft

    def extract(self, **kwargs):
        return self._draft


class _CreateDeduplicator:
    def check_duplicate(self, draft, candidates):
        return DuplicateDecision.CREATE, None


@pytest.mark.asyncio
async def test_engine_create_uses_generator_disambiguated_alias():
    """WRITE CREATE 产出的原子携带生成器消歧后的别名。

    捕获引擎绕过生成器、仍按静态规则拼出已被占用别名的缺陷。
    """
    identity_scope = make_memory_identity_scope()
    store = _SearchlessStore()
    store.occupy(identity_scope.workspace_identity, "fact_test_alias")
    draft = ExtractedMemoryDraft(
        title="测试记忆",
        summary="这是一段足够长的测试摘要用于通过验证",
        tags=["t1"],
        memory_type="FACT",
        content="内容",
        confidence_score=0.9,
        has_value=True,
        alias_suffix="test_alias",
    )
    engine = MemoryGenerationEngine(
        mid_term=store,
        extractor=_Extractor(draft),
        deduplicator=_CreateDeduplicator(),
    )

    outcomes = await engine.process(
        GenerationRequest(context=GenerationContext(), write_focus=WriteFocus(content="内容")),
        belong_to=identity_scope.workspace_identity,
        from_actor=identity_scope.actor_identity,
    )

    assert [outcome.duplicate_decision for outcome in outcomes] == [DuplicateDecision.CREATE]
    assert outcomes[0].atom.index.alias == "fact_test_alias_2"
