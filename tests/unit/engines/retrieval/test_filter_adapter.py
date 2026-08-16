"""
QdrantFilterConverter 单元测试

测试覆盖 MutiAgentSystem.md §3.3.1 Visibility Scope Filtering:
  (visibility == 'PUBLIC')
  OR (visibility == 'WORKSPACE' AND team_id == current_team_id)
  OR (visibility == 'PRIVATE' AND source_agent_id == current_active_agent_id)
  PLUS: user_id 作为不可覆盖的 must 安全基线
"""

import pytest
from qdrant_client.models import Filter, FieldCondition, MatchValue

from hivememory.core.models import Identity, MemoryType
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.models import QueryFilters


def _convert(identity=None, memory_type=None, min_confidence=0.0) -> Filter:
    converter = QdrantFilterConverter()
    filters = QueryFilters(
        identity=identity,
        memory_type=memory_type,
        min_confidence=min_confidence,
    )
    return converter.convert(filters)


def _must_keys(f: Filter):
    return [c.key for c in (f.must or []) if isinstance(c, FieldCondition)]


def _should_visibility_values(f: Filter):
    """从 should 子句中提取所有 visibility match 值"""
    values = []
    for item in (f.must or []):
        if isinstance(item, Filter) and item.should:
            for scope in item.should:
                for cond in scope.must:
                    if isinstance(cond, FieldCondition) and cond.key == "meta.visibility":
                        values.append(cond.match.value)
    return values


class TestUserIdBaseline:
    """user_id 安全基线：必须出现在 must 条件中"""

    def test_user_id_in_must(self):
        f = _convert(identity=Identity(user_id="u1"))
        assert "meta.user_id" in _must_keys(f)

    def test_user_id_value(self):
        f = _convert(identity=Identity(user_id="alice"))
        uid_cond = next(c for c in f.must if isinstance(c, FieldCondition) and c.key == "meta.user_id")
        assert uid_cond.match.value == "alice"

    def test_no_identity_no_user_id_condition(self):
        f = _convert(identity=None)
        assert "meta.user_id" not in _must_keys(f)


class TestPublicScope:
    """PUBLIC 作用域：任何 identity 都应包含 PUBLIC should 分支"""

    def test_public_scope_present(self):
        f = _convert(identity=Identity(user_id="u1"))
        assert "PUBLIC" in _should_visibility_values(f)

    def test_public_scope_no_team_id(self):
        """无 team_id 时 PUBLIC 仍然存在"""
        f = _convert(identity=Identity(user_id="u1", team_id=None))
        assert "PUBLIC" in _should_visibility_values(f)


class TestWorkspaceScope:
    """WORKSPACE 作用域：仅当 team_id 存在时才注入"""

    def test_workspace_present_when_team_id(self):
        f = _convert(identity=Identity(user_id="u1", team_id="team_a"))
        assert "WORKSPACE" in _should_visibility_values(f)

    def test_workspace_absent_without_team_id(self):
        f = _convert(identity=Identity(user_id="u1", team_id=None))
        assert "WORKSPACE" not in _should_visibility_values(f)

    def test_workspace_team_id_value(self):
        """WORKSPACE 分支中 team_id 条件值正确"""
        converter = QdrantFilterConverter()
        f = converter.convert(QueryFilters(identity=Identity(user_id="u1", team_id="team_x")))
        workspace_scope = None
        for item in (f.must or []):
            if isinstance(item, Filter) and item.should:
                for scope in item.should:
                    for cond in scope.must:
                        if isinstance(cond, FieldCondition) and cond.key == "meta.team_id":
                            workspace_scope = cond
        assert workspace_scope is not None
        assert workspace_scope.match.value == "team_x"


class TestPrivateScope:
    """PRIVATE 作用域：仅当 agent_id 存在时才注入"""

    def test_private_present_when_agent_id(self):
        f = _convert(identity=Identity(user_id="u1", agent_id="coder_doll"))
        assert "PRIVATE" in _should_visibility_values(f)

    def test_private_present_for_default_agent(self):
        """agent_id="default" 时仍注入 PRIVATE（default 是有效字符串）"""
        f = _convert(identity=Identity(user_id="u1", agent_id="default"))
        assert "PRIVATE" in _should_visibility_values(f)

    def test_private_agent_id_value(self):
        """PRIVATE 分支中 source_agent_id 条件值正确"""
        converter = QdrantFilterConverter()
        f = converter.convert(QueryFilters(identity=Identity(user_id="u1", agent_id="reviewer_doll")))
        private_cond = None
        for item in (f.must or []):
            if isinstance(item, Filter) and item.should:
                for scope in item.should:
                    for cond in scope.must:
                        if isinstance(cond, FieldCondition) and cond.key == "meta.source_agent_id":
                            private_cond = cond
        assert private_cond is not None
        assert private_cond.match.value == "reviewer_doll"


class TestFullIdentity:
    """完整 identity (user_id + agent_id + team_id) 时三个作用域全部存在"""

    def test_all_three_scopes(self):
        f = _convert(identity=Identity(user_id="u1", agent_id="coder_doll", team_id="team_a"))
        scopes = _should_visibility_values(f)
        assert "PUBLIC" in scopes
        assert "WORKSPACE" in scopes
        assert "PRIVATE" in scopes


class TestBusinessFilters:
    """业务过滤维度（memory_type / min_confidence）"""

    def test_memory_type_in_must(self):
        f = _convert(identity=Identity(user_id="u1"), memory_type=MemoryType.CODE_SNIPPET)
        assert "index.memory_type" in _must_keys(f)

    def test_min_confidence_in_must(self):
        f = _convert(identity=Identity(user_id="u1"), min_confidence=0.7)
        assert "meta.confidence_score" in _must_keys(f)

    def test_min_confidence_zero_not_in_must(self):
        f = _convert(identity=Identity(user_id="u1"), min_confidence=0.0)
        assert "meta.confidence_score" not in _must_keys(f)


class TestEmptyFilter:
    """identity=None 且无业务过滤时返回空 Filter"""

    def test_empty_filter(self):
        f = _convert(identity=None)
        assert f.must is None
