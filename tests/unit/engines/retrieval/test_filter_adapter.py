"""Qdrant Memory ownership 与 actor policy 过滤投影。"""

from typing import Any

import pytest
from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import Identity, MemoryType, build_internal_workspace_access
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
from hivememory.engines.retrieval.models import QueryFilters


def _access(workspace_id: str = "main_workspace"):
    return build_internal_workspace_access(
        Identity(user_id="u1", agent_id="agent-a", team_id="team-a"),
        workspace_id,
        f"filter-{workspace_id}",
    )


def _field_values(condition: Any) -> dict[str, set[Any]]:
    values: dict[str, set[Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, FieldCondition):
            if item.match is not None:
                values.setdefault(item.key, set()).add(item.match.value)
            return
        if isinstance(item, Filter):
            for child in [*(item.must or []), *(item.should or []), *(item.must_not or [])]:
                visit(child)

    visit(condition)
    return values


def _empty_fields(condition: Any) -> set[str]:
    result: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, IsEmptyCondition):
            result.add(item.is_empty.key)
            return
        if isinstance(item, Filter):
            for child in [*(item.must or []), *(item.should or []), *(item.must_not or [])]:
                visit(child)

    visit(condition)
    return result


def test_main_workspace_filter_contains_current_and_legacy_owner_branches() -> None:
    """捕获 main 查询漏掉受控 legacy 读取，或只按 user_id 过滤的缺陷。"""
    result = QdrantFilterConverter().convert(QueryFilters(), _access())
    values = _field_values(result)

    assert values["meta.owner_user_id"] == {"u1"}
    assert values["meta.workspace_id"] == {"main_workspace"}
    assert values["meta.user_id"] == {"u1"}
    assert _empty_fields(result) >= {
        "meta.owner_user_id",
        "meta.workspace_key",
        "meta.workspace_id",
    }


def test_isolation_workspace_filter_excludes_legacy_branch() -> None:
    """捕获第二 Workspace 召回缺 owner/workspace 历史记录的缺陷。"""
    result = QdrantFilterConverter().convert(
        QueryFilters(),
        _access("isolation_workspace"),
    )
    values = _field_values(result)

    assert values["meta.workspace_id"] == {"isolation_workspace"}
    assert "meta.user_id" not in values
    assert "meta.owner_user_id" not in _empty_fields(result)


def test_v2_actor_policy_targets_are_distinct_from_provenance() -> None:
    """捕获 PRIVATE/TEAM 继续用 source 字段充当 v2 ACL 的缺陷。"""
    result = QdrantFilterConverter().convert(QueryFilters(), _access())
    values = _field_values(result)

    assert values["meta.access_policy.target_agent_id"] == {"agent-a"}
    assert values["meta.access_policy.target_team_id"] == {"team-a"}
    assert "meta.source_agent_id" in values  # 仅 legacy PRIVATE compatibility branch
    assert values["meta.access_policy.visibility"] == {"PUBLIC", "PRIVATE", "TEAM"}


def test_business_filters_are_added_without_replacing_hard_boundary() -> None:
    """捕获 memory_type/min_confidence 覆盖 owner/workspace must 条件的缺陷。"""
    result = QdrantFilterConverter().convert(
        QueryFilters(memory_type=MemoryType.FACT, min_confidence=0.7),
        _access("isolation_workspace"),
    )
    values = _field_values(result)

    assert values["meta.workspace_id"] == {"isolation_workspace"}
    assert values["index.memory_type"] == {"FACT"}
    confidence = next(
        item
        for item in result.must or []
        if isinstance(item, FieldCondition) and item.key == "meta.confidence_score"
    )
    assert confidence.range.gte == pytest.approx(0.7)


def test_filter_converter_rejects_missing_access_context() -> None:
    """捕获检索内部边界在 scope 缺失时退回无过滤查询的缺陷。"""
    with pytest.raises(ScopeRequiredError):
        QdrantFilterConverter().convert(QueryFilters(), None)  # type: ignore[arg-type]
