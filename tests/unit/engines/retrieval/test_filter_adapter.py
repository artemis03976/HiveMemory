"""Qdrant Memory ownership 与 actor policy 过滤投影。"""

from typing import Any

import pytest
from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition

from hivememory.core.models import (
    ActorIdentity,
    MemoryType,
    WorkspaceIdentity,
    system_actor_for_workspace,
)
from hivememory.core.models.query import QueryFilters
from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter


def _workspace(workspace_id: str = "main_workspace") -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id="u1", workspace_key=workspace_id, workspace_id=workspace_id
    )


def _actor() -> ActorIdentity:
    return ActorIdentity(user_id="u1", agent_id="agent-a", team_id="team-a")


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


def test_main_workspace_ownership_filter_has_no_legacy_branch() -> None:
    """捕获 main 查询仍保留 meta.user_id legacy OR 分支或 IsEmpty 守卫的缺陷。"""
    result = QdrantFilterConverter().convert(QueryFilters(), _workspace(), from_actor=_actor())
    values = _field_values(result)

    assert values["meta.owner_user_id"] == {"u1"}
    assert values["meta.workspace_id"] == {"main_workspace"}
    # legacy 分支已删除：不再按平铺 user_id / IsEmpty 兜底读取历史记录。
    assert "meta.user_id" not in values
    assert _empty_fields(result) == set()


def test_isolation_workspace_filter_scoped_to_its_own_workspace() -> None:
    """捕获第二 Workspace 查询越界读取其他 Workspace 记录的缺陷。"""
    result = QdrantFilterConverter().convert(
        QueryFilters(),
        _workspace("isolation_workspace"),
        from_actor=_actor(),
    )
    values = _field_values(result)

    assert values["meta.workspace_id"] == {"isolation_workspace"}
    assert values["meta.owner_user_id"] == {"u1"}
    assert "meta.user_id" not in values
    assert _empty_fields(result) == set()


def test_v2_actor_policy_targets_are_distinct_from_provenance() -> None:
    """捕获 PRIVATE/TEAM 继续用 source 字段充当 v2 ACL 的缺陷。"""
    result = QdrantFilterConverter().convert(QueryFilters(), _workspace(), from_actor=_actor())
    values = _field_values(result)

    assert values["meta.access_policy.target_agent_id"] == {"agent-a"}
    assert values["meta.access_policy.target_team_id"] == {"team-a"}
    # legacy PRIVATE 分支已删除：来源 provenance 字段不再进入授权过滤。
    assert "meta.provenance.source_agent_id" not in values
    assert "meta.visibility" not in values
    assert values["meta.access_policy.visibility"] == {"PUBLIC", "PRIVATE", "TEAM"}


def test_business_filters_are_added_without_replacing_hard_boundary() -> None:
    """捕获 memory_type/min_confidence 覆盖 owner/workspace must 条件的缺陷。"""
    result = QdrantFilterConverter().convert(
        QueryFilters(memory_type=MemoryType.FACT, min_confidence=0.7),
        _workspace("isolation_workspace"),
        from_actor=_actor(),
    )
    values = _field_values(result)

    assert values["meta.workspace_id"] == {"isolation_workspace"}
    assert values["index.memory_type"] == {"FACT"}
    confidence = next(
        item
        for item in result.must or []
        if isinstance(item, FieldCondition) and item.key == "meta.lifecycle.confidence_score"
    )
    assert confidence.range.gte == pytest.approx(0.7)


def test_source_agent_filter_matches_contributors_and_source_branches() -> None:
    """来源 Agent 过滤匹配贡献者集合，并保留 source 字段兼容历史记录。

    SETTLE 记忆的 meta.source_agent_id 是保留 system，实际参与内容的
    Agent 在 contributing_agent_ids 中；只匹配 source 字段会漏检
    "参与过但未收尾"的 Agent。
    """
    result = QdrantFilterConverter().convert(
        QueryFilters(source_agent_id="agent-a"),
        _workspace("isolation_workspace"),
        from_actor=_actor(),
    )

    values = _field_values(result)
    assert values["meta.provenance.contributing_agent_ids"] == {"agent-a"}
    assert values["meta.provenance.source_agent_id"] == {"agent-a"}
    assert values["meta.workspace_id"] == {"isolation_workspace"}


def test_filter_converter_requires_actor_for_visibility_query() -> None:
    """检索必须明确传入发起者，不能因缺失主体退回无策略查询。"""
    with pytest.raises(TypeError, match="from_actor"):
        QdrantFilterConverter().convert(QueryFilters(), _workspace())  # type: ignore[call-arg]


def test_management_filter_keeps_ownership_and_business_filters() -> None:
    """管理读取跳过 actor 策略，仍保留 Workspace 与业务筛选。"""
    result = QdrantFilterConverter().convert(
        QueryFilters(memory_type=MemoryType.FACT),
        _workspace("isolation_workspace"),
        from_actor=_actor(),
        enforce_actor_visibility=False,
    )
    values = _field_values(result)

    assert values["meta.owner_user_id"] == {"u1"}
    assert values["meta.workspace_id"] == {"isolation_workspace"}
    assert values["index.memory_type"] == {"FACT"}
    assert "meta.access_policy.visibility" not in values


def test_system_actor_filter_has_no_team_visibility() -> None:
    """system 无团队策略；PRIVATE 的 system target 另由领域模型禁止。"""
    workspace = _workspace()
    result = QdrantFilterConverter().convert(
        QueryFilters(), workspace, from_actor=system_actor_for_workspace(workspace)
    )
    values = _field_values(result)

    assert values["meta.access_policy.visibility"] == {"PUBLIC", "PRIVATE"}
    assert values["meta.access_policy.target_agent_id"] == {"system"}
    assert "meta.access_policy.target_team_id" not in values
