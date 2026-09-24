"""Memory schema 2.1 ownership、provenance、lifecycle 与时间契约。"""

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.errors import OwnerMismatchError, ScopeRequiredError
from hivememory.core.models import (
    ActorIdentity,
    Artifacts,
    IdentityScope,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryProvenance,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.provenance import normalize_contributing_agent_ids

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def _workspace(user_id: str = "u1", workspace_id: str = "main_workspace") -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id=user_id,
        workspace_key=workspace_id,
        workspace_id=workspace_id,
    )


def _lifecycle(**overrides) -> MemoryLifecycleState:
    kwargs = {"decay_anchor_at": NOW}
    kwargs.update(overrides)
    return MemoryLifecycleState(**kwargs)


def _meta(**overrides) -> MetaData:
    kwargs = {
        "workspace_identity": _workspace(),
        "provenance": MemoryProvenance(
            source_agent_id="source-agent",
            source_team_id="source-team",
        ),
        "access_policy": MemoryAccessPolicy.public(),
        "lifecycle": _lifecycle(),
    }
    kwargs.update(overrides)
    return MetaData(**kwargs)


def _atom() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=_meta(),
        index=IndexLayer(
            title="Workspace policy",
            summary="Memory schema 2.1 keeps one canonical ownership authority.",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="canonical ownership"),
    )


@pytest.mark.parametrize(
    ("visibility", "target_agent_id", "target_team_id"),
    [
        (MemoryVisibility.PUBLIC, "agent-a", None),
        (MemoryVisibility.PRIVATE, None, None),
        (MemoryVisibility.PRIVATE, "agent-a", "team-a"),
        (MemoryVisibility.TEAM, None, None),
        (MemoryVisibility.TEAM, "agent-a", "team-a"),
    ],
)
def test_access_policy_rejects_invalid_target_combinations(
    visibility: MemoryVisibility,
    target_agent_id: str | None,
    target_team_id: str | None,
) -> None:
    """捕获 source/target 混用或 PUBLIC 携带隐式授权目标的缺陷。"""
    with pytest.raises(ValidationError):
        MemoryAccessPolicy(
            visibility=visibility,
            target_agent_id=target_agent_id,
            target_team_id=target_team_id,
        )


@pytest.mark.parametrize("target_field", ["target_agent_id", "target_team_id"])
def test_access_policy_rejects_system_actor_as_target(target_field: str) -> None:
    """保留 system 只表示'没有具体 Agent 作为操作来源主体'，不得成为授权目标。"""
    kwargs = {"target_agent_id": None, "target_team_id": None}
    kwargs[target_field] = "system"

    if target_field == "target_agent_id":
        expected_visibility = MemoryVisibility.PRIVATE
    else:
        expected_visibility = MemoryVisibility.TEAM

    with pytest.raises(ValidationError, match="system"):
        MemoryAccessPolicy(visibility=expected_visibility, **kwargs)


def test_qdrant_payload_projects_schema_2_1_owner_without_legacy_user_authority() -> None:
    """捕获新写入继续双写 legacy user_id、形成第二 owner 权威的缺陷。"""
    payload = _atom().to_qdrant_payload()

    assert payload["schema_version"] == "2.1"
    assert payload["meta"]["workspace_identity"] == {
        "owner_user_id": "u1",
        "workspace_key": "main_workspace",
        "workspace_id": "main_workspace",
    }
    assert payload["meta"]["owner_user_id"] == "u1"
    assert "user_id" not in payload["meta"]


def test_memory_atom_rejects_unknown_schema() -> None:
    """捕获未知 schema 被直接送入 2.1 领域模型的缺陷。"""
    with pytest.raises(ValidationError):
        MemoryAtom.model_validate({**_atom().model_dump(mode="json"), "schema_version": 3})


def test_identity_scope_rejects_actor_owner_drift() -> None:
    """捕获生成入口用 actor user 覆盖另一资源 owner 的缺陷。"""
    with pytest.raises(OwnerMismatchError):
        IdentityScope(
            actor_identity=ActorIdentity(user_id="actor", agent_id="agent-a"),
            workspace_identity=_workspace(user_id="owner"),
        )


def test_memory_key_construction_rejects_missing_scope() -> None:
    """防止 WorkspaceMemoryKey.from_identity_scope 对缺失作用域退回 AttributeError。"""
    with pytest.raises(ScopeRequiredError) as caught:
        WorkspaceMemoryKey.from_identity_scope(None, uuid4())

    assert caught.value.code == "workspace.scope_required"


def test_provenance_contributors_normalized_and_system_excluded() -> None:
    """贡献者集合按首次出现顺序去重，system 与空白标识不是内容贡献者。"""
    provenance = MemoryProvenance(
        source_agent_id="system",
        contributing_agent_ids=["b2", " a1 ", "b2", "system", "", "a1"],
    )

    assert provenance.contributing_agent_ids == ("b2", "a1")


def test_normalize_helper_keeps_first_occurrence_order() -> None:
    """共享归一化 helper 保持首次出现顺序并剔除非法标识。"""
    assert normalize_contributing_agent_ids(["a1", "b2", "a1", "system", ""]) == ("a1", "b2")


def test_provenance_fields_do_not_participate_in_access_policy() -> None:
    """捕获 provenance 字段被当作授权 target 或影响可见性的缺陷。"""
    meta = _meta(
        provenance=MemoryProvenance(source_agent_id="system", contributing_agent_ids=("a1",))
    )

    assert meta.access_policy == MemoryAccessPolicy.public()
    assert meta.access_policy.target_agent_id is None
    assert meta.access_policy.target_team_id is None


def test_lifecycle_defaults_and_utc_contract() -> None:
    """lifecycle 默认值沿用既有口径，时间字段强制 aware。"""
    lifecycle = _lifecycle()

    assert lifecycle.access_count == 0
    assert lifecycle.last_accessed_at is None
    assert lifecycle.event_vitality_boost == 0.0
    assert lifecycle.vitality_score == 100.0
    assert lifecycle.confidence_score == 0.6
    assert lifecycle.verification_status.value == "UNVERIFIED"


def test_meta_time_fields_default_to_utc_now() -> None:
    """created_at/updated_at 默认值为 timezone-aware UTC，不再产生 naive 时间。"""
    before = datetime.now(UTC)
    meta = _meta()
    after = datetime.now(UTC)

    assert before <= meta.created_at <= after
    assert meta.created_at.tzinfo is not None
    assert meta.updated_at.tzinfo is not None


def test_meta_rejects_naive_time_fields() -> None:
    """naive 业务时间在模型边界 fail closed，不猜测时区。"""
    with pytest.raises(ValidationError):
        _meta(created_at=datetime(2026, 9, 22, 12, 0, 0))
    with pytest.raises(ValidationError):
        _meta(updated_at=datetime(2026, 9, 22, 12, 0, 0))
    with pytest.raises(ValidationError):
        _meta(lifecycle=_lifecycle(decay_anchor_at=datetime(2026, 9, 22, 12, 0, 0)))


def test_meta_time_offset_inputs_normalized_to_utc() -> None:
    """带 offset 的输入规范化为 UTC 等价时点。"""
    cst = timezone(timedelta(hours=8))
    aware = datetime(2026, 9, 22, 20, 0, 0, tzinfo=cst)

    meta = _meta(created_at=aware, updated_at=aware)

    assert meta.created_at == NOW
    assert meta.updated_at == NOW
    assert meta.created_at.utcoffset() == timedelta(0)


def test_lifecycle_decay_anchor_is_required() -> None:
    """decay_anchor_at 是衰减唯一基准，创建路径必须显式提供。"""
    with pytest.raises(ValidationError):
        MemoryLifecycleState()


def test_flat_provenance_and_session_fields_are_removed() -> None:
    """旧平铺 provenance/session 字段不再是 MetaData 字段，防止旧写入路径复活。"""
    for removed in (
        "source_agent_id",
        "source_team_id",
        "contributing_agent_ids",
        "session_id",
        "last_accessed_at",
        "access_count",
        "vitality_score",
        "confidence_score",
        "event_vitality_boost",
        "verification_status",
    ):
        assert removed not in MetaData.model_fields


def test_payload_history_summary_removed_and_agent_config_moved() -> None:
    """history_summary 退出 schema；agent_config 移到 payload 顶层。"""
    assert "history_summary" not in PayloadLayer.model_fields
    assert "agent_config" in PayloadLayer.model_fields
    assert "agent_config" not in Artifacts.model_fields


# ─── Index 字段取值口径：必填、合法空值与规范化 ────────────────────────────


@pytest.mark.parametrize("blank_title", ["", "   "])
def test_index_rejects_blank_title(blank_title: str) -> None:
    """标题去除首尾空白后必填：空串与纯空白都不是合法标题。"""
    with pytest.raises(ValidationError, match="title"):
        IndexLayer(title=blank_title, memory_type=MemoryType.FACT)


def test_index_accepts_empty_summary() -> None:
    """空摘要是合法值（如未填写描述的 Agent Profile），缺省即为空串。"""
    index = IndexLayer(title="Untitled summary memory", memory_type=MemoryType.FACT)

    assert index.summary == ""
    assert IndexLayer(title="t", summary="  ", memory_type=MemoryType.FACT).summary == ""


def test_index_normalizes_text_alias_and_tags() -> None:
    """首尾空白去除、空白 alias 视为未设置、tags 转小写并保序去重。"""
    index = IndexLayer(
        title="  Title  ",
        summary=" Short ",
        memory_type=MemoryType.FACT,
        alias="   ",
        tags=["Python", "utc", " ", "python ", "Date"],
    )

    assert index.title == "Title"
    assert index.summary == "Short"
    assert index.alias is None
    assert index.tags == ["python", "utc", "date"]


# ─── 赋值同样执行字段约束（validate_assignment）────────────────────────────


@pytest.mark.parametrize(
    ("layer", "field", "value"),
    [
        ("index", "title", "   "),
        ("index", "summary", "x" * 501),
        ("index", "alias", "a" * 61),
        ("meta", "version", 0),
        ("meta", "updated_at", datetime(2026, 9, 22, 12, 0, 0)),
        ("lifecycle", "confidence_score", 1.5),
        ("lifecycle", "decay_anchor_at", datetime(2026, 9, 22, 12, 0, 0)),
    ],
)
def test_assignment_rejects_invalid_value_and_keeps_previous(layer, field, value) -> None:
    """写入路径靠属性赋值维护原子：非法赋值必须当场拒绝，原值保持不变。

    捕获"构造时校验、赋值时绕过"导致非法值写入存储后无法再读回的缺陷。
    """
    atom = _atom()
    target = {"index": atom.index, "meta": atom.meta, "lifecycle": atom.meta.lifecycle}[layer]
    previous = getattr(target, field)

    with pytest.raises(ValidationError):
        setattr(target, field, value)

    assert getattr(target, field) == previous


def test_assignment_applies_normalization() -> None:
    """赋值与构造走同一规范化：tags/alias 编辑后即为 canonical 值。"""
    atom = _atom()

    atom.index.tags = ["B", "a", "b "]
    atom.index.alias = "  "

    assert atom.index.tags == ["b", "a"]
    assert atom.index.alias is None
