"""Memory/Interaction Artifact 模型契约：schema 2、完整快照约束与 UTC 时间。"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryProvenance,
    MemoryType,
    MetaData,
    PayloadLayer,
    WorkspaceIdentity,
)
from hivememory.core.models.artifact import (
    InteractionArtifact,
    InteractionTurnSnapshot,
    MemoryCreationArtifact,
    MemoryVersionArtifact,
    snapshot_memory_atom,
    validate_memory_atom_snapshot,
)

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def _workspace() -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id="u1",
        workspace_key="main_workspace",
        workspace_id="main_workspace",
    )


def _atom(*, memory_id: UUID | None = None, version: int = 1) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=MetaData(
            workspace_identity=_workspace(),
            provenance=MemoryProvenance(source_agent_id="a1"),
            access_policy=MemoryAccessPolicy.public(),
            version=version,
            lifecycle=MemoryLifecycleState(decay_anchor_at=NOW),
        ),
        index=IndexLayer(
            title="Test Title",
            summary="A test memory summary",
            tags=["tag1", "tag2"],
            memory_type=MemoryType.FACT,
            alias="fact_test",
        ),
        payload=PayloadLayer(content="content"),
    )


# ─── 完整原子 JSON 快照（替代裁剪型 MemoryVersionSnapshot）──────────────────


def test_snapshot_memory_atom_roundtrips_full_structure():
    """快照 = 捕获时点完整原子的 canonical JSON，嵌套字段不丢失。"""
    atom = _atom()
    atom.payload.agent_config = {"model_name": "default", "temperature": 0.2}

    snapshot = snapshot_memory_atom(atom)

    assert snapshot["schema_version"] == "2.1"
    assert snapshot["id"] == str(atom.id)
    assert snapshot["payload"]["content"] == "content"
    assert snapshot["payload"]["agent_config"]["model_name"] == "default"
    assert snapshot["meta"]["lifecycle"]["decay_anchor_at"].startswith("2026-09-22")
    assert snapshot["meta"]["provenance"]["source_agent_id"] == "a1"

    # 深拷贝语义：修改原原子不影响已生成的快照。
    atom.payload.content = "mutated"
    assert snapshot["payload"]["content"] == "content"


def test_snapshot_validation_rejects_missing_keys():
    """快照缺必需顶层键时 fail closed。"""
    incomplete = {"schema_version": "2.1", "id": "x"}
    with pytest.raises(ValueError, match="缺少必需顶层键"):
        validate_memory_atom_snapshot(incomplete)


def test_snapshot_validation_rejects_wrong_embedded_schema():
    """快照内嵌 schema 必须是 2.1，裁剪/未知版本拒绝。"""
    snapshot = snapshot_memory_atom(_atom())
    snapshot["schema_version"] = 2
    with pytest.raises(ValueError, match="2.1"):
        validate_memory_atom_snapshot(snapshot)


def test_snapshot_validation_rejects_placeholder_structure():
    """顶层键齐全但 meta 为空占位的"伪完整原子"不能冒充完整快照。

    旧裁剪快照只有 content/tags，补空结构后标记 2.1 属于 §7.2 禁止的伪装。
    """
    placeholder = {
        "schema_version": "2.1",
        "id": str(uuid4()),
        "meta": {},
        "index": {"tags": ["t1"]},
        "payload": {"content": "legacy trimmed content"},
        "relations": {},
    }
    with pytest.raises(ValueError, match="不是有效的完整 MemoryAtom"):
        validate_memory_atom_snapshot(placeholder)


@pytest.mark.parametrize(
    ("location", "unknown_path"),
    [
        ("top", "legacy_note"),
        ("artifacts", "payload.artifacts.agent_config"),
    ],
)
def test_snapshot_validation_rejects_unknown_fields(location, unknown_path):
    """任一层级的未知字段都拒绝，嵌套 extra="ignore" 不能静默吞掉旧布局。"""
    snapshot = snapshot_memory_atom(_atom())
    if location == "top":
        snapshot["legacy_note"] = "x"
    else:
        snapshot["payload"]["artifacts"]["agent_config"] = {"model_name": "old"}

    with pytest.raises(ValueError, match=unknown_path.replace(".", r"\.")):
        validate_memory_atom_snapshot(snapshot)


# ─── MemoryVersionArtifact / MemoryCreationArtifact schema "2" ──────────────


def _version_artifact(**overrides) -> MemoryVersionArtifact:
    """默认构造与内嵌原子一致的 v2 版本记录（同 ID、同 Workspace、同版本号）。"""
    memory_id = uuid4()
    kwargs = {
        "memory_id": str(memory_id),
        "workspace_identity": _workspace(),
        "provenance": MemoryProvenance(source_agent_id="a1"),
        "version_number": 2,
        "update_source": "UPDATE",
        "snapshot_before": snapshot_memory_atom(_atom(memory_id=memory_id, version=1)),
        "snapshot_after": snapshot_memory_atom(_atom(memory_id=memory_id, version=2)),
        "changed_at": NOW,
    }
    kwargs.update(overrides)
    return MemoryVersionArtifact(**kwargs)


def test_version_artifact_schema_2_with_structured_provenance():
    """schema 2 版本记录使用结构化 provenance，且独立于 Memory schema 轴。"""
    artifact = _version_artifact()

    assert artifact.schema_version == "2"
    assert artifact.provenance.source_agent_id == "a1"
    assert artifact.snapshot_after["schema_version"] == "2.1"


def test_version_artifact_rejects_flat_legacy_provenance_fields():
    """旧平铺来源字段不再是模型字段，防止旧写入路径复活。"""
    assert "source_agent_id" not in MemoryVersionArtifact.model_fields
    assert "contributing_agent_ids" not in MemoryVersionArtifact.model_fields
    assert "source_agent_id" not in MemoryCreationArtifact.model_fields


def test_create_v1_rejects_snapshot_before():
    """CREATE v1 不允许携带 snapshot_before（无修改前状态）。"""
    memory_id = uuid4()
    v1 = snapshot_memory_atom(_atom(memory_id=memory_id, version=1))
    with pytest.raises(ValidationError, match="CREATE v1 版本记录不允许携带 snapshot_before"):
        _version_artifact(
            memory_id=str(memory_id),
            version_number=1,
            update_source="CREATE",
            snapshot_before=v1,
            snapshot_after=v1,
        )


def test_version_number_must_match_embedded_atom_version():
    """version_number 与 snapshot_after.meta.version 必须一致（§3.4）。"""
    with pytest.raises(ValidationError, match="version_number=3"):
        _version_artifact(version_number=3)


def test_version_record_rejects_snapshot_of_other_memory():
    """快照原子 ID 与记录 memory_id 不一致时拒绝，防止历史挂到错误资源。"""
    with pytest.raises(ValidationError, match="snapshot_after.id"):
        _version_artifact(memory_id=str(uuid4()), snapshot_before=None)


def test_version_record_rejects_snapshot_from_other_workspace():
    """快照 Workspace 与记录归属不一致时拒绝（§5.3 同属一个 Workspace）。"""
    other = WorkspaceIdentity(
        owner_user_id="u1",
        workspace_key="isolation_workspace",
        workspace_id="isolation_workspace",
    )
    with pytest.raises(ValidationError, match="snapshot_after 的 Workspace"):
        _version_artifact(workspace_identity=other)


def test_snapshot_before_must_precede_recorded_version():
    """snapshot_before 必须是本版本之前的原子，不能与 after 同版本。"""
    memory_id = uuid4()
    same = snapshot_memory_atom(_atom(memory_id=memory_id, version=2))
    with pytest.raises(ValidationError, match="必须早于"):
        _version_artifact(memory_id=str(memory_id), snapshot_before=same, snapshot_after=same)


def test_version_artifact_naive_changed_at_rejected():
    """changed_at 是持久化业务时间，naive 值在模型边界拒绝。"""
    with pytest.raises(ValidationError):
        _version_artifact(changed_at=datetime(2026, 9, 22, 12, 0, 0))


def test_creation_artifact_schema_2():
    """创建 Artifact 升级为 schema 2 并复用结构化 provenance。"""
    creation = MemoryCreationArtifact(
        memory_id=str(uuid4()),
        workspace_identity=_workspace(),
        provenance=MemoryProvenance(source_agent_id="a1"),
        source_intent="MANUAL",
    )

    assert creation.schema_version == "2"
    assert creation.created_at.tzinfo is not None


# ─── InteractionArtifact / InteractionTurnSnapshot（布局保持不变）──────────


def test_artifact_requires_canonical_workspace_ownership():
    """新 Artifact 缺少 Workspace 归属时必须在领域边界拒绝。"""
    with pytest.raises(ValidationError, match="workspace_identity"):
        InteractionArtifact(topic_id="topic-1")


def _snapshot_payload(**overrides) -> dict:
    """经 model_dump 持久化的 canonical 快照 JSON。"""
    payload = {
        "block_id": "block-1",
        "turn_id": "turn-1",
        "created_at": 1757000000.0,
        "actor_identity": {"user_id": "u1", "agent_id": "omni_doll", "team_id": None},
        "user_query": "你好",
        "assistant_final_text": "你好！",
        "turn_events": [],
        "actions": [],
        "semantic_traces": [],
    }
    payload.update(overrides)
    return payload


def test_snapshot_roundtrips_actor_identity():
    """canonical 快照：actor_identity 单字段冻结并原样读回。"""
    snapshot = InteractionTurnSnapshot.model_validate(_snapshot_payload())
    dumped = snapshot.model_dump(mode="json")

    assert "user_id" not in dumped
    assert dumped["actor_identity"]["user_id"] == "u1"
    assert dumped["actor_identity"]["agent_id"] == "omni_doll"

    restored = InteractionTurnSnapshot.model_validate(dumped)
    assert restored.actor_identity == snapshot.actor_identity


def test_snapshot_preserves_concrete_agent_and_team():
    snapshot = InteractionTurnSnapshot.model_validate(
        _snapshot_payload(
            actor_identity={"user_id": "u1", "agent_id": "omni_doll", "team_id": "team-7"}
        )
    )

    assert snapshot.actor_identity.agent_id == "omni_doll"
    assert snapshot.actor_identity.team_id == "team-7"


def test_legacy_flat_json_without_actor_identity_fails_closed():
    """旧平铺 JSON 的读取升级分支已删除：缺 actor_identity 直接 fail closed。

    历史平铺记录只能通过迁移工具的 canonical replacement 访问，
    领域模型不再解释 user_id/agent_id/team_id 三元组。
    """
    legacy_flat = {
        "block_id": "block-1",
        "turn_id": "turn-1",
        "user_id": "u1",
        "agent_id": "omni_doll",
        "team_id": None,
    }
    with pytest.raises(ValidationError, match="actor_identity"):
        InteractionTurnSnapshot.model_validate(legacy_flat)

    with pytest.raises(ValidationError, match="actor_identity"):
        InteractionTurnSnapshot.model_validate(
            {k: v for k, v in _snapshot_payload().items() if k != "actor_identity"}
        )


def test_flat_actor_fields_are_not_model_fields():
    """平铺三元组不是模型字段，extra="ignore" 不会复活旧语义。"""
    assert "user_id" not in InteractionTurnSnapshot.model_fields
    assert "agent_id" not in InteractionTurnSnapshot.model_fields
    assert "team_id" not in InteractionTurnSnapshot.model_fields


def test_interaction_artifact_has_no_top_level_agent_source_field():
    """InteractionArtifact 不设置顶层 Agent 来源字段。

    来源 provenance 按 block 粒度由 ``InteractionTurnSnapshot.actor_identity``
    记录；话题级单值来源无法表达同一话题内切换 Agent 的事实，也不得复活为
    Agent owner 语义。
    """
    assert "source_agent_id" not in InteractionArtifact.model_fields
    assert "owner_agent_id" not in InteractionArtifact.model_fields
