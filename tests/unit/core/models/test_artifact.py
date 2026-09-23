"""Memory/Interaction Artifact 模型契约：schema 2、完整快照约束与 UTC 时间。"""

from datetime import UTC, datetime
from uuid import uuid4

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


def _atom() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            workspace_identity=_workspace(),
            provenance=MemoryProvenance(source_agent_id="a1"),
            access_policy=MemoryAccessPolicy.public(),
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


# ─── MemoryVersionArtifact / MemoryCreationArtifact schema "2" ──────────────


def _version_artifact(**overrides) -> MemoryVersionArtifact:
    kwargs = {
        "memory_id": str(uuid4()),
        "workspace_identity": _workspace(),
        "provenance": MemoryProvenance(source_agent_id="a1"),
        "version_number": 2,
        "update_source": "UPDATE",
        "snapshot_after": snapshot_memory_atom(_atom()),
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
    with pytest.raises(ValidationError, match="snapshot_before"):
        _version_artifact(
            version_number=1,
            update_source="CREATE",
            snapshot_before=snapshot_memory_atom(_atom()),
        )


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
