from uuid import uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.artifact import (
    InteractionArtifact,
    InteractionTurnSnapshot,
    MemoryVersionSnapshot,
)
from tests.helpers.memory import make_memory_metadata


def test_memory_version_snapshot_from_memory_atom_captures_mutable_fields():
    atom = MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="Test Title",
            summary="A test memory summary",
            tags=["tag1", "tag2"],
            memory_type=MemoryType.FACT,
            alias="fact_test",
        ),
        payload=PayloadLayer(content="content"),
    )

    snapshot = MemoryVersionSnapshot.from_memory_atom(atom)

    assert snapshot.content == "content"
    assert snapshot.alias == "fact_test"
    assert snapshot.title == "Test Title"
    assert snapshot.summary == "A test memory summary"
    assert set(snapshot.tags) == {"tag1", "tag2"}
    assert snapshot.memory_type == "FACT"


def test_artifact_requires_canonical_workspace_ownership():
    """新 Artifact 缺少 Workspace 归属时必须在领域边界拒绝。"""
    with pytest.raises(ValidationError, match="workspace_identity"):
        InteractionArtifact(topic_id="topic-1")


# ─── InteractionTurnSnapshot actor 单字段收敛（读取升级分支已删除）──────────


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
