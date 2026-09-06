from uuid import uuid4

import pytest
from pydantic import ValidationError

from hivememory.core.constants import SYSTEM_AGENT_ID
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


# ─── InteractionTurnSnapshot actor 收敛与旧 JSON 读兼容（v0.6.2 B2）──────────


def _legacy_snapshot_payload(**overrides) -> dict:
    """v0.6.2 之前经 model_dump 持久化的平铺 actor 三元组 JSON。"""
    payload = {
        "block_id": "block-1",
        "turn_id": "turn-1",
        "created_at": 1757000000.0,
        "user_id": "u1",
        "agent_id": "omni_doll",
        "team_id": None,
        "user_query": "你好",
        "assistant_final_text": "你好！",
        "turn_events": [],
        "actions": [],
        "semantic_traces": [],
    }
    payload.update(overrides)
    return payload


def test_new_format_snapshot_roundtrips_actor_identity():
    """新格式：actor_identity 单字段冻结并原样读回。"""
    snapshot = InteractionTurnSnapshot.model_validate(_legacy_snapshot_payload())
    dumped = snapshot.model_dump(mode="json")

    assert "user_id" not in dumped
    assert dumped["actor_identity"]["user_id"] == "u1"
    assert dumped["actor_identity"]["agent_id"] == "omni_doll"

    restored = InteractionTurnSnapshot.model_validate(dumped)
    assert restored.actor_identity == snapshot.actor_identity


def test_legacy_flat_json_upgrades_missing_agent_to_system_actor():
    """旧平铺 JSON 缺少具体 Agent 时重建为保留 system，不得回落 omni_doll。"""
    snapshot = InteractionTurnSnapshot.model_validate(
        _legacy_snapshot_payload(agent_id="")
    )

    assert snapshot.actor_identity.user_id == "u1"
    assert snapshot.actor_identity.agent_id == SYSTEM_AGENT_ID
    assert snapshot.actor_identity.team_id is None
    assert snapshot.user_query == "你好"


def test_legacy_flat_json_preserves_concrete_agent_and_team():
    snapshot = InteractionTurnSnapshot.model_validate(
        _legacy_snapshot_payload(team_id="team-7")
    )

    assert snapshot.actor_identity.agent_id == "omni_doll"
    assert snapshot.actor_identity.team_id == "team-7"


def test_legacy_flat_json_without_user_ownership_fails_closed():
    """旧数据缺少用户归属时 fail closed，进入迁移诊断而非猜测归属。"""
    with pytest.raises(ValidationError, match="user_id"):
        InteractionTurnSnapshot.model_validate(
            _legacy_snapshot_payload(user_id="")
        )

    with pytest.raises(ValidationError, match="user_id"):
        InteractionTurnSnapshot.model_validate(
            {k: v for k, v in _legacy_snapshot_payload().items()
             if k not in {"user_id", "agent_id", "team_id"}}
        )


def test_legacy_flat_fields_are_dropped_after_upgrade():
    """升级后旧平铺字段不再出现在模型字段与 JSON dump 中。"""
    snapshot = InteractionTurnSnapshot.model_validate(_legacy_snapshot_payload())

    assert "user_id" not in type(snapshot).model_fields
    assert "agent_id" not in type(snapshot).model_fields
    assert "team_id" not in type(snapshot).model_fields
