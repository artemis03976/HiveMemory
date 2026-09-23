"""Memory codec 受控解码：schema "2.1" 新格式与整数 2 旧格式只读兼容。"""

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from hivememory.engines.retrieval.memory_codec import (
    MemoryDecodeError,
    decode_memory_payload,
)

AWARE_TIME = datetime(2026, 9, 6, 0, 0, 0, tzinfo=UTC)


def _workspace_block() -> dict:
    return {
        "owner_user_id": "u1",
        "workspace_key": "main_workspace",
        "workspace_id": "main_workspace",
    }


def _projection() -> dict:
    return {
        "owner_user_id": "u1",
        "workspace_key": "main_workspace",
        "workspace_id": "main_workspace",
    }


def _v21_payload() -> dict:
    """schema "2.1" payload 模板（嵌套 provenance/lifecycle + Workspace 投影）。"""
    return {
        "schema_version": "2.1",
        "id": str(uuid4()),
        "meta": {
            "workspace_identity": _workspace_block(),
            "provenance": {"source_agent_id": "source-agent"},
            "access_policy": {"visibility": "PUBLIC"},
            "created_at": AWARE_TIME.isoformat(),
            "updated_at": AWARE_TIME.isoformat(),
            "version": 1,
            "lifecycle": {
                "access_count": 3,
                "last_accessed_at": AWARE_TIME.isoformat(),
                "decay_anchor_at": AWARE_TIME.isoformat(),
            },
            **_projection(),
        },
        "index": {
            "title": "Memory",
            "summary": "Canonical 2.1 record used to verify decoding.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {"content": "content"},
        "relations": {},
    }


def _legacy_v2_payload() -> dict:
    """旧整数 schema 2 payload（平铺 provenance/动态字段，无 lifecycle 聚合）。"""
    return {
        "schema_version": 2,
        "id": str(uuid4()),
        "meta": {
            "source_agent_id": "source-agent",
            "source_team_id": "team-core",
            "workspace_identity": _workspace_block(),
            "access_policy": {"visibility": "PUBLIC"},
            "created_at": "2026-09-06T00:00:00+00:00",
            "updated_at": "2026-09-06T00:00:00+00:00",
            "last_accessed_at": "2026-09-07T00:00:00+00:00",
            "access_count": 5,
            "vitality_score": 88.0,
            "event_vitality_boost": 5.0,
            "confidence_score": 0.9,
            "verification_status": "VERIFIED",
            "session_id": "legacy-session",
            "version": 2,
            **_projection(),
        },
        "index": {
            "title": "Legacy Memory",
            "summary": "Legacy v2 record used to verify read-only conversion.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {
            "content": "legacy content",
            "history_summary": ["2026-09-06: legacy line"],
            "artifacts": {
                "refs": [],
                "events": [],
                "agent_config": {"model_name": "legacy-model"},
            },
        },
        "relations": {},
    }


# ─── schema "2.1" 新格式 ────────────────────────────────────────────────────


def test_v21_payload_decodes_to_canonical_atom() -> None:
    """canonical 2.1 记录解码为领域对象，lifecycle/provenance 完整还原。"""
    atom = decode_memory_payload(_v21_payload())

    assert atom.schema_version == "2.1"
    assert atom.workspace_identity.workspace_id == "main_workspace"
    assert atom.meta.provenance.source_agent_id == "source-agent"
    assert atom.meta.lifecycle.access_count == 3
    assert atom.meta.lifecycle.decay_anchor_at == AWARE_TIME


def test_v21_naive_time_is_rejected() -> None:
    """新格式中的 naive 时间 fail closed，不做时区猜测。"""
    payload = _v21_payload()
    payload["meta"]["updated_at"] = "2026-09-06T00:00:00"

    with pytest.raises(MemoryDecodeError, match="无效的 Memory schema 2.1"):
        decode_memory_payload(payload)


def test_v21_offset_time_normalized_to_utc() -> None:
    """带 offset 的时间规范化为 UTC。"""
    payload = _v21_payload()
    cst = timezone(timedelta(hours=8))
    payload["meta"]["updated_at"] = datetime(2026, 9, 6, 8, 0, 0, tzinfo=cst).isoformat()

    atom = decode_memory_payload(payload)

    assert atom.meta.updated_at == AWARE_TIME
    assert atom.meta.updated_at.utcoffset() == timedelta(0)


def test_missing_schema_version_is_rejected() -> None:
    """缺少 schema_version 的记录 fail closed。"""
    payload = _v21_payload()
    payload.pop("schema_version")

    with pytest.raises(MemoryDecodeError, match="schema_version"):
        decode_memory_payload(payload)


def test_unknown_schema_version_is_rejected() -> None:
    """未知版本不被静默按任何已知格式读取。"""
    payload = _v21_payload()
    payload["schema_version"] = 7

    with pytest.raises(MemoryDecodeError, match="schema_version"):
        decode_memory_payload(payload)


def test_v21_projection_mismatch_is_rejected() -> None:
    """捕获平铺索引字段覆盖领域 canonical ownership 的缺陷。"""
    payload = _v21_payload()
    payload["meta"]["workspace_id"] = "isolation_workspace"
    payload["meta"]["workspace_key"] = "isolation_workspace"

    with pytest.raises(MemoryDecodeError, match="不一致"):
        decode_memory_payload(payload)


def test_v21_partial_projection_is_rejected() -> None:
    """部分 Workspace 投影拒绝猜测补齐。"""
    payload = _v21_payload()
    payload["meta"].pop("workspace_key")

    with pytest.raises(MemoryDecodeError, match="部分 Workspace 投影"):
        decode_memory_payload(payload)


def test_v21_invalid_policy_target_is_rejected() -> None:
    """PRIVATE 策略缺少合法 target 时 fail closed，不回落默认值。"""
    payload = _v21_payload()
    payload["meta"]["access_policy"] = {"visibility": "PRIVATE"}

    with pytest.raises(MemoryDecodeError, match="无效的 Memory schema 2.1"):
        decode_memory_payload(payload)


# ─── 旧整数 schema 2 只读兼容 ───────────────────────────────────────────────


def test_legacy_v2_converts_to_aggregated_domain_object() -> None:
    """旧记录只读转换为 2.1 领域对象：平铺字段聚合、session/history 丢弃。"""
    atom = decode_memory_payload(_legacy_v2_payload())

    assert atom.schema_version == "2.1"
    assert atom.meta.provenance.source_agent_id == "source-agent"
    assert atom.meta.provenance.source_team_id == "team-core"
    assert atom.meta.lifecycle.access_count == 5
    assert atom.meta.lifecycle.vitality_score == 88.0
    assert atom.meta.lifecycle.confidence_score == 0.9
    assert atom.meta.lifecycle.verification_status.value == "VERIFIED"
    assert "session_id" not in type(atom.meta).model_fields
    assert "history_summary" not in type(atom.payload).model_fields


def test_legacy_v2_decay_anchor_initialized_from_updated_at() -> None:
    """兼容推断：decay_anchor_at 以旧 updated_at 初始化，保持既有衰减行为。"""
    atom = decode_memory_payload(_legacy_v2_payload())

    assert atom.meta.lifecycle.decay_anchor_at == AWARE_TIME
    assert atom.meta.updated_at == AWARE_TIME


def test_legacy_v2_agent_config_moves_to_new_location() -> None:
    """旧 artifacts.agent_config 迁入 payload.agent_config。"""
    atom = decode_memory_payload(_legacy_v2_payload())

    assert atom.payload.agent_config == {"model_name": "legacy-model"}
    assert "agent_config" not in type(atom.payload.artifacts).model_fields


def test_legacy_v2_agent_config_conflict_is_rejected() -> None:
    """agent_config 新旧位置同时存在且值不一致时拒绝猜测。"""
    payload = _legacy_v2_payload()
    payload["payload"]["agent_config"] = {"model_name": "new-model"}

    with pytest.raises(MemoryDecodeError, match="agent_config"):
        decode_memory_payload(payload)


def test_legacy_v2_naive_time_interpreted_as_server_local_utc() -> None:
    """naive 旧时间按写入时服务器本地时区解释并规范化为 UTC（只读兼容）。"""
    payload = _legacy_v2_payload()
    payload["meta"]["updated_at"] = "2026-09-06T00:00:00"

    atom = decode_memory_payload(payload)

    assert atom.meta.updated_at.tzinfo is not None
    assert atom.meta.updated_at.utcoffset() == timedelta(0)


def test_legacy_v2_missing_updated_at_is_rejected() -> None:
    """缺失历史事实不用当前时间补值，直接拒绝。"""
    payload = _legacy_v2_payload()
    payload["meta"].pop("updated_at")

    with pytest.raises(MemoryDecodeError, match="updated_at"):
        decode_memory_payload(payload)


def test_legacy_v2_missing_source_agent_is_rejected() -> None:
    """缺 provenance 来源的旧记录 fail closed。"""
    payload = _legacy_v2_payload()
    payload["meta"].pop("source_agent_id")

    with pytest.raises(MemoryDecodeError, match="source_agent_id"):
        decode_memory_payload(payload)


def test_strict_mode_rejects_legacy_payload() -> None:
    """mutation 入口的严格模式：旧 schema 记录只读，拒绝读出后回写。"""
    from hivememory.engines.retrieval.memory_codec import MemorySchemaReadOnlyError

    with pytest.raises(MemorySchemaReadOnlyError, match="只读"):
        decode_memory_payload(_legacy_v2_payload(), allow_legacy=False)


def test_strict_mode_accepts_current_schema() -> None:
    """严格模式不影响新格式读取。"""
    atom = decode_memory_payload(_v21_payload(), allow_legacy=False)

    assert atom.schema_version == "2.1"
