"""Memory codec 受控解码：只接受 schema "2.1" canonical 格式，其余 fail closed。"""

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


@pytest.mark.parametrize("schema_version", [2, "2", 7])
def test_non_current_schema_version_is_rejected(schema_version) -> None:
    """只接受 "2.1"：已迁移完毕的整数 2 旧格式、Artifact 轴的 "2" 与未知版本都 fail closed。"""
    payload = _v21_payload()
    payload["schema_version"] = schema_version

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
