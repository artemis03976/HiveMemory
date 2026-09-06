"""Memory v2 受控解码与 fail-closed 行为（legacy v1 解释分支已删除）。"""

from uuid import uuid4

import pytest

from hivememory.engines.retrieval.memory_codec import (
    MemoryDecodeError,
    decode_memory_payload,
)


def _v2_payload() -> dict:
    """schema v2 payload 模板（嵌套归属 + 完整 Workspace 投影，同 to_qdrant_payload）。"""
    return {
        "schema_version": 2,
        "id": str(uuid4()),
        "meta": {
            "source_agent_id": "source-agent",
            "workspace_identity": {
                "owner_user_id": "u1",
                "workspace_key": "main_workspace",
                "workspace_id": "main_workspace",
            },
            "owner_user_id": "u1",
            "workspace_key": "main_workspace",
            "workspace_id": "main_workspace",
            "access_policy": {"visibility": "PUBLIC"},
            "created_at": "2026-09-06T00:00:00",
            "version": 1,
        },
        "index": {
            "title": "Memory",
            "summary": "Canonical v2 record used to verify decoding.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {"content": "content"},
        "relations": {},
    }


def test_v2_payload_decodes_to_canonical_atom() -> None:
    """canonical v2 记录解码为领域对象，缺贡献者集合按空集合处理。"""
    atom = decode_memory_payload(_v2_payload())

    assert atom.schema_version == 2
    assert atom.workspace_identity.workspace_id == "main_workspace"
    assert atom.meta.source_agent_id == "source-agent"
    assert atom.meta.contributing_agent_ids == ()


def test_v2_contributor_list_decodes_to_normalized_tuple() -> None:
    """贡献者集合从存储数组解码为去重、去 system、保持顺序的元组。"""
    payload = _v2_payload()
    payload["meta"]["contributing_agent_ids"] = ["b2", "a1", "b2", "system"]

    atom = decode_memory_payload(payload)

    assert atom.meta.contributing_agent_ids == ("b2", "a1")


def test_missing_schema_version_is_rejected() -> None:
    """legacy v1 解释分支已删除：缺少 schema_version 的记录 fail closed。"""
    payload = _v2_payload()
    payload.pop("schema_version")

    with pytest.raises(MemoryDecodeError, match="schema_version"):
        decode_memory_payload(payload)


def test_unknown_schema_version_is_rejected() -> None:
    """捕获未知版本被静默按 v1 或 v2 读取的缺陷。"""
    payload = _v2_payload()
    payload["schema_version"] = 7

    with pytest.raises(MemoryDecodeError, match="schema_version"):
        decode_memory_payload(payload)


def test_v2_projection_mismatch_is_rejected() -> None:
    """捕获 Qdrant 平铺索引字段覆盖领域 canonical ownership 的缺陷。"""
    payload = _v2_payload()
    payload["meta"]["workspace_id"] = "isolation_workspace"
    payload["meta"]["workspace_key"] = "isolation_workspace"

    with pytest.raises(MemoryDecodeError, match="不一致"):
        decode_memory_payload(payload)


def test_v2_nested_ownership_conflicting_with_projection_is_rejected() -> None:
    """嵌套归属与存储索引投影冲突时拒绝读取。"""
    payload = _v2_payload()
    payload["meta"]["workspace_identity"]["owner_user_id"] = "someone-else"

    with pytest.raises(MemoryDecodeError, match="不一致"):
        decode_memory_payload(payload)


def test_v2_partial_projection_is_rejected() -> None:
    """部分 Workspace 投影拒绝猜测补齐。"""
    payload = _v2_payload()
    payload["meta"].pop("workspace_key")

    with pytest.raises(MemoryDecodeError, match="部分 Workspace 投影"):
        decode_memory_payload(payload)


def test_v2_invalid_policy_target_is_rejected() -> None:
    """PRIVATE 策略缺少合法 target 时 fail closed，不回落默认值。"""
    payload = _v2_payload()
    payload["meta"]["access_policy"] = {"visibility": "PRIVATE"}

    with pytest.raises(MemoryDecodeError, match="无效的 Memory schema v2"):
        decode_memory_payload(payload)
