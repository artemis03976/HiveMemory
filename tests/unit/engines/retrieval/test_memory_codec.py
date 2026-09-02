"""Memory v1 compatibility decoder 与 v2 fail-closed 行为。"""

from copy import deepcopy
from uuid import uuid4

import pytest

from hivememory.core.models import MemoryVisibility, WorkspaceIdentity
from hivememory.engines.retrieval.memory_codec import (
    MemoryDecodeError,
    decode_memory_payload,
)


def _legacy_payload(*, visibility: str = "PUBLIC", team_id: str | None = None) -> dict:
    return {
        "id": str(uuid4()),
        "meta": {
            "source_agent_id": "source-agent",
            "user_id": "u1",
            "team_id": team_id,
            "visibility": visibility,
        },
        "index": {
            "title": "Legacy memory",
            "summary": "Legacy record used to verify safe compatibility read.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {"content": "legacy content"},
        "relations": {},
    }


def test_v1_private_normalizes_owner_provenance_and_target_separately() -> None:
    """捕获 legacy PRIVATE 把 source 字段继续当作 v2 ACL 权威的缺陷。"""
    atom = decode_memory_payload(_legacy_payload(visibility="PRIVATE"))

    assert atom.schema_version == 2
    assert atom.workspace_identity.workspace_id == "main_workspace"
    assert atom.meta.source_agent_id == "source-agent"
    assert atom.meta.access_policy.visibility == MemoryVisibility.PRIVATE
    assert atom.meta.access_policy.target_agent_id == "source-agent"


def test_v1_workspace_normalizes_to_team_without_linking_source_and_target() -> None:
    """捕获旧 WORKSPACE 被误解释为新 Workspace-wide visibility 的缺陷。"""
    atom = decode_memory_payload(
        _legacy_payload(visibility="WORKSPACE", team_id="team-a")
    )

    assert atom.meta.source_team_id == "team-a"
    assert atom.meta.access_policy.visibility == MemoryVisibility.TEAM
    assert atom.meta.access_policy.target_team_id == "team-a"


def test_v1_partial_workspace_projection_is_rejected() -> None:
    """捕获部分 owner/workspace 字段被 active Workspace 猜测补齐的缺陷。"""
    payload = _legacy_payload()
    payload["meta"]["owner_user_id"] = "u1"

    with pytest.raises(MemoryDecodeError, match="部分 Workspace 投影"):
        decode_memory_payload(payload)


def test_v2_projection_mismatch_is_rejected() -> None:
    """捕获 Qdrant 平铺索引字段覆盖领域 canonical ownership 的缺陷。"""
    legacy = _legacy_payload()
    atom = decode_memory_payload(legacy)
    payload = atom.to_qdrant_payload()
    payload["meta"]["workspace_id"] = "isolation_workspace"
    payload["meta"]["workspace_key"] = "isolation_workspace"

    with pytest.raises(MemoryDecodeError, match="不一致"):
        decode_memory_payload(payload)


def test_unknown_schema_version_is_rejected() -> None:
    """捕获未知版本被静默按 v1 或 v2 读取的缺陷。"""
    payload = deepcopy(_legacy_payload())
    payload["schema_version"] = 7

    with pytest.raises(MemoryDecodeError, match="schema_version"):
        decode_memory_payload(payload)


def test_v1_workspace_without_team_target_is_rejected() -> None:
    """捕获 legacy WORKSPACE 缺 team_id 时制造默认 target 的缺陷。"""
    with pytest.raises(MemoryDecodeError, match="缺少 team_id"):
        decode_memory_payload(_legacy_payload(visibility="WORKSPACE"))


def test_legacy_owner_only_belongs_to_corresponding_main_workspace() -> None:
    """捕获 legacy user_id 被解释为任意当前 Workspace owner 的缺陷。"""
    atom = decode_memory_payload(_legacy_payload())
    isolation = WorkspaceIdentity(
        owner_user_id="u1",
        workspace_key="isolation_workspace",
        workspace_id="isolation_workspace",
    )

    assert atom.workspace_identity != isolation
