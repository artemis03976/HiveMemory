"""V1 Memory 迁移转换规则与 legacy Artifact 判定的单元测试。

被测单元：``hivememory.tools.v1_legacy_migration`` 的纯转换/判定逻辑；
存储与引擎协作边界由 ``tests/integration/tools/`` 下的流程测试覆盖。
"""

from uuid import uuid4

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    MemoryVisibility,
    VerificationStatus,
    WorkspaceIdentity,
)
from hivememory.tools.v1_legacy_migration import (
    REPLACEMENT_ID_PREFIX,
    _is_legacy_artifact,
    convert_v1_memory_payload,
    replacement_artifact_id,
    resolve_memory_workspace_from_meta,
)


def _v1_payload(*, visibility: str | None = "PUBLIC", user_id: str = "u1") -> dict:
    """schema v1 Qdrant payload 模板（与历史写入形状一致）。"""
    meta: dict = {
        "source_agent_id": "agent-a",
        "team_id": None,
        "created_at": "2026-06-14T12:00:00",
        "version": 1,
    }
    if visibility is not None:
        meta["visibility"] = visibility
    if user_id is not None:
        meta["user_id"] = user_id
    return {
        "id": str(uuid4()),
        "meta": meta,
        "index": {
            "title": "Legacy memory",
            "summary": "Legacy record used to verify migration conversion.",
            "memory_type": "FACT",
            "tags": ["legacy"],
        },
        "payload": {"content": "legacy content"},
        "relations": {"relates_to": [], "supersedes": [], "depends_on": []},
    }


def _main_workspace(user_id: str = "u1") -> WorkspaceIdentity:
    return WorkspaceIdentity(
        owner_user_id=user_id,
        workspace_key="main_workspace",
        workspace_id="main_workspace",
    )


# ============ 基本字段映射 ============


def test_public_v1_converts_to_schema_2_1_with_main_workspace_identity() -> None:
    """缺失 workspace_identity 的 V1 记录映射到 user 的 main_workspace。"""
    result = convert_v1_memory_payload(_v1_payload(), missing_visibility_policy="public")

    assert result.atom is not None
    assert result.atom.schema_version == "2.1"
    assert result.atom.workspace_identity == _main_workspace("u1")
    assert result.atom.meta.workspace_identity == _main_workspace("u1")
    assert result.atom.meta.access_policy.visibility == MemoryVisibility.PUBLIC
    assert result.atom.meta.provenance.source_agent_id == "agent-a"


def test_v1_flat_legacy_meta_fields_are_dropped_from_canonical_meta() -> None:
    """user_id/team_id/visibility 不再出现在 canonical meta 的权威语义里。"""
    payload = _v1_payload()
    payload["meta"]["team_id"] = "team-a"
    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is not None
    dumped = result.atom.meta.model_dump()
    assert "user_id" not in dumped
    assert "visibility" not in dumped
    # 平铺来源字段聚合进 provenance，不再是 meta 的直接键。
    assert "source_agent_id" not in dumped
    assert "source_team_id" not in dumped
    assert result.atom.meta.provenance.source_team_id == "team-a"


def test_contributors_are_never_guessed_during_v1_conversion() -> None:
    """V1 记录没有可信贡献者证据，迁移不得回填猜测集合。"""
    result = convert_v1_memory_payload(_v1_payload(), missing_visibility_policy="public")

    assert result.atom is not None
    assert result.atom.meta.provenance.contributing_agent_ids == ()


def test_v1_dynamic_fields_aggregate_into_lifecycle_with_time_evidence() -> None:
    """平铺动态字段聚合为 meta.lifecycle；decay_anchor_at 以内容时间为基准。"""
    result = convert_v1_memory_payload(_v1_payload(), missing_visibility_policy="public")

    assert result.atom is not None
    atom = result.atom
    dumped_meta = atom.meta.model_dump()
    for legacy_flat in (
        "access_count",
        "last_accessed_at",
        "vitality_score",
        "confidence_score",
        "event_vitality_boost",
        "verification_status",
        "session_id",
    ):
        assert legacy_flat not in dumped_meta
    lifecycle = atom.meta.lifecycle
    # 从未修订的 V1 记录：decay_anchor_at 以 created_at（== updated_at）初始化。
    assert lifecycle.decay_anchor_at == atom.meta.updated_at
    assert lifecycle.decay_anchor_at.tzinfo is not None
    assert atom.meta.created_at.tzinfo is not None
    assert lifecycle.access_count == 0
    assert lifecycle.last_accessed_at is None
    assert lifecycle.verification_status == VerificationStatus.UNVERIFIED


def test_v1_payload_session_and_history_summary_are_dropped_and_agent_config_moves() -> None:
    """session_id 与 payload.history_summary 在产物中不存在；agent_config 归位。"""
    payload = _v1_payload()
    payload["meta"]["session_id"] = "sess-legacy"
    payload["payload"]["history_summary"] = "legacy summary"
    payload["payload"]["artifacts"] = {
        "agent_config": {"model_name": "legacy-model"},
        "refs": [],
        "events": [],
    }

    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is not None
    dumped_payload = result.atom.payload.model_dump()
    assert "history_summary" not in dumped_payload
    # payload.artifacts.agent_config 移到顶层 payload.agent_config。
    assert dumped_payload["agent_config"] == {"model_name": "legacy-model"}
    assert "agent_config" not in dumped_payload["artifacts"]
    # session_id 丢弃不影响来源 provenance 聚合。
    assert result.atom.meta.provenance.source_agent_id == "agent-a"


# ============ visibility → access_policy 映射 ============


def test_private_v1_maps_policy_target_from_source_agent() -> None:
    """legacy PRIVATE 的可见性 target 是来源 Agent（codec 一致语义）。"""
    result = convert_v1_memory_payload(
        _v1_payload(visibility="PRIVATE"), missing_visibility_policy="public"
    )

    assert result.atom is not None
    policy = result.atom.meta.access_policy
    assert policy.visibility == MemoryVisibility.PRIVATE
    assert policy.target_agent_id == "agent-a"
    assert policy.target_team_id is None


def test_workspace_v1_maps_to_team_policy_with_team_target() -> None:
    """legacy WORKSPACE 解释为 TEAM，target 来自历史 team_id。"""
    payload = _v1_payload(visibility="WORKSPACE")
    payload["meta"]["team_id"] = "team-a"
    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is not None
    policy = result.atom.meta.access_policy
    assert policy.visibility == MemoryVisibility.TEAM
    assert policy.target_team_id == "team-a"


def test_missing_visibility_defaults_to_public_with_explicit_count() -> None:
    """public 策略下缺失 visibility 按 PUBLIC 迁移，且必须逐条计数。"""
    result = convert_v1_memory_payload(
        _v1_payload(visibility=None), missing_visibility_policy="public"
    )

    assert result.atom is not None
    assert result.defaulted_public is True
    assert result.atom.meta.access_policy.visibility == MemoryVisibility.PUBLIC


def test_missing_visibility_fails_closed_under_fail_policy() -> None:
    """fail 策略下缺失 visibility 不允许静默放宽为 PUBLIC。"""
    result = convert_v1_memory_payload(
        _v1_payload(visibility=None), missing_visibility_policy="fail"
    )

    assert result.atom is None
    assert "fail-closed" in result.reason


def test_unknown_visibility_fails_closed() -> None:
    """未知可见性值不得猜测映射。"""
    result = convert_v1_memory_payload(
        _v1_payload(visibility="SECRET"), missing_visibility_policy="public"
    )

    assert result.atom is None
    assert "visibility" in result.reason


def test_workspace_visibility_without_team_id_fails_closed() -> None:
    """legacy WORKSPACE 缺少 team_id 时无法构造合法 TEAM 策略。"""
    result = convert_v1_memory_payload(
        _v1_payload(visibility="WORKSPACE"), missing_visibility_policy="public"
    )

    assert result.atom is None
    assert "team_id" in result.reason


def test_private_policy_target_system_is_rejected() -> None:
    """保留 system 不是可授权主体，PRIVATE target=system 必须 fail closed。"""
    payload = _v1_payload(visibility="PRIVATE")
    payload["meta"]["source_agent_id"] = SYSTEM_AGENT_ID

    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is None
    assert "system" in result.reason


# ============ Workspace 归属 fail-closed ============


def test_missing_user_id_without_projection_fails_closed() -> None:
    """无 user_id 且无投影时无法确定归属。"""
    result = convert_v1_memory_payload(
        _v1_payload(user_id=None), missing_visibility_policy="public"
    )

    assert result.atom is None
    assert "user_id" in result.reason


def test_user_id_conflicting_with_projection_fails_closed() -> None:
    """平铺 user_id 与已投影 owner 冲突时拒绝猜测。"""
    payload = _v1_payload(user_id="u2")
    payload["meta"].update(
        {
            "owner_user_id": "u1",
            "workspace_key": "main_workspace",
            "workspace_id": "main_workspace",
        }
    )
    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is None
    assert "冲突" in result.reason


def test_partial_workspace_projection_fails_closed() -> None:
    """部分 Workspace 投影拒绝猜测补齐（与 codec 一致）。"""
    payload = _v1_payload()
    payload["meta"]["owner_user_id"] = "u1"
    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is None
    assert "部分 Workspace 投影" in result.reason


def test_complete_projection_takes_precedence_over_user_id_workspace() -> None:
    """完整投影是归属权威；与 user_id 一致时按投影迁移（支持非默认 Workspace）。"""
    payload = _v1_payload(user_id="u1")
    payload["meta"].update(
        {
            "owner_user_id": "u1",
            "workspace_key": "isolation_workspace",
            "workspace_id": "isolation_workspace",
        }
    )
    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is not None
    assert result.atom.workspace_identity.workspace_id == "isolation_workspace"


def test_missing_source_agent_id_fails_closed() -> None:
    """来源 provenance 是必填字段，缺失时不得编造。"""
    payload = _v1_payload()
    payload["meta"].pop("source_agent_id")
    result = convert_v1_memory_payload(payload, missing_visibility_policy="public")

    assert result.atom is None
    assert "source_agent_id" in result.reason


# ============ legacy Artifact 判定与迁移命名空间 ============


def test_legacy_artifact_detected_by_owner_agent_id_key() -> None:
    """owner_agent_id 键（当前模型已删除）是 legacy 形状标记。"""
    assert _is_legacy_artifact({"artifact_type": "document", "owner_agent_id": "omni_doll"}) is True
    assert _is_legacy_artifact({"artifact_type": "document"}) is False


def test_legacy_interaction_detected_by_flat_turn_fields() -> None:
    """任一 turn 缺少 actor_identity 即代表旧平铺形状。"""
    legacy_turn = {"user_id": "u1", "agent_id": "a1"}
    canonical_turn = {"actor_identity": {"user_id": "u1", "agent_id": "a1"}}

    assert _is_legacy_artifact({"artifact_type": "interaction", "turns": [legacy_turn]}) is True
    assert _is_legacy_artifact({"artifact_type": "interaction", "turns": [canonical_turn]}) is False


def test_replacement_artifact_id_is_deterministic_and_namespaced() -> None:
    """同一条旧记录永远映射到同一迁移命名空间 ID（幂等性基础）。"""
    workspace = _main_workspace("u1")

    first = replacement_artifact_id(workspace, "art_old_1")
    second = replacement_artifact_id(workspace, "art_old_1")
    other = replacement_artifact_id(workspace, "art_old_2")
    other_workspace = replacement_artifact_id(_main_workspace("u2"), "art_old_1")

    assert first == second
    assert first.startswith(REPLACEMENT_ID_PREFIX)
    assert first != other
    # 同名 artifact_id 在不同 Workspace 下映射到不同 replacement。
    assert first != other_workspace


# ============ repair 模式：ref workspace 回填 ============


def _payload_with_bare_refs() -> dict:
    """refs 缺失 workspace_identity 的最早代 V1 payload（死簇形状）。"""
    payload = _v1_payload()
    payload["payload"]["artifacts"] = {
        "refs": [
            {"artifact_id": "art_old_1", "artifact_type": "memory_creation"},
            {"artifact_id": "art_old_2", "artifact_type": "interaction"},
        ],
        "events": [
            {
                "event_type": "created",
                "artifact_refs": [
                    {"artifact_id": "art_old_1", "artifact_type": "memory_creation"},
                ],
            },
        ],
    }
    return payload


def test_repair_backfills_missing_ref_workspace_from_own_identity() -> None:
    """repair 模式下缺失 workspace_identity 的 ref 用本记录归属回填。"""
    payload = _payload_with_bare_refs()
    import copy

    original = copy.deepcopy(payload)

    result = convert_v1_memory_payload(
        payload, missing_visibility_policy="public", repair_missing_ref_workspace=True
    )

    assert result.atom is not None
    assert result.backfilled_refs == 3
    workspace = _main_workspace("u1")
    for ref in result.atom.payload.artifacts.refs:
        assert ref.workspace_identity == workspace
    for event in result.atom.payload.artifacts.events:
        for ref in event.artifact_refs:
            assert ref.workspace_identity == workspace
    # 回填不得修改调用方的原始 payload（报告与重跑依赖原始数据）。
    assert payload == original


def test_without_repair_bare_refs_fail_closed() -> None:
    """未启用 repair 时，缺 workspace_identity 的 ref 维持 fail closed。"""
    result = convert_v1_memory_payload(
        _payload_with_bare_refs(), missing_visibility_policy="public"
    )

    assert result.atom is None
    assert result.backfilled_refs == 0
    assert "workspace_identity" in result.reason


def test_ref_with_workspace_identity_is_not_overwritten_by_repair() -> None:
    """已有归属的 ref 不被回填覆盖。"""
    payload = _payload_with_bare_refs()
    payload["payload"]["artifacts"]["refs"][0]["workspace_identity"] = {
        "owner_user_id": "other",
        "workspace_key": "main_workspace",
        "workspace_id": "main_workspace",
    }

    result = convert_v1_memory_payload(
        payload, missing_visibility_policy="public", repair_missing_ref_workspace=True
    )

    assert result.atom is not None
    assert result.backfilled_refs == 2
    ref = result.atom.payload.artifacts.refs[0]
    assert ref.workspace_identity.owner_user_id == "other"


# ============ repair 模式：meta 归属解析（Artifact 采纳依据） ============


def test_resolve_workspace_from_v1_meta_uses_user_id() -> None:
    """V1 meta 的 user_id 解析为 main_workspace 归属。"""
    payload = _v1_payload(user_id="u9")

    assert resolve_memory_workspace_from_meta(payload) == _main_workspace("u9")


def test_resolve_workspace_from_v2_meta_prefers_projection() -> None:
    """V2 meta（含投影）解析出投影归属，供重跑时采纳保持一致。"""
    payload = _v1_payload(user_id="u9")
    payload["meta"].update(
        {
            "schema_version": 2,
            "workspace_identity": {
                "owner_user_id": "u9",
                "workspace_key": "ws9",
                "workspace_id": "ws9",
            },
            "owner_user_id": "u9",
            "workspace_key": "ws9",
            "workspace_id": "ws9",
            "access_policy": {"visibility": "PUBLIC"},
        }
    )

    assert resolve_memory_workspace_from_meta(payload).workspace_id == "ws9"


def test_resolve_workspace_returns_none_without_evidence() -> None:
    """缺 user_id / 部分投影 / owner 冲突时均不可采纳。"""
    missing = _v1_payload(user_id=None)
    partial = _v1_payload()
    partial["meta"]["owner_user_id"] = "u1"
    conflict = _v1_payload(user_id="u2")
    conflict["meta"].update(
        {"owner_user_id": "u1", "workspace_key": "main_workspace", "workspace_id": "main_workspace"}
    )

    assert resolve_memory_workspace_from_meta(missing) is None
    assert resolve_memory_workspace_from_meta(partial) is None
    assert resolve_memory_workspace_from_meta(conflict) is None
