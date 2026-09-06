"""V1 Memory 与 Artifact legacy 迁移引擎的端到端流程测试。

验证真实协作边界：迁移引擎 + 真实 ``ArtifactStore`` /
``FilesystemArtifactStorageAdapter``（tmp_path 文件系统）+ 内存版 Qdrant
访问端口。Qdrant 与 Embedding 基础设施不在本边界内（由 fake 承载），
需要真实 Qdrant 的验证属于 ``real_infra`` 任务。
"""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope, MemoryAtom, WorkspaceIdentity
from hivememory.engines.retrieval.memory_codec import decode_memory_payload
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.stores import ArtifactStore
from hivememory.tools.v1_legacy_migration import (
    MigrationOptions,
    V1LegacyMigrator,
    scan_artifact_records,
)

WORKSPACE = WorkspaceIdentity(
    owner_user_id="u1",
    workspace_key="main_workspace",
    workspace_id="main_workspace",
)
_WS_DICT = {
    "owner_user_id": "u1",
    "workspace_key": "main_workspace",
    "workspace_id": "main_workspace",
}


def _artifact_ref_dict(artifact_id: str, artifact_type: str) -> dict[str, Any]:
    """旧 ArtifactRef 的 JSON 形状（与历史 store.put 返回一致）。"""
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "workspace_identity": dict(_WS_DICT),
        "uri": f"legacy://{artifact_id}",
        "sha256": "legacy-hash",
        "created_at": "2026-06-14T12:00:00",
        "summary": "",
    }


def _legacy_interaction_artifact(artifact_id: str, *, bad_turn: bool = False) -> dict:
    turns = [
        {
            "block_id": "b1",
            "turn_id": "t1",
            "created_at": 1.0,
            "user_id": "u1",
            "agent_id": "agent-a",
            "team_id": None,
            "user_query": "q1",
            "assistant_final_text": "a1",
        },
        {
            "block_id": "b2",
            "turn_id": "t2",
            "user_id": "" if bad_turn else "u1",
            "agent_id": "",
            "team_id": None,
            "user_query": "q2",
            "assistant_final_text": "a2",
        },
    ]
    return {
        "artifact_id": artifact_id,
        "artifact_type": "interaction",
        "schema_version": "1",
        "created_at": "2026-06-14T12:00:00",
        "owner_agent_id": "omni_doll",
        "workspace_identity": dict(_WS_DICT),
        "title": "",
        "summary": "",
        "topic_id": "topic-1",
        "topic_title": "历史话题",
        "topic_summary": "",
        "turns": turns,
        "captured_at": "2026-06-14T12:00:00",
    }


def _legacy_memory_version_artifact(
    artifact_id: str, memory_id: str, interaction_id: str
) -> dict:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "memory_version",
        "schema_version": "1",
        "created_at": "2026-06-14T12:00:00",
        "owner_agent_id": "agent-a",
        "workspace_identity": dict(_WS_DICT),
        "title": "",
        "summary": "",
        "memory_id": memory_id,
        "version_number": 1,
        "update_source": "CREATE",
        "snapshot_before": None,
        "snapshot_after": {"content": "legacy content", "tags": []},
        "changed_at": "2026-06-14T12:00:00",
        "source_artifacts": [_artifact_ref_dict(interaction_id, "interaction")],
        "source_memory_refs": [],
    }


def _legacy_memory_creation_artifact(
    artifact_id: str,
    memory_id: str,
    interaction_id: str,
    version_id: str,
) -> dict:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "memory_creation",
        "schema_version": "1",
        "created_at": "2026-06-14T12:00:00",
        "owner_agent_id": "agent-a",
        "workspace_identity": dict(_WS_DICT),
        "title": "",
        "summary": "",
        "memory_id": memory_id,
        "source_intent": "WRITE",
        "generation_view": {},
        "source_artifacts": [_artifact_ref_dict(interaction_id, "interaction")],
        "source_memory_refs": [],
        "initial_version_ref": _artifact_ref_dict(version_id, "memory_version"),
    }


def _v1_memory_payload(memory_id: str, creation_id: str) -> dict:
    return {
        "id": memory_id,
        "meta": {
            "source_agent_id": "agent-a",
            "user_id": "u1",
            "team_id": None,
            "visibility": "PUBLIC",
            "created_at": "2026-06-14T12:00:00",
            "version": 1,
        },
        "index": {
            "title": "Legacy memory",
            "summary": "Legacy record used to verify migration flow.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {
            "content": "legacy content",
            "artifacts": {
                "refs": [_artifact_ref_dict(creation_id, "memory_creation")],
                "events": [],
            },
        },
        "relations": {"relates_to": [], "supersedes": [], "depends_on": []},
    }


def _v2_memory_payload(memory_id: str, ref_id: str, ref_type: str) -> dict:
    """schema v2 payload（含投影字段，同 to_qdrant_payload 形状）。"""
    return {
        "schema_version": 2,
        "id": memory_id,
        "meta": {
            "workspace_identity": dict(_WS_DICT),
            "source_agent_id": "agent-a",
            "contributing_agent_ids": [],
            "access_policy": {"visibility": "PUBLIC"},
            "created_at": "2026-06-14T12:00:00",
            "version": 1,
            "owner_user_id": "u1",
            "workspace_key": "main_workspace",
            "workspace_id": "main_workspace",
        },
        "index": {
            "title": "V2 memory",
            "summary": "V2 record whose refs still point to legacy artifacts.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {
            "content": "v2 content",
            "artifacts": {"refs": [_artifact_ref_dict(ref_id, ref_type)], "events": []},
        },
        "relations": {"relates_to": [], "supersedes": [], "depends_on": []},
    }


class FakeMemoryAccess:
    """内存版 MemoryMigrationAccess：raw payload 字典模拟 Qdrant 点位。"""

    def __init__(self, points: dict[str, dict] | None = None) -> None:
        self.points: dict[str, dict] = dict(points or {})
        self.published: list[tuple[str, MemoryAtom]] = []

    async def iter_raw_points(self, batch_size: int):
        for point_id, payload in list(self.points.items()):
            yield point_id, payload

    async def replace_point(self, memory: MemoryAtom, *, previous_point_id: str) -> None:
        self.published.append((previous_point_id, memory))
        self.points.pop(previous_point_id, None)
        # 模拟 canonical 发布：V2 payload 出现在存储中，旧点消失。
        self.points[f"canonical::{memory.id}"] = memory.to_qdrant_payload()


def _write_legacy_file(root: Path, artifact_id: str, raw: dict) -> Path:
    path = root / "legacy" / f"{artifact_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def _management_scope() -> IdentityScope:
    return IdentityScope(
        actor_identity=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
        workspace_identity=WORKSPACE,
    )


def _default_user_scope() -> IdentityScope:
    """死簇夹具归属 default 用户的 main_workspace，读取须用对应管理作用域。"""
    return IdentityScope(
        actor_identity=ActorIdentity(user_id="default", agent_id=SYSTEM_AGENT_ID),
        workspace_identity=WorkspaceIdentity(
            owner_user_id="default",
            workspace_key="main_workspace",
            workspace_id="main_workspace",
        ),
    )


def _new_id_by_old(report: Any) -> dict[str, str]:
    return {
        entry["old_artifact_id"]: entry["new_artifact_id"]
        for entry in report.artifact_mappings
    }


def _count(report: Any, name: str) -> int:
    """counts 字典只包含发生过事件的键，未发生的计数为 0。"""
    return report.counts.get(name, 0)


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def artifact_store(artifact_root: Path) -> ArtifactStore:
    return ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(artifact_root)))


def _build_migrator(
    artifact_store: ArtifactStore,
    artifact_root: Path,
    access: FakeMemoryAccess,
    *,
    dry_run: bool,
    checkpoint_path: Path | None = None,
    repair: bool = False,
) -> V1LegacyMigrator:
    return V1LegacyMigrator(
        artifact_store=artifact_store,
        artifacts_root=artifact_root,
        memory_access=access,
        options=MigrationOptions(dry_run=dry_run, repair_legacy_ownership=repair),
        checkpoint_path=checkpoint_path,
    )


# ============ 主流程 ============

@pytest.mark.asyncio
async def test_apply_migrates_legacy_artifacts_and_v1_memory_with_ref_rewrite(
    artifact_store, artifact_root, tmp_path
) -> None:
    """legacy 图谱（interaction/version/creation + V1 Memory）整体迁移到 canonical。"""
    interaction_id, version_id, creation_id = "art_old_i1", "art_old_v1", "art_old_c1"
    memory_id = str(uuid4())
    _write_legacy_file(
        artifact_root, interaction_id, _legacy_interaction_artifact(interaction_id)
    )
    _write_legacy_file(
        artifact_root,
        version_id,
        _legacy_memory_version_artifact(version_id, memory_id, interaction_id),
    )
    _write_legacy_file(
        artifact_root,
        creation_id,
        _legacy_memory_creation_artifact(creation_id, memory_id, interaction_id, version_id),
    )
    legacy_bytes = (artifact_root / "legacy" / f"{creation_id}.json").read_bytes()
    access = FakeMemoryAccess({"legacy-point": _v1_memory_payload(memory_id, creation_id)})

    migrator = _build_migrator(
        artifact_store,
        artifact_root,
        access,
        dry_run=False,
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    report = await migrator.run()

    assert _count(report, "artifact_replaced") == 3
    assert _count(report, "memory_v1_migrated") == 1
    assert _count(report, "memory_failed") == 0
    assert report.diagnostics == []

    # 旧新 ID 映射完整（superseded 记录在报告中）。
    mapping = _new_id_by_old(report)
    assert set(mapping) == {interaction_id, version_id, creation_id}
    assert all(new_id != old_id for old_id, new_id in mapping.items())
    # 旧记录 append-only：文件内容保持原样。
    assert (artifact_root / "legacy" / f"{creation_id}.json").read_bytes() == legacy_bytes

    scope = _management_scope()
    interaction_new, version_new, creation_new = (
        mapping[interaction_id],
        mapping[version_id],
        mapping[creation_id],
    )

    interaction_data = await artifact_store.get(scope, interaction_new)
    assert "owner_agent_id" not in interaction_data
    assert interaction_data["turns"][0]["actor_identity"]["agent_id"] == "agent-a"
    # 缺具体 Agent 的 turn 使用保留 system，不回落到真实 Agent。
    assert interaction_data["turns"][1]["actor_identity"]["agent_id"] == SYSTEM_AGENT_ID
    assert interaction_data["turns"][1]["actor_identity"]["user_id"] == "u1"

    version_data = await artifact_store.get(scope, version_new)
    assert "owner_agent_id" not in version_data
    # owner_agent_id 是旧 builder 从 memory.meta.source_agent_id 原样复制的
    # source 载体，可迁移为 source_agent_id；贡献者不可证明时保持空。
    assert version_data["source_agent_id"] == "agent-a"
    assert version_data["contributing_agent_ids"] == []
    assert version_data["source_artifacts"][0]["artifact_id"] == interaction_new

    creation_data = await artifact_store.get(scope, creation_new)
    assert "owner_agent_id" not in creation_data
    assert creation_data["source_agent_id"] == "agent-a"
    # 贡献者从关联 InteractionArtifact 的 turn 聚合（system 不算贡献者）。
    assert creation_data["contributing_agent_ids"] == ["agent-a"]
    assert creation_data["initial_version_ref"]["artifact_id"] == version_new
    assert creation_data["source_artifacts"][0]["artifact_id"] == interaction_new

    # V1 Memory → V2：归属、策略收敛，引用链重写到 replacement。
    assert len(access.published) == 1
    atom = access.published[0][1]
    assert atom.schema_version == 2
    assert atom.workspace_identity == WORKSPACE
    assert atom.meta.access_policy.visibility.value == "PUBLIC"
    assert "user_id" not in atom.meta.model_dump()
    assert atom.payload.artifacts.refs[0].artifact_id == creation_new


@pytest.mark.asyncio
async def test_apply_rerun_is_idempotent_via_checkpoint(
    artifact_store, artifact_root, tmp_path
) -> None:
    """重跑按 checkpoint 与内容 hash 幂等跳过，不产生重复 replacement。"""
    interaction_id = "art_old_i1"
    memory_id = str(uuid4())
    _write_legacy_file(
        artifact_root, interaction_id, _legacy_interaction_artifact(interaction_id)
    )
    access = FakeMemoryAccess({"legacy-point": _v1_memory_payload(memory_id, "art_old_c1")})
    checkpoint_path = tmp_path / "checkpoint.json"

    first = await _build_migrator(
        artifact_store, artifact_root, access,
        dry_run=False, checkpoint_path=checkpoint_path,
    ).run()
    first_mapping = _new_id_by_old(first)
    assert _count(first, "artifact_replaced") == 1

    second = await _build_migrator(
        artifact_store, artifact_root, access,
        dry_run=False, checkpoint_path=checkpoint_path,
    ).run()

    assert _count(second, "artifact_replaced") == 0
    assert _count(second, "artifact_replacement_resumed") == 1
    assert _count(second, "memory_v1_found") == 0
    assert _count(second, "memory_v2_skipped_already_canonical") == 1
    assert len(access.published) == 1  # 第二轮没有新的 Memory 发布
    assert _new_id_by_old(second)[interaction_id] == first_mapping[interaction_id]


@pytest.mark.asyncio
async def test_dry_run_reports_plan_without_writing_anything(
    artifact_store, artifact_root
) -> None:
    """dry-run 只扫描、转换与计数；不写 replacement、不改 Memory。"""
    interaction_id = "art_old_i1"
    memory_id = str(uuid4())
    _write_legacy_file(
        artifact_root, interaction_id, _legacy_interaction_artifact(interaction_id)
    )
    files_before = sorted(artifact_root.rglob("*.json"))
    access = FakeMemoryAccess({"legacy-point": _v1_memory_payload(memory_id, "art_old_c1")})

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=True
    ).run()

    assert _count(report, "artifact_replacement_planned") == 1
    assert _count(report, "memory_v1_would_migrate") == 1
    assert access.published == []
    assert sorted(artifact_root.rglob("*.json")) == files_before
    assert not list(artifact_root.rglob("*art_mig2_*"))


# ============ fail-closed 与悬空引用 ============

@pytest.mark.asyncio
async def test_interaction_turn_without_user_fails_closed_and_blocks_consumers(
    artifact_store, artifact_root
) -> None:
    """缺用户归属的 turn 拒绝转换；引用它的 Memory 也必须 fail closed。"""
    interaction_id = "art_old_bad"
    memory_id = str(uuid4())
    _write_legacy_file(
        artifact_root,
        interaction_id,
        _legacy_interaction_artifact(interaction_id, bad_turn=True),
    )
    access = FakeMemoryAccess({"legacy-point": _v1_memory_payload(memory_id, interaction_id)})

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False
    ).run()

    assert _count(report, "artifact_replaced") == 0
    assert _count(report, "memory_failed") == 1
    artifact_reasons = [d["reason"] for d in report.diagnostics
                        if d["resource_type"] == "artifact"]
    memory_reasons = [d["reason"] for d in report.diagnostics
                      if d["resource_type"] == "memory"]
    assert any("user_id" in reason for reason in artifact_reasons)
    assert any("引用链无法完整重写" in reason for reason in memory_reasons)
    assert access.published == []
    # 失败记录没有写入任何 canonical replacement 文件。
    replacements = [
        record for record in scan_artifact_records(artifact_root)
        if record.artifact_id.startswith("art_mig2_")
    ]
    assert replacements == []


@pytest.mark.asyncio
async def test_v1_memory_without_user_id_fails_closed_and_keeps_point(
    artifact_store, artifact_root
) -> None:
    """无归属的 V1 Memory 不迁移，原始 point 保持 V1 形状。"""
    payload = _v1_memory_payload(str(uuid4()), "art_old_c1")
    payload["meta"].pop("user_id")
    access = FakeMemoryAccess({"legacy-point": payload})

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False
    ).run()

    assert _count(report, "memory_failed") == 1
    assert _count(report, "memory_v1_migrated") == 0
    assert access.published == []
    # 原始 point 未被改动：仍是缺少 schema_version 的 V1 payload。
    assert access.points["legacy-point"].get("schema_version") is None
    assert "workspace_identity" not in access.points["legacy-point"]["meta"]


@pytest.mark.asyncio
async def test_dangling_ref_is_kept_and_reported(
    artifact_store, artifact_root
) -> None:
    """指向不存在 Artifact 的悬空引用原样保留并计数，不阻断迁移。"""
    memory_id = str(uuid4())
    payload = _v1_memory_payload(memory_id, "art_nonexistent")
    access = FakeMemoryAccess({"legacy-point": payload})

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False
    ).run()

    assert _count(report, "memory_v1_migrated") == 1
    assert len(report.unresolved_refs) == 1
    assert report.unresolved_refs[0]["ref_artifact_id"] == "art_nonexistent"
    assert access.published[0][1].payload.artifacts.refs[0].artifact_id == "art_nonexistent"


# ============ V2 记录的引用重写 ============

@pytest.mark.asyncio
async def test_v2_memory_with_legacy_refs_is_republished_with_replacements(
    artifact_store, artifact_root
) -> None:
    """迁移窗口期写入的 V2 Memory 若仍引用 legacy Artifact，需重写引用后重发布。"""
    interaction_id = "art_old_i1"
    memory_id = str(uuid4())
    _write_legacy_file(
        artifact_root, interaction_id, _legacy_interaction_artifact(interaction_id)
    )
    access = FakeMemoryAccess(
        {"v2-point": _v2_memory_payload(memory_id, interaction_id, "interaction")}
    )

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False
    ).run()

    assert _count(report, "memory_v2_refs_rewritten") == 1
    interaction_new = _new_id_by_old(report)[interaction_id]
    atom = access.published[0][1]
    assert atom.payload.artifacts.refs[0].artifact_id == interaction_new
    # 重发布 payload 仍能被运行时 codec 解码（canonical 契约）。
    republished = access.points[f"canonical::{memory_id}"]
    assert decode_memory_payload(republished).id == atom.id


@pytest.mark.asyncio
async def test_v2_memory_without_legacy_refs_is_left_untouched(
    artifact_store, artifact_root
) -> None:
    """引用链已是 canonical 的 V2 记录跳过，不做无意义重发布。"""
    memory_id = str(uuid4())
    payload = _v2_memory_payload(memory_id, "art_canonical_1", "document")
    access = FakeMemoryAccess({"v2-point": payload})

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False
    ).run()

    assert _count(report, "memory_v2_skipped_already_canonical") == 1
    assert access.published == []
    # 原始 point 未被改动（仍是原始 payload 对象内容）。
    assert access.points["v2-point"]["meta"]["access_policy"]["visibility"] == "PUBLIC"


# ============ repair 模式：v0.5 早期死簇 ============

def _bare_ref(artifact_id: str, artifact_type: str) -> dict[str, Any]:
    """v0.5 时代内部 ref：没有 workspace_identity 字段。"""
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "uri": f"legacy://{artifact_id}",
        "sha256": "legacy-hash",
        "created_at": "2026-07-19T10:00:00",
        "summary": "",
    }


def _v05_interaction_artifact(
    artifact_id: str, *, turn_user_ids: tuple[str, ...] = ("default",)
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "interaction",
        "schema_version": "1",
        "created_at": "2026-07-21T10:00:00",
        "owner_agent_id": "",
        "owner_user_id": "",
        "content_hash": "legacy-hash",
        "title": "",
        "summary": "",
        "topic_id": "topic-v05",
        "topic_title": "v0.5 话题",
        "topic_summary": "",
        "turns": [
            {
                "block_id": f"b{index}",
                "turn_id": f"t{index}",
                "user_id": user_id,
                "agent_id": "meal_assistant",
                "team_id": None,
                "user_query": "q",
                "assistant_final_text": "a",
            }
            for index, user_id in enumerate(turn_user_ids)
        ],
        "captured_at": "2026-07-21T10:00:00",
    }


def _v05_memory_creation_artifact(
    artifact_id: str,
    memory_id: str,
    *,
    interaction_id: str | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "memory_creation",
        "schema_version": "1",
        "created_at": "2026-07-19T10:00:00",
        "owner_agent_id": "",
        "owner_user_id": "",
        "content_hash": "legacy-hash",
        "title": "",
        "summary": "",
        "memory_id": memory_id,
        "source_intent": "WRITE",
        "generation_view": {},
        "source_artifacts": [_bare_ref(interaction_id, "interaction")] if interaction_id else [],
        "source_memory_refs": [],
        "initial_version_ref": _bare_ref(version_id, "memory_version") if version_id else None,
    }


def _v05_memory_version_artifact(
    artifact_id: str,
    memory_id: str,
    *,
    update_source: str = "CREATE",
    interaction_id: str | None = None,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "memory_version",
        "schema_version": "1",
        "created_at": "2026-07-19T10:00:00",
        "owner_agent_id": "",
        "owner_user_id": "",
        "content_hash": "legacy-hash",
        "title": "",
        "summary": "",
        "memory_id": memory_id,
        "version_number": 1,
        "update_source": update_source,
        "snapshot_before": None,
        "snapshot_after": {"content": "v0.5 content", "tags": []},
        "changed_at": "2026-07-19T10:00:00",
        "source_artifacts": [_bare_ref(interaction_id, "interaction")] if interaction_id else [],
        "source_memory_refs": [],
    }


def _v05_memory_payload(
    memory_id: str,
    ref_ids: list[tuple[str, str]],
    event_ref_ids: list[tuple[str, str]],
    *,
    source_agent_id: str = "meal_assistant",
) -> dict[str, Any]:
    """refs 缺失 workspace_identity 的最早代 V1 Memory payload。"""
    return {
        "id": memory_id,
        "meta": {
            "source_agent_id": source_agent_id,
            "user_id": "default",
            "team_id": None,
            "visibility": "PUBLIC",
            "created_at": "2026-07-19T10:00:00",
            "version": 1,
        },
        "index": {
            "title": "V05 memory",
            "summary": "V0.5 era memory whose refs carry no workspace.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {
            "content": "v0.5 content",
            "artifacts": {
                "refs": [_bare_ref(a, t) for a, t in ref_ids],
                "events": [
                    {
                        "event_type": "created",
                        "at": "2026-07-19T10:00:00",
                        "artifact_refs": [_bare_ref(a, t) for a, t in event_ref_ids],
                    }
                ],
            },
        },
        "relations": {"relates_to": [], "supersedes": [], "depends_on": []},
    }


def _build_v05_dead_cluster(
    artifact_root: Path, access: FakeMemoryAccess
) -> dict[str, str]:
    """按真实 dry-run 死簇的闭合关系构建 6 Artifact + 2 Memory 夹具。"""
    m1, m2 = str(uuid4()), str(uuid4())
    interaction_id = "art_v05_i1"
    c1, v1 = "art_v05_c1", "art_v05_v1"
    c2, v2a, v2b = "art_v05_c2", "art_v05_v2a", "art_v05_v2b"

    _write_legacy_file(artifact_root, interaction_id, _v05_interaction_artifact(interaction_id))
    _write_legacy_file(
        artifact_root, c1,
        _v05_memory_creation_artifact(c1, m1, interaction_id=interaction_id, version_id=v1),
    )
    _write_legacy_file(artifact_root, v1, _v05_memory_version_artifact(v1, m1))
    _write_legacy_file(
        artifact_root, c2,
        _v05_memory_creation_artifact(c2, m2, version_id=v2a),
    )
    _write_legacy_file(artifact_root, v2a, _v05_memory_version_artifact(v2a, m2))
    _write_legacy_file(
        artifact_root, v2b,
        _v05_memory_version_artifact(v2b, m2, update_source="MANUAL_EDIT"),
    )

    # M1：refs/events 指向自己的 creation/version/interaction（refs 无归属）。
    access.points["v05-point-1"] = _v05_memory_payload(
        m1,
        [(v1, "memory_version"), (c1, "memory_creation"), (interaction_id, "interaction")],
        [(v1, "memory_version"), (c1, "memory_creation")],
    )
    # M2：refs/events 指向自己的 creation 与两个 version。
    access.points["v05-point-2"] = _v05_memory_payload(
        m2,
        [(v2a, "memory_version"), (c2, "memory_creation"), (v2b, "memory_version")],
        [(v2a, "memory_version"), (c2, "memory_creation"), (v2b, "memory_version")],
        source_agent_id="ui",
    )
    return {"m1": m1, "m2": m2, "interaction_id": interaction_id, "c1": c1, "c2": c2}


@pytest.mark.asyncio
async def test_repair_mode_migrates_v05_dead_cluster_end_to_end(
    artifact_store, artifact_root
) -> None:
    """repair 模式：归属采纳 + source 缺证默认 omni_doll + ref 回填全链路生效。"""
    access = FakeMemoryAccess()
    cluster = _build_v05_dead_cluster(artifact_root, access)

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False, repair=True
    ).run()

    assert _count(report, "artifact_replaced") == 6
    assert _count(report, "memory_v1_migrated") == 2
    assert _count(report, "memory_failed") == 0
    assert _count(report, "artifact_workspace_adopted") == 6
    # 5 个 memory 类 Artifact 的来源缺证（interaction 无 source 字段）。
    assert _count(report, "artifact_source_defaulted_omni_doll") == 5
    # M1: 3 refs + 2 event refs；M2: 3 + 3。
    assert _count(report, "memory_refs_workspace_backfilled") == 11
    repair_rules = {entry["rule"] for entry in report.repairs}
    assert {
        "artifact_workspace_from_memory",
        "artifact_workspace_from_interaction_turns",
        "artifact_source_defaulted",
        "memory_ref_workspace_backfilled",
    } <= repair_rules

    mapping = _new_id_by_old(report)
    scope = _default_user_scope()
    creation_data = await artifact_store.get(scope, mapping[cluster["c1"]])
    # 来源缺证按用户批准的默认归属填入 omni_doll。
    assert creation_data["source_agent_id"] == "omni_doll"
    # 贡献者仍只从关联 interaction 的 turn 聚合（meal_assistant）。
    assert creation_data["contributing_agent_ids"] == ["meal_assistant"]
    assert "owner_agent_id" not in creation_data
    assert creation_data["workspace_identity"]["owner_user_id"] == "default"

    other_creation = await artifact_store.get(scope, mapping[cluster["c2"]])
    assert other_creation["source_agent_id"] == "omni_doll"
    assert other_creation["contributing_agent_ids"] == []

    interaction_data = await artifact_store.get(scope, mapping[cluster["interaction_id"]])
    assert interaction_data["turns"][0]["actor_identity"]["agent_id"] == "meal_assistant"

    # 两条 Memory 的 refs/events 全部重写到 replacement 并可被 codec 解码。
    published = {previous_id: atom for previous_id, atom in access.published}
    atom1 = published["v05-point-1"]
    assert [ref.artifact_id for ref in atom1.payload.artifacts.refs] == [
        mapping["art_v05_v1"], mapping["art_v05_c1"], mapping[cluster["interaction_id"]],
    ]
    assert atom1.payload.artifacts.events[0].artifact_refs[0].artifact_id == mapping["art_v05_v1"]
    assert atom1.workspace_identity.owner_user_id == "default"
    atom2 = published["v05-point-2"]
    assert [ref.artifact_id for ref in atom2.payload.artifacts.refs] == [
        mapping["art_v05_v2a"], mapping["art_v05_c2"], mapping["art_v05_v2b"],
    ]
    for atom in (atom1, atom2):
        decoded = decode_memory_payload(atom.to_qdrant_payload())
        assert decoded.workspace_identity == atom.workspace_identity


@pytest.mark.asyncio
async def test_repair_mode_off_keeps_v05_dead_cluster_fail_closed(
    artifact_store, artifact_root
) -> None:
    """未启用 repair 时死簇维持 fail closed：6 个 Artifact 不可归一化，2 条 Memory 校验失败。"""
    access = FakeMemoryAccess()
    _build_v05_dead_cluster(artifact_root, access)

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False, repair=False
    ).run()

    assert _count(report, "artifact_files_unreadable") == 6
    assert _count(report, "artifact_replaced") == 0
    assert _count(report, "memory_failed") == 2
    assert _count(report, "memory_v1_migrated") == 0
    assert report.repairs == []
    assert access.published == []


@pytest.mark.asyncio
async def test_repair_mode_skips_interaction_with_inconsistent_turn_owners(
    artifact_store, artifact_root
) -> None:
    """interaction 各 turn 的 user_id 不一致时无采纳证据，维持 fail closed。"""
    _write_legacy_file(
        artifact_root,
        "art_v05_bad_i",
        _v05_interaction_artifact("art_v05_bad_i", turn_user_ids=("default", "someone_else")),
    )
    access = FakeMemoryAccess()

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=False, repair=True
    ).run()

    assert _count(report, "artifact_workspace_adopted") == 0
    assert _count(report, "artifact_replaced") == 0
    assert any(
        d["resource_id"] == "art_v05_bad_i" and "无法归一化" in d["reason"]
        for d in report.diagnostics
    )


@pytest.mark.asyncio
async def test_repair_mode_dry_run_plans_dead_cluster_without_writing(
    artifact_store, artifact_root
) -> None:
    """repair + dry-run：预测全量迁移计划，不写任何存储。"""
    access = FakeMemoryAccess()
    _build_v05_dead_cluster(artifact_root, access)
    files_before = sorted(artifact_root.rglob("*.json"))

    report = await _build_migrator(
        artifact_store, artifact_root, access, dry_run=True, repair=True
    ).run()

    assert _count(report, "artifact_replacement_planned") == 6
    assert _count(report, "memory_v1_would_migrate") == 2
    assert _count(report, "artifact_workspace_adopted") == 6
    assert _count(report, "artifact_source_defaulted_omni_doll") == 5
    assert access.published == []
    assert sorted(artifact_root.rglob("*.json")) == files_before
