"""V1 Memory 与 Artifact legacy 数据迁移引擎。

对应 docs/plans/v0.6.2-v1-memory-legacy-migration.md，是一次性维护工具，
不属于任何请求链路；CLI 入口见 ``scripts/migrate_v1_memory_and_artifacts.py``。

迁移目标（Plan §2.1）：

1. 把 Qdrant 中全部 V1 Memory payload 迁移为 canonical schema v2（补齐
   ``workspace_identity`` / ``access_policy``，移除平铺 ``meta.user_id`` 权威语义）；
2. 把 ArtifactStore 中 legacy Artifact 转换为当前模型形状的 canonical replacement
   （append-only：写入新记录并保留旧记录，不改写、不删除旧文件）；
3. fail closed：无法安全推断归属、策略或 provenance 的记录不写入，进入诊断清单；
4. 输出结构化迁移报告（计数、旧新 ID 映射、逐条原因、ref 重写摘要）。

实现前冻结的迁移契约（Plan §4.2 要求，不得在运行中变更）：

- **迁移命名空间**：canonical replacement 使用确定性新 ID
  ``art_mig2_<sha256(owner|workspace_id|old_artifact_id)[:32]>``；同一条旧记录
  永远映射到同一个新 ID，是 checkpoint/resume 幂等性的基础；
- **canonical Artifact schema_version**：使用当前 Artifact 模型默认值 ``"1"``，
  与 Memory ``schema_version=2`` 是相互独立的版本轴，不得混用；
- **旧记录审计标记**：旧 Artifact 文件保持原样（append-only），superseded 状态
  只记录在迁移报告与 checkpoint 的映射中，不回写旧记录；
- **缺失 visibility 的默认策略**：作为显式脚本选项（``public`` / ``fail``），
  选择 ``public`` 时逐条计入报告，不静默放宽可见性；
- **修复策略（显式启用 ``repair_legacy_ownership``，未启用维持 fail closed）**：
  针对 v0.5 早期"死簇"记录（Artifact ``owner_user_id`` 为空串、V1 Memory
  refs 缺 ``workspace_identity``）允许三类基于证据的修复，全部逐条写入
  报告 ``repairs`` 清单：
  1. Artifact 归属采纳 —— memory 类 Artifact 从 ``memory_id`` 关联到的
     Memory 解析 Workspace；interaction 在所有 turn 的 ``user_id`` 一致时
     采纳该归属；
  2. 来源缺证默认 —— memory 类 Artifact 无法确定 ``source_agent_id`` 时，
     按用户批准的默认归属填入 ``omni_doll``（``DEFAULT_AGENT_ID``；仅作
     provenance 展示，不参与授权）；
  3. ref Workspace 回填 —— V1 Memory 与 Artifact 内部 ref 缺
     ``workspace_identity`` 时，用记录自身（或采纳后的）Workspace 回填。

写入顺序遵循 Plan §4.3：先完成 Artifact replacement 并校验 hash/ref 可解析，
再重写并发布 Memory canonical 记录；Qdrant 与 ArtifactStore 之间没有跨存储
事务，"切换"由确定性 ID、checkpoint 映射表和可恢复的发布顺序保证。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol
from uuid import uuid4

from hivememory.core.constants import DEFAULT_AGENT_ID, SYSTEM_AGENT_ID
from hivememory.core.models import (
    MAIN_WORKSPACE_ID,
    ActorIdentity,
    IdentityScope,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryVisibility,
    MetaData,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import (
    ArtifactRef,
    DocumentArtifact,
    InteractionArtifact,
    MemoryCreationArtifact,
    MemoryVersionArtifact,
)
from hivememory.engines.retrieval.memory_codec import (
    MemoryDecodeError,
    decode_memory_payload,
)
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.stores import ArtifactStore

if TYPE_CHECKING:
    from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
    from hivememory.system.config.patchouli import QdrantConfig

logger = logging.getLogger(__name__)

# ============ 冻结的迁移契约常量 ============

REPLACEMENT_ID_PREFIX = "art_mig2_"
"""迁移命名空间前缀；与确定性摘要一起构成 canonical replacement 的 artifact_id。"""

ARTIFACT_SCHEMA_VERSION = "1"
"""canonical replacement 使用的 Artifact schema_version（当前模型默认值）。

注意：这是 Artifact 自身的版本轴，与 Memory ``schema_version=2`` 无关。
"""

CHECKPOINT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

_WORKSPACE_PROJECTION_FIELDS = ("owner_user_id", "workspace_key", "workspace_id")
_V1_META_FIELDS_TO_DROP = (*_WORKSPACE_PROJECTION_FIELDS, "user_id", "team_id", "visibility")

# Artifact replacement 的发布批次：引用链从叶子指向根，先发布被引用者。
# interaction/document 不引用其他 Artifact；memory_version 可能引用前者；
# memory_creation 通过 initial_version_ref 引用 memory_version。
_ARTIFACT_PUBLISH_ROUNDS: tuple[tuple[str, ...], ...] = (
    ("interaction", "document"),
    ("memory_version",),
    ("memory_creation",),
)

_CHECKPOINT_SAVE_INTERVAL = 25


class MigrationRejected(Exception):
    """单条记录无法安全迁移（fail closed），原因进入诊断清单。"""


# ============ 运行选项 / 报告 / checkpoint ============

@dataclass
class MigrationOptions:
    """迁移执行选项；dry_run=True 时只扫描、转换与计数，不写入任何存储。"""

    missing_visibility_policy: Literal["public", "fail"] = "public"
    """V1 Memory 缺失 ``meta.visibility`` 时的显式策略：

    - ``public``：与当前 codec 兼容默认一致，按 PUBLIC 迁移并逐条计数；
    - ``fail``：证据不足时 fail closed，进入诊断清单。
    """

    dry_run: bool = True
    batch_size: int = 200
    repair_legacy_ownership: bool = False
    """是否启用针对 v0.5 早期"死簇"的修复策略（见模块 docstring 的冻结契约）。

    修复包含三类显式规则：Artifact 归属采纳、source 缺证默认 omni_doll、
    ref 缺 workspace_identity 回填。全部逐条计入报告 ``repairs`` 清单；
    未启用时这些记录维持 fail closed。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_visibility_policy": self.missing_visibility_policy,
            "dry_run": self.dry_run,
            "batch_size": self.batch_size,
            "repair_legacy_ownership": self.repair_legacy_ownership,
        }


@dataclass
class MigrationReport:
    """结构化迁移报告（Plan §2.1-4 / §4.3-5）。"""

    options: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    artifact_mappings: list[dict[str, Any]] = field(default_factory=list)
    memory_mappings: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    unresolved_refs: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)

    def inc(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount

    def add_repair(
        self,
        *,
        resource_type: str,
        resource_id: str,
        rule: str,
        detail: str,
    ) -> None:
        """记录一条已应用的修复策略（repair 模式的逐条审计要求）。"""
        self.repairs.append(
            {
                "resource_type": resource_type,
                "resource_id": resource_id,
                "rule": rule,
                "detail": detail,
            }
        )

    def add_diagnostic(
        self,
        *,
        resource_type: str,
        resource_id: str,
        location: str,
        reason: str,
        partial_replacement: bool = False,
    ) -> None:
        """记录一条 fail-closed / 失败诊断（Plan §4.3-3 的逐条原因要求）。"""
        self.diagnostics.append(
            {
                "resource_type": resource_type,
                "resource_id": resource_id,
                "location": location,
                "reason": reason,
                "partial_replacement": partial_replacement,
            }
        )

    def add_unresolved_ref(
        self,
        *,
        record_type: str,
        record_id: str,
        ref_artifact_id: str,
        reason: str,
    ) -> None:
        """记录一个保留原样的悬空引用（迁移前已损坏，不在迁移中修复或扩大）。"""
        self.unresolved_refs.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "ref_artifact_id": ref_artifact_id,
                "reason": reason,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "options": dict(self.options),
            "counts": dict(self.counts),
            "artifact_mappings": self.artifact_mappings,
            "memory_mappings": self.memory_mappings,
            "diagnostics": self.diagnostics,
            "unresolved_refs": self.unresolved_refs,
            "repairs": self.repairs,
        }


class MigrationCheckpoint:
    """迁移进度持久化（Plan §4.3-4）。

    只记录"已完成"的记录：Artifact replacement（含旧内容 hash 与 canonical
    ref），Memory 迁移/重写。失败记录不落 checkpoint，重跑时自动重试；
    已完成记录靠确定性 ID 与内容 hash 幂等跳过，不产生重复 replacement。
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._memories: dict[str, dict[str, Any]] = {}
        self._dirty = 0

    def load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("迁移 checkpoint 版本不受支持，请人工确认后删除旧 checkpoint")
        self._artifacts = raw.get("artifacts", {})
        self._memories = raw.get("memories", {})

    # ---- Artifact ----

    @staticmethod
    def artifact_key(workspace: WorkspaceIdentity, artifact_id: str) -> str:
        return f"{workspace.owner_user_id}|{workspace.workspace_id}|{artifact_id}"

    def artifact_done(self, key: str, old_content_hash: str) -> Optional[dict[str, Any]]:
        """返回已完成的 replacement 记录；旧内容 hash 不一致时视为异常返回 None。"""
        entry = self._artifacts.get(key)
        if entry is None:
            return None
        if entry.get("old_content_hash") != old_content_hash:
            return None
        return entry

    def artifact_hash_mismatch(self, key: str, old_content_hash: str) -> bool:
        entry = self._artifacts.get(key)
        return entry is not None and entry.get("old_content_hash") != old_content_hash

    def mark_artifact(
        self,
        key: str,
        *,
        old_content_hash: str,
        new_artifact_id: str,
        ref: dict[str, Any],
    ) -> None:
        self._artifacts[key] = {
            "old_content_hash": old_content_hash,
            "new_artifact_id": new_artifact_id,
            "ref": ref,
        }
        self._dirty += 1

    # ---- Memory ----

    def memory_done(self, point_id: str) -> Optional[dict[str, Any]]:
        return self._memories.get(point_id)

    def mark_memory(self, point_id: str, *, status: str) -> None:
        self._memories[point_id] = {"status": status}
        self._dirty += 1

    # ---- 持久化 ----

    def maybe_save(self, *, force: bool = False) -> None:
        if self._path is None:
            return
        if not force and self._dirty < _CHECKPOINT_SAVE_INTERVAL:
            return
        self.save()

    def save(self) -> None:
        if self._path is None:
            return
        self._dirty = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "updated_at": datetime.now().isoformat(),
            "artifacts": self._artifacts,
            "memories": self._memories,
        }
        temp_path = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            temp_path.replace(self._path)
        finally:
            temp_path.unlink(missing_ok=True)


# ============ Artifact 扫描与 legacy 判定 ============

def replacement_artifact_id(workspace: WorkspaceIdentity, old_artifact_id: str) -> str:
    """确定性迁移命名空间 ID：同一条旧记录永远映射到同一个 replacement。"""
    digest = hashlib.sha256(
        f"{workspace.owner_user_id}|{workspace.workspace_id}|{old_artifact_id}".encode("utf-8")
    ).hexdigest()
    return f"{REPLACEMENT_ID_PREFIX}{digest[:32]}"


def _is_legacy_artifact(raw: dict[str, Any]) -> bool:
    """判定历史 Artifact 是否需要 canonical replacement。

    当前模型已删除 ``BaseArtifact.owner_agent_id`` 且平铺 turn 三元组升级为
    ``actor_identity``，因此出现任一 legacy 键即代表旧形状；``extra="ignore"``
    的新写入不会产生这些键。
    """
    artifact_type = raw.get("artifact_type")
    if artifact_type == "interaction":
        if "owner_agent_id" in raw:
            return True
        turns = raw.get("turns")
        if isinstance(turns, list):
            return any(
                isinstance(turn, dict) and "actor_identity" not in turn
                for turn in turns
            )
        return False
    if artifact_type in ("memory_creation", "memory_version", "document"):
        return "owner_agent_id" in raw
    return False


@dataclass
class ArtifactScanRecord:
    """单个 Artifact JSON 文件的扫描结果。"""

    path: Path
    raw: dict[str, Any] = field(default_factory=dict)
    artifact_id: str = ""
    artifact_type: str = ""
    workspace: Optional[WorkspaceIdentity] = None
    is_legacy: bool = False
    parse_error: Optional[str] = None

    @property
    def key(self) -> tuple[str, str, str]:
        assert self.workspace is not None
        return (self.workspace.owner_user_id, self.workspace.workspace_id, self.artifact_id)


def _normalize_artifact_workspace(raw: dict[str, Any]) -> Optional[WorkspaceIdentity]:
    """把 Artifact 记录的归属归一为 WorkspaceIdentity；无法归一化返回 None。

    复用运行时读取器（FilesystemArtifactStorageAdapter）的静态归一化逻辑，
    保证迁移对"这个文件属于哪个 Workspace"的判断与运行时完全一致。
    """
    try:
        return FilesystemArtifactStorageAdapter._workspace_from_data(raw)
    except (ValueError, TypeError):
        return None


def scan_artifact_records(root: Path) -> Iterator[ArtifactScanRecord]:
    """扫描 ArtifactStore 根目录下的全部 JSON 记录文件（同步遍历）。

    与适配器的受控 legacy 扫描同构：逐文件解析，单文件失败不中断扫描，
    只产生带 parse_error 的记录进入诊断清单。
    """
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("."):
            # 索引（.artifact_index.json）与临时文件（.xxx.tmp）不是 Artifact 记录。
            continue
        record = ArtifactScanRecord(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Artifact 记录必须是 JSON 对象")
        except Exception as exc:
            record.parse_error = f"{type(exc).__name__}: {exc}"
            yield record
            continue
        record.raw = raw
        record.artifact_id = raw.get("artifact_id") or ""
        record.artifact_type = raw.get("artifact_type") or ""
        record.is_legacy = _is_legacy_artifact(raw)
        if record.artifact_id:
            record.workspace = _normalize_artifact_workspace(raw)
        yield record


# ============ 引用链重写 ============

_REF_STATUS_REWRITTEN = "rewritten"
_REF_STATUS_UNCHANGED = "unchanged"
_REF_STATUS_DANGLING = "dangling"
_REF_STATUS_BLOCKED = "blocked"


class _RefResolver:
    """基于扫描结果与 replacement 映射解析 ArtifactRef 的重写目标。

    状态语义（与 Plan §4.4 fail-closed 条款对应）：

    - ``rewritten``：目标 legacy Artifact 已有（或 dry-run 下已计划）canonical
      replacement，换用新 ref；
    - ``unchanged``：目标本身已是 canonical（无需替换），ref 保持原样；
    - ``dangling``：目标在扫描中不存在（迁移前已损坏），ref 保持原样并计数；
    - ``blocked``：目标存在但无法给出 canonical replacement，消费方必须
      fail closed，避免 canonical 记录继续引用 legacy 记录。
    """

    def __init__(
        self,
        records: dict[tuple[str, str, str], ArtifactScanRecord],
        orphan_ids: set[str],
    ) -> None:
        self._records = records
        self._orphan_ids = orphan_ids
        self._replacement_refs: dict[tuple[str, str, str], ArtifactRef] = {}
        self._planned_ids: set[tuple[str, str, str]] = set()

    def record_replacement(self, key: tuple[str, str, str], ref: ArtifactRef) -> None:
        self._replacement_refs[key] = ref

    def demote_replacement(self, key: tuple[str, str, str]) -> None:
        """replacement 发布后校验失败时撤回映射，下游引用改为 blocked。"""
        self._replacement_refs.pop(key, None)

    def record_planned(self, key: tuple[str, str, str]) -> None:
        """dry-run 记录"计划中"的 replacement，供 Memory 阶段预测重写范围。"""
        self._planned_ids.add(key)

    def resolve(
        self,
        *,
        owner_user_id: str,
        workspace_id: str,
        artifact_id: str,
        allow_planned: bool = False,
    ) -> tuple[str, Optional[ArtifactRef]]:
        key = (owner_user_id, workspace_id, artifact_id)
        ref = self._replacement_refs.get(key)
        if ref is not None:
            return _REF_STATUS_REWRITTEN, ref
        if allow_planned and key in self._planned_ids:
            return _REF_STATUS_REWRITTEN, None
        record = self._records.get(key)
        if record is not None:
            if record.is_legacy:
                return _REF_STATUS_BLOCKED, None
            return _REF_STATUS_UNCHANGED, None
        if artifact_id in self._orphan_ids:
            return _REF_STATUS_BLOCKED, None
        return _REF_STATUS_DANGLING, None


def _ref_coordinates(ref: Any, default_workspace: WorkspaceIdentity) -> tuple[str, str, str]:
    """从 dict 或 ArtifactRef 中提取引用目标的 (owner, workspace_id, artifact_id)。"""
    if isinstance(ref, ArtifactRef):
        workspace = ref.workspace_identity
        artifact_id = ref.artifact_id
    else:
        raw_workspace = ref.get("workspace_identity")
        if isinstance(raw_workspace, dict):
            try:
                workspace = WorkspaceIdentity.model_validate(raw_workspace)
            except ValidationError:
                workspace = default_workspace
        else:
            workspace = default_workspace
        artifact_id = str(ref.get("artifact_id") or "")
    return (workspace.owner_user_id, workspace.workspace_id, artifact_id)


def _upgrade_legacy_turn_actor(turn: dict[str, Any]) -> Optional[dict[str, Any]]:
    """读取 turn 的执行者身份（用于贡献者证据聚合，不做 fail-closed 判定）。"""
    actor = turn.get("actor_identity")
    if isinstance(actor, dict):
        return actor
    user_id = (turn.get("user_id") or "").strip()
    if not user_id:
        return None
    return {
        "user_id": user_id,
        "agent_id": (turn.get("agent_id") or "").strip() or SYSTEM_AGENT_ID,
        "team_id": turn.get("team_id"),
    }


# ============ Artifact replacement 构造 ============

def _resolve_memory_artifact_source(raw: dict[str, Any], *, kind: str) -> str:
    """确定 legacy MemoryCreation/VersionArtifact 的 ``source_agent_id``。

    旧 builder 把 ``memory.meta.source_agent_id`` 原样写入 ``owner_agent_id``，
    因此对 Memory 类 Artifact 而言 ``owner_agent_id`` 是 source 字段的历史
    载体，可直接迁移；它不代表 Workspace owner，也不得用于回填贡献者。
    显式 ``source_intent=SYSTEM`` / ``update_source=SYSTEM_REWRITE`` 是
    "没有具体 Agent"的可信证据，允许使用保留 ``system``（Plan §4.2）。
    """
    explicit = raw.get("source_agent_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if kind == "memory_creation" and raw.get("source_intent") == "SYSTEM":
        return SYSTEM_AGENT_ID
    owner = raw.get("owner_agent_id")
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    if kind == "memory_version" and raw.get("update_source") == "SYSTEM_REWRITE":
        return SYSTEM_AGENT_ID
    raise MigrationRejected(
        f"{kind} 的 source_agent_id 无法确定：缺少 source_agent_id / owner_agent_id "
        "证据，且无显式 SYSTEM 语义（fail closed，不做猜测）"
    )


def _aggregate_interaction_contributors(
    raw: dict[str, Any],
    records: dict[tuple[str, str, str], ArtifactScanRecord],
    workspace: WorkspaceIdentity,
) -> list[str]:
    """从关联 InteractionArtifact 的 turn 聚合可信内容贡献者。

    只有"确实参与了来源交互"的 Agent 才构成贡献证据；最后访问 Agent、
    artifact owner 等单值字段不作为证据（Plan §4.4 / §6.2）。system 表示
    无具体 Agent，不是贡献者。无法证明时返回空集合，由报告记录。
    """
    contributors: list[str] = []
    source_artifacts = raw.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        return contributors
    for ref in source_artifacts:
        if not isinstance(ref, dict) or ref.get("artifact_type") != "interaction":
            continue
        target_id = str(ref.get("artifact_id") or "")
        if not target_id:
            continue
        target = records.get((workspace.owner_user_id, workspace.workspace_id, target_id))
        if target is None:
            continue
        turns = target.raw.get("turns")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            actor = _upgrade_legacy_turn_actor(turn)
            if actor is None:
                continue
            agent_id = (actor.get("agent_id") or "").strip()
            if agent_id and agent_id != SYSTEM_AGENT_ID and agent_id not in contributors:
                contributors.append(agent_id)
    return contributors


@dataclass
class _ReplacementPlan:
    """一条 legacy Artifact 的转换与发布计划。"""

    record: ArtifactScanRecord
    new_artifact_id: str
    replacement: Any
    ref: Optional[ArtifactRef] = None


def _management_identity_scope(workspace: WorkspaceIdentity) -> IdentityScope:
    """为 owner 构造管理读取作用域（保留 ``system`` actor，与 server 管理入口一致）。"""
    return IdentityScope(
        actor_identity=ActorIdentity(user_id=workspace.owner_user_id, agent_id=SYSTEM_AGENT_ID),
        workspace_identity=workspace,
    )


class ArtifactLegacyMigrator:
    """Artifact 阶段：扫描 → 转换 → 按 publish round 发布 → 校验。"""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        root: Path,
        report: MigrationReport,
        checkpoint: MigrationCheckpoint,
        dry_run: bool,
        repair_legacy_ownership: bool = False,
        memory_workspace_index: Optional[dict[str, Optional[WorkspaceIdentity]]] = None,
    ) -> None:
        self._store = store
        self._root = root
        self._report = report
        self._checkpoint = checkpoint
        self._dry_run = dry_run
        self._repair = repair_legacy_ownership
        self._memory_workspace_index = memory_workspace_index or {}
        self._records: dict[tuple[str, str, str], ArtifactScanRecord] = {}
        self._orphan_ids: set[str] = set()
        self._resolver = _RefResolver(self._records, self._orphan_ids)

    async def run(self) -> _RefResolver:
        await self._scan()
        for round_types in _ARTIFACT_PUBLISH_ROUNDS:
            plans: list[_ReplacementPlan] = []
            for record in self._records.values():
                if not record.is_legacy or record.artifact_type not in round_types:
                    continue
                plan = self._prepare(record)
                if plan is not None:
                    plans.append(plan)
            await self._publish(plans)
        self._report.inc(
            "artifact_already_canonical",
            sum(1 for record in self._records.values() if not record.is_legacy),
        )
        return self._resolver

    # ---- 扫描 ----

    async def _scan(self) -> None:
        records = await asyncio.to_thread(lambda: list(scan_artifact_records(self._root)))
        for record in records:
            self._report.inc("artifact_files_scanned")
            if record.parse_error is not None:
                self._report.inc("artifact_files_unreadable")
                self._report.add_diagnostic(
                    resource_type="artifact",
                    resource_id=record.path.name,
                    location=str(record.path),
                    reason=f"记录文件无法解析：{record.parse_error}",
                )
                continue
            if not record.artifact_id or record.workspace is None:
                adopted = None
                if record.artifact_id and record.workspace is None and self._repair:
                    adopted = self._adopt_workspace(record)
                if adopted is None:
                    self._report.inc("artifact_files_unreadable")
                    if record.artifact_id:
                        self._orphan_ids.add(record.artifact_id)
                    self._report.add_diagnostic(
                        resource_type="artifact",
                        resource_id=record.artifact_id or record.path.name,
                        location=str(record.path),
                        reason="无法归一化 Artifact Workspace 归属（fail closed）",
                    )
                    continue
                # 修复策略采纳归属后按正常记录处理。
                record.workspace = adopted
            if record.key in self._records:
                # 同 Workspace 重复 artifact_id：保留首条，重复文件进入诊断。
                self._report.add_diagnostic(
                    resource_type="artifact",
                    resource_id=record.artifact_id,
                    location=str(record.path),
                    reason="同 Workspace 下存在重复 artifact_id 记录，仅处理首个文件",
                )
                continue
            self._records[record.key] = record

    # ---- repair：归属采纳 ----

    def _adopt_workspace(self, record: ArtifactScanRecord) -> Optional[WorkspaceIdentity]:
        """repair 模式下为无归属 Artifact 采纳 Workspace；无证据返回 None。

        - memory 类 Artifact：``memory_id`` 关联到的 Memory 有可解析归属时
          采纳（历届 builder 均以 memory.workspace_identity 写入 Artifact）；
        - interaction：所有 turn 的 ``user_id`` 一致且非空时采纳
          （IdentityScope 不变量保证捕获时 actor.user_id == workspace.owner）。
        """
        if record.artifact_type in ("memory_creation", "memory_version"):
            memory_id = str(record.raw.get("memory_id") or "")
            if not memory_id:
                return None
            workspace = self._memory_workspace_index.get(memory_id)
            if workspace is None:
                return None
            self._report.inc("artifact_workspace_adopted")
            self._report.add_repair(
                resource_type="artifact",
                resource_id=record.artifact_id,
                rule="artifact_workspace_from_memory",
                detail=f"owner_user_id 为空，从关联 memory_id={memory_id} 采纳归属 "
                       f"{workspace.owner_user_id}/{workspace.workspace_id}",
            )
            return workspace
        if record.artifact_type == "interaction":
            user_id = self._consistent_interaction_user_id(record.raw)
            if user_id is None:
                return None
            workspace = WorkspaceIdentity(
                owner_user_id=user_id,
                workspace_key=MAIN_WORKSPACE_ID,
                workspace_id=MAIN_WORKSPACE_ID,
            )
            self._report.inc("artifact_workspace_adopted")
            self._report.add_repair(
                resource_type="artifact",
                resource_id=record.artifact_id,
                rule="artifact_workspace_from_interaction_turns",
                detail=f"owner_user_id 为空，从全部 turn 的一致 user_id={user_id!r} 采纳归属",
            )
            return workspace
        return None

    @staticmethod
    def _consistent_interaction_user_id(raw: dict[str, Any]) -> Optional[str]:
        """所有 turn 的执行者 user_id 一致且非空时返回它，否则 None。"""
        turns = raw.get("turns")
        if not isinstance(turns, list) or not turns:
            return None
        user_ids: set[str] = set()
        for turn in turns:
            if not isinstance(turn, dict):
                return None
            actor = turn.get("actor_identity")
            if isinstance(actor, dict):
                user_id = (actor.get("user_id") or "").strip()
            else:
                user_id = (turn.get("user_id") or "").strip()
            if not user_id:
                return None
            user_ids.add(user_id)
        if len(user_ids) != 1:
            return None
        return user_ids.pop()

    # ---- 转换 ----

    def _prepare(self, record: ArtifactScanRecord) -> Optional[_ReplacementPlan]:
        """转换一条 legacy Artifact；失败进入诊断清单并返回 None。"""
        self._report.inc("artifact_legacy_found")
        workspace = record.workspace
        assert workspace is not None
        key = self._checkpoint.artifact_key(workspace, record.artifact_id)
        old_content_hash = (
            "" if self._dry_run else hashlib.sha256(record.path.read_bytes()).hexdigest()
        )

        if not self._dry_run and self._checkpoint.artifact_hash_mismatch(key, old_content_hash):
            self._report.add_diagnostic(
                resource_type="artifact",
                resource_id=record.artifact_id,
                location=str(record.path),
                reason="checkpoint 记录的旧内容 hash 与当前文件不一致，"
                       "旧记录疑似在迁移后被改动，拒绝重复 replacement",
            )
            return None

        done = (
            None
            if self._dry_run
            else self._checkpoint.artifact_done(key, old_content_hash)
        )
        if done is not None:
            ref = ArtifactRef.model_validate(done["ref"])
            self._resolver.record_replacement(record.key, ref)
            self._report.artifact_mappings.append({
                "owner_user_id": workspace.owner_user_id,
                "workspace_id": workspace.workspace_id,
                "artifact_type": record.artifact_type,
                "old_artifact_id": record.artifact_id,
                "new_artifact_id": done["new_artifact_id"],
                "status": "resumed_from_checkpoint",
            })
            self._report.inc("artifact_replacement_resumed")
            return None

        new_artifact_id = replacement_artifact_id(workspace, record.artifact_id)
        try:
            replacement = self._convert(record, workspace, new_artifact_id)
        except MigrationRejected as exc:
            self._report.add_diagnostic(
                resource_type="artifact",
                resource_id=record.artifact_id,
                location=str(record.path),
                reason=str(exc),
            )
            return None
        except Exception as exc:
            self._report.add_diagnostic(
                resource_type="artifact",
                resource_id=record.artifact_id,
                location=str(record.path),
                reason=f"canonical replacement 构造失败：{type(exc).__name__}: {exc}",
            )
            return None

        self._report.inc("artifact_replacement_planned")
        if self._dry_run:
            self._resolver.record_planned(record.key)
            self._report.artifact_mappings.append({
                "owner_user_id": workspace.owner_user_id,
                "workspace_id": workspace.workspace_id,
                "artifact_type": record.artifact_type,
                "old_artifact_id": record.artifact_id,
                "new_artifact_id": new_artifact_id,
                "status": "planned",
            })
            return None
        return _ReplacementPlan(record, new_artifact_id, replacement)

    def _convert(
        self,
        record: ArtifactScanRecord,
        workspace: WorkspaceIdentity,
        new_artifact_id: str,
    ) -> Any:
        """把 legacy 原始 dict 转换为当前模型的 canonical replacement。"""
        override: dict[str, Any] = {
            "artifact_id": new_artifact_id,
            "content_hash": None,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "workspace_identity": workspace,
        }
        artifact_type = record.artifact_type
        if artifact_type == "interaction":
            # 平铺 turn 三元组由 InteractionTurnSnapshot 的读取升级分支重建为
            # actor_identity；缺 user_id 的 turn 触发 ValidationError → 诊断。
            return InteractionArtifact.model_validate({**record.raw, **override})
        if artifact_type == "document":
            # DocumentArtifact 没有任何 Agent provenance 字段，也不补造来源；
            # owner_agent_id 等旧键由 extra="ignore" 丢弃。
            return DocumentArtifact.model_validate({**record.raw, **override})

        # memory_creation / memory_version：确定 source，聚合贡献者，重写引用链。
        try:
            override["source_agent_id"] = _resolve_memory_artifact_source(
                record.raw, kind=artifact_type
            )
        except MigrationRejected:
            if not self._repair:
                raise
            # 用户批准的修复规则：来源缺证时按默认人偶归属（仅 provenance，
            # 不参与授权）；显式 SYSTEM 语义仍优先于默认值。
            override["source_agent_id"] = DEFAULT_AGENT_ID
            self._report.inc("artifact_source_defaulted_omni_doll")
            self._report.add_repair(
                resource_type="artifact",
                resource_id=record.artifact_id,
                rule="artifact_source_defaulted",
                detail="source_agent_id 无法确定（owner_agent_id 为空且无 SYSTEM 语义），"
                       f"按默认归属填入 {DEFAULT_AGENT_ID}",
            )
        if artifact_type == "memory_creation":
            override.setdefault(
                "contributing_agent_ids",
                _aggregate_interaction_contributors(record.raw, self._records, workspace),
            )
        else:
            override.setdefault("contributing_agent_ids", ())

        dangling, blocked = self._rewrite_raw_refs(
            record.raw,
            override,
            workspace=workspace,
        )
        if blocked:
            raise MigrationRejected(
                "引用链无法完整重写，目标 Artifact 存在但无法给出 canonical "
                f"replacement: {', '.join(sorted(blocked))}（fail closed）"
            )
        for ref_id in dangling:
            self._report.add_unresolved_ref(
                record_type=artifact_type,
                record_id=record.artifact_id,
                ref_artifact_id=ref_id,
                reason="引用目标在 ArtifactStore 中不存在（迁移前已损坏），ref 原样保留",
            )

        if artifact_type == "memory_creation":
            return MemoryCreationArtifact.model_validate({**record.raw, **override})
        return MemoryVersionArtifact.model_validate({**record.raw, **override})

    def _rewrite_raw_refs(
        self,
        raw: dict[str, Any],
        override: dict[str, Any],
        *,
        workspace: WorkspaceIdentity,
    ) -> tuple[list[str], list[str]]:
        """在原始 dict 上重写 memory Artifact 的内部引用链。

        返回 (dangling_ids, blocked_ids)。dangling 引用保持原样（悬空是迁移前
        已存在的问题）；blocked 引用使整条记录 fail closed。dry-run 下目标
        replacement 可能只是"计划中"（尚无新 ref 内容），此时保留原引用。
        """
        dangling: list[str] = []
        blocked: list[str] = []

        def _resolve(ref: dict[str, Any]) -> tuple[str, Optional[ArtifactRef]]:
            owner, workspace_id, artifact_id = _ref_coordinates(ref, workspace)
            return self._resolver.resolve(
                owner_user_id=owner,
                workspace_id=workspace_id,
                artifact_id=artifact_id,
                allow_planned=self._dry_run,
            )

        def _repair_ref(ref: dict[str, Any]) -> dict[str, Any]:
            """repair 模式：内部 ref 缺 workspace_identity 时回填本记录归属。"""
            if self._repair and not ref.get("workspace_identity"):
                self._report.inc("artifact_refs_workspace_backfilled")
                return {**ref, "workspace_identity": workspace.model_dump()}
            return ref

        def _rewrite_ref_list(refs: Any) -> Any:
            if not isinstance(refs, list):
                return refs
            rewritten: list[Any] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    rewritten.append(ref)
                    continue
                ref = _repair_ref(ref)
                status, new_ref = _resolve(ref)
                if status == _REF_STATUS_REWRITTEN and new_ref is not None:
                    rewritten.append(json.loads(new_ref.model_dump_json()))
                elif status == _REF_STATUS_BLOCKED:
                    blocked.append(str(ref.get("artifact_id") or ""))
                    rewritten.append(ref)
                else:
                    if status == _REF_STATUS_DANGLING:
                        dangling.append(str(ref.get("artifact_id") or ""))
                    rewritten.append(ref)
            return rewritten

        override["source_artifacts"] = _rewrite_ref_list(raw.get("source_artifacts"))
        initial_ref = raw.get("initial_version_ref")
        if isinstance(initial_ref, dict):
            initial_ref = _repair_ref(initial_ref)
            status, new_ref = _resolve(initial_ref)
            if status == _REF_STATUS_REWRITTEN and new_ref is not None:
                override["initial_version_ref"] = json.loads(new_ref.model_dump_json())
            else:
                if status == _REF_STATUS_DANGLING:
                    dangling.append(str(initial_ref.get("artifact_id") or ""))
                if status == _REF_STATUS_BLOCKED:
                    blocked.append(str(initial_ref.get("artifact_id") or ""))
                override["initial_version_ref"] = initial_ref
        return dangling, blocked

    # ---- 发布与校验 ----

    async def _publish(self, plans: list[_ReplacementPlan]) -> None:
        for plan in plans:
            try:
                plan.ref = await self._store.put(plan.replacement)
            except Exception as exc:
                self._report.inc("artifact_replacement_write_failed")
                self._report.add_diagnostic(
                    resource_type="artifact",
                    resource_id=plan.record.artifact_id,
                    location=str(plan.record.path),
                    reason=f"canonical replacement 写入失败：{type(exc).__name__}: {exc}",
                )
            else:
                self._resolver.record_replacement(plan.record.key, plan.ref)
        for plan in plans:
            if plan.ref is not None:
                await self._verify(plan)
        for plan in plans:
            if plan.ref is None:
                continue
            workspace = plan.record.workspace
            assert workspace is not None
            key = self._checkpoint.artifact_key(workspace, plan.record.artifact_id)
            self._checkpoint.mark_artifact(
                key,
                old_content_hash=hashlib.sha256(plan.record.path.read_bytes()).hexdigest(),
                new_artifact_id=plan.new_artifact_id,
                ref=json.loads(plan.ref.model_dump_json()),
            )
            self._report.artifact_mappings.append({
                "owner_user_id": workspace.owner_user_id,
                "workspace_id": workspace.workspace_id,
                "artifact_type": plan.record.artifact_type,
                "old_artifact_id": plan.record.artifact_id,
                "new_artifact_id": plan.new_artifact_id,
                "status": "superseded",
            })
            self._report.inc("artifact_replaced")
        self._checkpoint.maybe_save(force=True)

    async def _verify(self, plan: _ReplacementPlan) -> None:
        """发布后回读校验：hash 一致且 ref 可解析（Plan §4.3-2）。"""
        assert plan.record.workspace is not None
        scope = _management_identity_scope(plan.record.workspace)
        try:
            data = await self._store.get(scope, plan.ref)
            if data.get("artifact_id") != plan.new_artifact_id:
                raise ValueError("回读记录与 replacement ID 不一致")
        except Exception as exc:
            self._resolver.demote_replacement(plan.record.key)
            plan.ref = None
            self._report.inc("artifact_replacement_verify_failed")
            self._report.add_diagnostic(
                resource_type="artifact",
                resource_id=plan.record.artifact_id,
                location=str(plan.record.path),
                reason=f"replacement 已写入但校验失败（存在部分 replacement）"
                       f" new_artifact_id={plan.new_artifact_id}: {exc}",
                partial_replacement=True,
            )
            return
        self._report.inc("artifact_replacement_verified")


# ============ Memory V1 → V2 转换 ============

@dataclass
class V1ConversionResult:
    """单条 V1 Memory 的转换结果。"""

    atom: Optional[MemoryAtom] = None
    reason: Optional[str] = None
    defaulted_public: bool = False
    backfilled_refs: int = 0


def resolve_memory_workspace_from_meta(payload: dict[str, Any]) -> Optional[WorkspaceIdentity]:
    """仅从 Memory payload 的 meta 解析归属，供 repair 模式的归属采纳使用。

    解析规则与 V1 转换一致：完整投影优先，其次 ``user_id`` → main_workspace；
    部分投影、owner 冲突或缺归属返回 None（不可采纳，不猜测）。不校验
    payload 其余部分，因此对 meta 已损坏但归属完好的死簇记录仍可用。
    """
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    try:
        projected = _extract_complete_projection(meta)
        user_id = _optional_non_empty(meta.get("user_id"), "user_id")
    except MigrationRejected:
        return None
    if projected is not None:
        if user_id is not None and user_id != projected.owner_user_id:
            return None
        return projected
    if user_id is None:
        return None
    return WorkspaceIdentity(
        owner_user_id=user_id,
        workspace_key=MAIN_WORKSPACE_ID,
        workspace_id=MAIN_WORKSPACE_ID,
    )


def _backfill_ref_workspaces(payload: Any, workspace: WorkspaceIdentity) -> tuple[Any, int]:
    """给 payload.artifacts 中缺失 ``workspace_identity`` 的 ref 回填归属。

    仅在 repair 模式调用。不修改输入 dict（按需复制路径），返回
    (可能替换后的 payload, 回填数量)。"缺失"指键不存在或值为空。
    """
    if not isinstance(payload, dict):
        return payload, 0
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return payload, 0
    ws_dump = workspace.model_dump()
    filled = 0

    def _patch_ref(ref: Any) -> Any:
        nonlocal filled
        if isinstance(ref, dict) and not ref.get("workspace_identity"):
            filled += 1
            return {**ref, "workspace_identity": dict(ws_dump)}
        return ref

    patched: dict[str, Any] = dict(artifacts)
    refs = artifacts.get("refs")
    if isinstance(refs, list):
        patched["refs"] = [_patch_ref(ref) for ref in refs]
    events = artifacts.get("events")
    if isinstance(events, list):
        new_events: list[Any] = []
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("artifact_refs"), list):
                new_refs = [_patch_ref(ref) for ref in event["artifact_refs"]]
                new_events.append({**event, "artifact_refs": new_refs})
            else:
                new_events.append(event)
        patched["events"] = new_events
    if filled == 0:
        return payload, 0
    return {**payload, "artifacts": patched}, filled


def _optional_non_empty(value: Any, field_name: str) -> Optional[str]:
    """与 memory_codec 相同的规范化：缺失返回 None，空白/非字符串 fail closed。"""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MigrationRejected(f"Memory meta.{field_name} 必须是非空字符串")
    return value.strip()


def convert_v1_memory_payload(
    raw: dict[str, Any],
    *,
    missing_visibility_policy: Literal["public", "fail"],
    repair_missing_ref_workspace: bool = False,
) -> V1ConversionResult:
    """把 V1 Memory payload 转换为 canonical v2 MemoryAtom（Plan §4.1 字段映射）。

    与 ``memory_codec._decode_v1`` 的兼容语义一致，但把"缺失 visibility 默认
    PUBLIC"提升为显式策略选项，并把每个 fail-closed 分支的原因透出给报告。

    ``repair_missing_ref_workspace`` 仅在 repair 模式启用：payload refs 缺失
    ``workspace_identity`` 时用本记录解析出的 Workspace 回填（"引用始终与
    Memory 同 Workspace"是历届 builder 的写入不变量）；未启用时这类记录
    因无法通过 ArtifactRef 校验而 fail closed。
    """
    try:
        return _convert_v1_inner(
            raw,
            missing_visibility_policy=missing_visibility_policy,
            repair_missing_ref_workspace=repair_missing_ref_workspace,
        )
    except MigrationRejected as exc:
        return V1ConversionResult(reason=str(exc))
    except Exception as exc:
        return V1ConversionResult(
            reason=f"V1 Memory 转换失败：{type(exc).__name__}: {exc}"
        )


def _convert_v1_inner(
    raw: dict[str, Any],
    *,
    missing_visibility_policy: Literal["public", "fail"],
    repair_missing_ref_workspace: bool,
) -> V1ConversionResult:
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        raise MigrationRejected("Memory meta 必须是对象")

    # 1. Workspace 归属：完整投影优先；user_id 是唯一合法的 legacy 归属来源。
    projected = _extract_complete_projection(meta)
    legacy_user_id = _optional_non_empty(meta.get("user_id"), "user_id")
    if projected is not None:
        if legacy_user_id is not None and legacy_user_id != projected.owner_user_id:
            raise MigrationRejected(
                f"legacy user_id ({legacy_user_id!r}) 与 Workspace owner "
                f"({projected.owner_user_id!r}) 冲突（fail closed）"
            )
        workspace = projected
    elif legacy_user_id is not None:
        workspace = WorkspaceIdentity(
            owner_user_id=legacy_user_id,
            workspace_key=MAIN_WORKSPACE_ID,
            workspace_id=MAIN_WORKSPACE_ID,
        )
    else:
        raise MigrationRejected(
            "meta.user_id 缺失或为空，且无 Workspace 投影，无法确定资源归属（fail closed）"
        )

    # 2. 来源 provenance：source_agent_id 必填；team_id 可选。
    source_agent_id = _optional_non_empty(meta.get("source_agent_id"), "source_agent_id")
    if source_agent_id is None:
        raise MigrationRejected(
            "meta.source_agent_id 缺失，无法迁移来源 provenance（fail closed）"
        )
    source_team_id = _optional_non_empty(meta.get("team_id"), "team_id")

    # 3. 可见性 → access_policy：缺失 visibility 走显式脚本策略，不静默放宽。
    had_visibility = "visibility" in meta
    if not had_visibility and missing_visibility_policy == "fail":
        raise MigrationRejected(
            "meta.visibility 缺失且迁移策略为 fail-closed，拒绝默认 PUBLIC（进入诊断清单）"
        )
    visibility_value = meta.get("visibility", "PUBLIC")
    if hasattr(visibility_value, "value"):
        visibility_value = visibility_value.value
    policy = _adapt_v1_policy(
        visibility=str(visibility_value),
        source_agent_id=source_agent_id,
        source_team_id=source_team_id,
    )

    # 4. 组装 canonical meta：移除平铺 legacy 字段；贡献者不可证明时不回填。
    domain_meta = {k: v for k, v in meta.items() if k not in _V1_META_FIELDS_TO_DROP}
    domain_meta.update(
        {
            "workspace_identity": workspace,
            "source_agent_id": source_agent_id,
            "source_team_id": source_team_id,
            "access_policy": policy,
        }
    )
    domain_meta.setdefault("contributing_agent_ids", ())

    # 5. repair 模式：refs 缺 workspace_identity 时按本记录归属回填。
    payload_value = raw["payload"]
    backfilled_refs = 0
    if repair_missing_ref_workspace:
        payload_value, backfilled_refs = _backfill_ref_workspaces(payload_value, workspace)

    try:
        atom = MemoryAtom(
            id=raw["id"],
            meta=MetaData.model_validate(domain_meta),
            index=raw["index"],
            payload=payload_value,
            relations=raw.get("relations", {}),
        )
    except MigrationRejected:
        raise
    except Exception as exc:
        raise MigrationRejected(
            f"canonical v2 模型校验失败（含 PRIVATE/TEAM target 合法性）：{exc}"
        ) from exc
    return V1ConversionResult(
        atom=atom,
        defaulted_public=not had_visibility,
        backfilled_refs=backfilled_refs,
    )


def _extract_complete_projection(meta: dict[str, Any]) -> Optional[WorkspaceIdentity]:
    """与 codec 一致：完整投影才可用，部分投影拒绝猜测补齐。"""
    present = [name for name in _WORKSPACE_PROJECTION_FIELDS if meta.get(name) is not None]
    if not present:
        return None
    if len(present) != len(_WORKSPACE_PROJECTION_FIELDS):
        raise MigrationRejected("Memory 包含部分 Workspace 投影，拒绝猜测补齐（fail closed）")
    try:
        return WorkspaceIdentity(
            owner_user_id=meta["owner_user_id"],
            workspace_key=meta["workspace_key"],
            workspace_id=meta["workspace_id"],
        )
    except Exception as exc:
        raise MigrationRejected(f"无效的 Workspace 索引投影：{exc}") from exc


def _adapt_v1_policy(
    *,
    visibility: str,
    source_agent_id: str,
    source_team_id: Optional[str],
) -> MemoryAccessPolicy:
    """legacy visibility → V2 读取策略（与 codec 一致；PRIVATE target=来源 Agent）。"""
    try:
        if visibility == "PUBLIC":
            return MemoryAccessPolicy.public()
        if visibility == "PRIVATE":
            return MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id=source_agent_id,
            )
        if visibility == "WORKSPACE":
            if source_team_id is None:
                raise MigrationRejected("legacy WORKSPACE Memory 缺少 team_id（fail closed）")
            return MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id=source_team_id,
            )
    except MigrationRejected:
        raise
    except Exception as exc:
        # 典型场景：PRIVATE target 落在保留 system 上，被 V2 模型校验拒绝。
        raise MigrationRejected(f"legacy visibility 无法映射为 V2 策略：{exc}") from exc
    raise MigrationRejected(f"未知的 legacy Memory visibility: {visibility!r}（fail closed）")


def _verify_roundtrip(atom: MemoryAtom) -> None:
    """发布前用运行时 codec 回读校验 canonical payload（Plan §4.3-2）。"""
    payload = atom.to_qdrant_payload()
    try:
        decoded = decode_memory_payload(payload)
    except MemoryDecodeError as exc:
        raise MigrationRejected(f"canonical payload 无法被运行时 codec 解码：{exc}") from exc
    if decoded.workspace_identity != atom.workspace_identity:
        raise MigrationRejected("回读 Workspace 归属与迁移结果不一致")
    if decoded.meta.access_policy != atom.meta.access_policy:
        raise MigrationRejected("回读读取策略与迁移结果不一致")


# ============ Memory 阶段 ============

class MemoryMigrationAccess(Protocol):
    """Memory 阶段对存储的访问端口（生产实现包装 QdrantMemoryStore）。"""

    def iter_raw_points(
        self, batch_size: int
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """按批次遍历全部原始 point（payload 不做任何解释）。"""
        ...

    async def replace_point(self, memory: MemoryAtom, *, previous_point_id: str) -> None:
        """以 canonical 写入路径发布 Memory，并清理旧 point。"""
        ...


async def _iter_qdrant_points(
    client: Any,
    collection_name: str,
    batch_size: int,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """按批 scroll 全部原始 point；需要未经解码的 payload，不走 typed 读取。"""
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return
        for point in points:
            yield str(point.id), dict(point.payload or {})
        if offset is None:
            return


class QdrantScrollOnlyMemoryAccess:
    """dry-run 专用的 Memory 访问：只 scroll 原始 payload。

    dry-run 不发布任何 Memory，因此不加载 Embedding 服务（
    ``QdrantMemoryStore`` 构造即加载）；``replace_point`` 被误用时显式失败。
    """

    def __init__(self, qdrant_config: "QdrantConfig") -> None:
        from hivememory.infrastructure.storage.qdrant_client import (
            create_async_qdrant_client,
        )

        self._client = create_async_qdrant_client(qdrant_config)
        self._collection_name = qdrant_config.collection_name

    def iter_raw_points(
        self, batch_size: int
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        return _iter_qdrant_points(self._client, self._collection_name, batch_size)

    async def replace_point(
        self, memory: MemoryAtom, *, previous_point_id: str
    ) -> None:
        raise RuntimeError("dry-run 模式不应发布 Memory（内部错误）")


class QdrantMemoryMigrationAccess:
    """``MemoryMigrationAccess`` 的 Qdrant 生产实现。

    发布走 ``QdrantMemoryStore.upsert_memory`` canonical 写入路径（重新生成
    向量，并自动清理 ``str(memory.id)`` 旧点），另外防御性清理其他残留旧点。
    """

    def __init__(self, store: "QdrantMemoryStore") -> None:
        self._store = store

    def iter_raw_points(
        self, batch_size: int
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        return _iter_qdrant_points(
            self._store.client, self._store.collection_name, batch_size
        )

    async def replace_point(
        self, memory: MemoryAtom, *, previous_point_id: str
    ) -> None:
        await self._store.upsert_memory(memory)
        composite_id = self._store._point_id(
            WorkspaceMemoryKey(
                workspace_identity=memory.workspace_identity,
                memory_id=memory.id,
            )
        )
        stale_ids = {previous_point_id, str(memory.id)} - {composite_id}
        for stale_id in stale_ids:
            try:
                await self._store.client.delete(
                    collection_name=self._store.collection_name,
                    points_selector=[stale_id],
                )
            except Exception as exc:
                logger.warning("清理残留旧 Memory 点失败: %s (%s)", stale_id, exc)


class MemoryPhaseMigrator:
    """Memory 阶段：V1 → V2 转换、V2 ref 重写与 canonical 发布。"""

    def __init__(
        self,
        *,
        access: MemoryMigrationAccess,
        report: MigrationReport,
        checkpoint: MigrationCheckpoint,
        options: MigrationOptions,
        resolver: _RefResolver,
    ) -> None:
        self._access = access
        self._report = report
        self._checkpoint = checkpoint
        self._options = options
        self._resolver = resolver

    async def run(self) -> None:
        async for point_id, payload in self._access.iter_raw_points(self._options.batch_size):
            self._report.inc("memory_points_scanned")
            await self._migrate_point(point_id, payload)
            self._checkpoint.maybe_save()
        self._checkpoint.maybe_save(force=True)

    async def _migrate_point(self, point_id: str, payload: dict[str, Any]) -> None:
        if self._checkpoint.memory_done(point_id) is not None:
            self._report.inc("memory_resumed_from_checkpoint")
            return

        schema_version = payload.get("schema_version")
        if schema_version is None:
            await self._migrate_v1(point_id, payload)
            return
        if schema_version == 2:
            await self._rewrite_v2_refs(point_id, payload)
            return
        self._report.add_diagnostic(
            resource_type="memory",
            resource_id=point_id,
            location="qdrant",
            reason=f"不支持的 Memory schema_version: {schema_version!r}（fail closed）",
        )

    async def _migrate_v1(self, point_id: str, payload: dict[str, Any]) -> None:
        self._report.inc("memory_v1_found")
        memory_id = str(payload.get("id") or point_id)
        result = convert_v1_memory_payload(
            payload,
            missing_visibility_policy=self._options.missing_visibility_policy,
            repair_missing_ref_workspace=self._options.repair_legacy_ownership,
        )
        if result.atom is None:
            self._report.inc("memory_failed")
            self._report.add_diagnostic(
                resource_type="memory",
                resource_id=memory_id,
                location=f"qdrant point {point_id}",
                reason=result.reason or "未知转换失败",
            )
            return
        atom = result.atom
        if result.defaulted_public:
            self._report.inc("memory_missing_visibility_defaulted_public")
        if result.backfilled_refs:
            self._report.inc("memory_refs_workspace_backfilled", result.backfilled_refs)
            self._report.add_repair(
                resource_type="memory",
                resource_id=memory_id,
                rule="memory_ref_workspace_backfilled",
                detail=f"refs 缺失 workspace_identity，按本记录归属回填 "
                       f"{result.backfilled_refs} 个 ref",
            )

        changed, ok = self._rewrite_atom_refs(atom, record_id=memory_id)
        if not ok:
            self._report.inc("memory_failed")
            return
        if self._options.dry_run:
            self._report.inc("memory_v1_would_migrate")
            self._report.memory_mappings.append({
                "old_point_id": point_id,
                "memory_id": memory_id,
                "owner_user_id": atom.workspace_identity.owner_user_id,
                "workspace_id": atom.workspace_identity.workspace_id,
                "action": "would_migrate_v1_to_v2",
            })
            return
        try:
            _verify_roundtrip(atom)
        except MigrationRejected as exc:
            self._report.inc("memory_failed")
            self._report.add_diagnostic(
                resource_type="memory",
                resource_id=memory_id,
                location=f"qdrant point {point_id}",
                reason=str(exc),
            )
            return
        try:
            await self._access.replace_point(atom, previous_point_id=point_id)
        except Exception as exc:
            self._report.inc("memory_failed")
            self._report.add_diagnostic(
                resource_type="memory",
                resource_id=memory_id,
                location=f"qdrant point {point_id}",
                reason=f"canonical Memory 发布失败：{type(exc).__name__}: {exc}",
            )
            return
        self._checkpoint.mark_memory(point_id, status="migrated_v1_to_v2")
        self._report.inc("memory_v1_migrated")
        self._report.memory_mappings.append({
            "old_point_id": point_id,
            "memory_id": memory_id,
            "owner_user_id": atom.workspace_identity.owner_user_id,
            "workspace_id": atom.workspace_identity.workspace_id,
            "action": "migrated_v1_to_v2",
        })

    async def _rewrite_v2_refs(self, point_id: str, payload: dict[str, Any]) -> None:
        """V2 记录仅在其引用链仍指向 legacy Artifact 时重写并重新发布。"""
        try:
            atom = decode_memory_payload(payload)
        except MemoryDecodeError as exc:
            self._report.inc("memory_failed")
            self._report.add_diagnostic(
                resource_type="memory",
                resource_id=str(payload.get("id") or point_id),
                location=f"qdrant point {point_id}",
                reason=f"V2 Memory 无法被运行时 codec 解码：{exc}",
            )
            return
        memory_id = str(atom.id)
        changed, ok = self._rewrite_atom_refs(atom, record_id=memory_id)
        if not ok:
            self._report.inc("memory_failed")
            return
        if not changed:
            self._report.inc("memory_v2_skipped_already_canonical")
            return
        self._report.inc("memory_v2_refs_rewritten")
        if self._options.dry_run:
            self._report.inc("memory_v2_would_rewrite_refs")
            self._report.memory_mappings.append({
                "old_point_id": point_id,
                "memory_id": memory_id,
                "owner_user_id": atom.workspace_identity.owner_user_id,
                "workspace_id": atom.workspace_identity.workspace_id,
                "action": "would_rewrite_artifact_refs",
            })
            return
        try:
            await self._access.replace_point(atom, previous_point_id=point_id)
        except Exception as exc:
            self._report.inc("memory_failed")
            self._report.add_diagnostic(
                resource_type="memory",
                resource_id=memory_id,
                location=f"qdrant point {point_id}",
                reason=f"V2 ref 重写发布失败：{type(exc).__name__}: {exc}",
            )
            return
        self._checkpoint.mark_memory(point_id, status="refs_rewritten")
        self._report.memory_mappings.append({
            "old_point_id": point_id,
            "memory_id": memory_id,
            "owner_user_id": atom.workspace_identity.owner_user_id,
            "workspace_id": atom.workspace_identity.workspace_id,
            "action": "rewrote_artifact_refs",
        })

    def _rewrite_atom_refs(
        self,
        atom: MemoryAtom,
        *,
        record_id: str,
    ) -> tuple[bool, bool]:
        """就地重写 MemoryAtom 引用链。

        返回 (是否有变化, 是否可继续)。目标存在但无法给出 canonical
        replacement 时整条记录 fail closed；悬空引用保持原样并计数。
        dry-run 下"计划中"的 replacement 视为将重写（保持原 ref 内容）。
        """
        changed = False
        artifacts = atom.payload.artifacts

        new_refs: list[ArtifactRef] = []
        for ref in artifacts.refs:
            status, replacement = self._resolve_atom_ref(ref)
            if status == _REF_STATUS_BLOCKED:
                self._report.add_diagnostic(
                    resource_type="memory",
                    resource_id=record_id,
                    location="memory.payload.artifacts.refs",
                    reason=f"引用链无法完整重写，目标 Artifact 存在但迁移失败: "
                           f"{ref.artifact_id}（fail closed）",
                )
                return changed, False
            if status == _REF_STATUS_REWRITTEN:
                changed = True
                new_refs.append(replacement or ref)
            else:
                if status == _REF_STATUS_DANGLING:
                    self._report.add_unresolved_ref(
                        record_type="memory",
                        record_id=record_id,
                        ref_artifact_id=ref.artifact_id,
                        reason="引用目标在 ArtifactStore 中不存在（迁移前已损坏），ref 原样保留",
                    )
                new_refs.append(ref)
        if changed:
            artifacts.refs = new_refs

        for event in artifacts.events:
            new_event_refs: list[ArtifactRef] = []
            event_changed = False
            for ref in event.artifact_refs:
                status, replacement = self._resolve_atom_ref(ref)
                if status == _REF_STATUS_BLOCKED:
                    self._report.add_diagnostic(
                        resource_type="memory",
                        resource_id=record_id,
                        location="memory.payload.artifacts.events",
                        reason=f"事件引用链无法完整重写，目标 Artifact 迁移失败: "
                               f"{ref.artifact_id}（fail closed）",
                    )
                    return changed, False
                if status == _REF_STATUS_REWRITTEN:
                    event_changed = True
                    new_event_refs.append(replacement or ref)
                else:
                    if status == _REF_STATUS_DANGLING:
                        self._report.add_unresolved_ref(
                            record_type="memory",
                            record_id=record_id,
                            ref_artifact_id=ref.artifact_id,
                            reason="事件引用目标不存在（迁移前已损坏），ref 原样保留",
                        )
                    new_event_refs.append(ref)
            if event_changed:
                event.artifact_refs = new_event_refs
                changed = True
        return changed, True

    def _resolve_atom_ref(self, ref: ArtifactRef) -> tuple[str, Optional[ArtifactRef]]:
        workspace = ref.workspace_identity
        return self._resolver.resolve(
            owner_user_id=workspace.owner_user_id,
            workspace_id=workspace.workspace_id,
            artifact_id=ref.artifact_id,
            allow_planned=self._options.dry_run,
        )


# ============ 组合入口 ============

class V1LegacyMigrator:
    """迁移引擎组合入口：先 Artifact replacement，后 Memory canonical 发布。"""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        artifacts_root: Path,
        memory_access: MemoryMigrationAccess,
        options: Optional[MigrationOptions] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        self.options = options or MigrationOptions()
        self.report = MigrationReport(options=self.options.to_dict())
        self.checkpoint = MigrationCheckpoint(
            None if self.options.dry_run else checkpoint_path
        )
        self._artifact_store = artifact_store
        self._artifacts_root = artifacts_root
        self._memory_access = memory_access

    async def run(self) -> MigrationReport:
        started = time.monotonic()
        self.checkpoint.load()
        logger.info(
            "迁移开始: dry_run=%s, artifacts_root=%s",
            self.options.dry_run,
            self._artifacts_root,
        )

        # repair 预扫描：memory_id → 归属，供 Artifact 归属采纳使用。
        memory_workspace_index: dict[str, Optional[WorkspaceIdentity]] = {}
        if self.options.repair_legacy_ownership:
            memory_workspace_index = await self._build_memory_workspace_index()

        artifact_migrator = ArtifactLegacyMigrator(
            store=self._artifact_store,
            root=self._artifacts_root,
            report=self.report,
            checkpoint=self.checkpoint,
            dry_run=self.options.dry_run,
            repair_legacy_ownership=self.options.repair_legacy_ownership,
            memory_workspace_index=memory_workspace_index,
        )
        resolver = await artifact_migrator.run()

        memory_migrator = MemoryPhaseMigrator(
            access=self._memory_access,
            report=self.report,
            checkpoint=self.checkpoint,
            options=self.options,
            resolver=resolver,
        )
        await memory_migrator.run()

        self.checkpoint.save()
        self.report.generated_at = datetime.now().isoformat()
        self.report.counts["elapsed_seconds"] = int(time.monotonic() - started)
        logger.info("迁移完成: %s", self.report.counts)
        return self.report

    async def _build_memory_workspace_index(
        self,
    ) -> dict[str, Optional[WorkspaceIdentity]]:
        """遍历 Memory payload 的 meta 构建 memory_id → 归属索引。

        同一 memory_id 解析出不同归属（异常数据）时标记为 None，不可采纳。
        """
        index: dict[str, Optional[WorkspaceIdentity]] = {}
        async for point_id, payload in self._memory_access.iter_raw_points(
            self.options.batch_size
        ):
            memory_id = str(payload.get("id") or point_id)
            workspace = resolve_memory_workspace_from_meta(payload)
            if memory_id in index and index[memory_id] != workspace:
                index[memory_id] = None
            else:
                index[memory_id] = workspace
        return index


def summarize_report(report: MigrationReport) -> str:
    """生成人类可读摘要（Plan §4.3-5：JSON + 人类可读摘要）。"""
    counts = report.counts
    lines = [
        "=" * 62,
        "V1 Memory / Artifact legacy 迁移摘要",
        "=" * 62,
        f"执行模式: {'DRY-RUN（未写入）' if report.options.get('dry_run') else 'APPLY（已写入）'}",
        "",
        "[Artifact]",
        f"  扫描记录文件: {counts.get('artifact_files_scanned', 0)}",
        f"  无法解析/归属不明: {counts.get('artifact_files_unreadable', 0)}",
        f"  已是 canonical（跳过）: {counts.get('artifact_already_canonical', 0)}",
        f"  legacy 待迁移: {counts.get('artifact_legacy_found', 0)}",
        f"  replacement 已发布并校验: {counts.get('artifact_replaced', 0)}"
        f"（校验失败 {counts.get('artifact_replacement_verify_failed', 0)}）",
        f"  checkpoint 续跑跳过: {counts.get('artifact_replacement_resumed', 0)}",
        "",
        "[Memory]",
        f"  扫描 point: {counts.get('memory_points_scanned', 0)}",
        f"  V1 记录: {counts.get('memory_v1_found', 0)}"
        f"（迁移 {counts.get('memory_v1_migrated', 0)}，"
        f"dry-run 预计迁移 {counts.get('memory_v1_would_migrate', 0)}）",
        f"  缺失 visibility 默认 PUBLIC: "
        f"{counts.get('memory_missing_visibility_defaulted_public', 0)}",
        f"  V2 引用重写: {counts.get('memory_v2_refs_rewritten', 0)}"
        f"（dry-run 预计 {counts.get('memory_v2_would_rewrite_refs', 0)}）",
        f"  V2 已是 canonical（跳过）: {counts.get('memory_v2_skipped_already_canonical', 0)}",
        f"  失败（fail closed）: {counts.get('memory_failed', 0)}",
        "",
        f"悬空引用保留数: {len(report.unresolved_refs)}",
        f"诊断条目数: {len(report.diagnostics)}",
        f"修复条目数: {len(report.repairs)}"
        f"（归属采纳 {counts.get('artifact_workspace_adopted', 0)}，"
        f"来源默认 omni_doll {counts.get('artifact_source_defaulted_omni_doll', 0)}，"
        f"ref 回填 {counts.get('memory_refs_workspace_backfilled', 0)}"
        f"+{counts.get('artifact_refs_workspace_backfilled', 0)}）",
    ]
    if report.repairs:
        lines.append("")
        lines.append("修复明细（最多展示 20 条，完整清单见报告 JSON）:")
        for entry in report.repairs[:20]:
            lines.append(
                f"  - [{entry['resource_type']}] {entry['resource_id']}"
                f" ({entry['rule']}): {entry['detail']}"
            )
    if report.diagnostics:
        lines.append("")
        lines.append("诊断明细（最多展示 20 条，完整清单见报告 JSON）:")
        for entry in report.diagnostics[:20]:
            lines.append(
                f"  - [{entry['resource_type']}] {entry['resource_id']}: {entry['reason']}"
            )
    lines.append("=" * 62)
    return "\n".join(lines)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MigrationCheckpoint",
    "MigrationOptions",
    "MigrationReport",
    "MemoryMigrationAccess",
    "MemoryPhaseMigrator",
    "QdrantMemoryMigrationAccess",
    "QdrantScrollOnlyMemoryAccess",
    "REPLACEMENT_ID_PREFIX",
    "V1LegacyMigrator",
    "convert_v1_memory_payload",
    "replacement_artifact_id",
    "resolve_memory_workspace_from_meta",
    "scan_artifact_records",
    "summarize_report",
]
