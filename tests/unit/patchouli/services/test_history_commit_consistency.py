"""A2-P MVL-3：完整历史构建、必需版本存储与顺序失败传播的行为测试。

验证契约（A2-P §5.1/§5.3、MVL-0 M0.3）：
- 捕获时点：版本快照在版本/内容时间分配之后、追加自身 refs/events 之前
  生成，不含本次版本记录自身的引用与事件；
- 版本记录是内容提交成功的前置条件：版本存储不可用或写入失败时，
  canonical 不发布，错误向调用方传播；
- canonical 写入失败或提交被取消：错误传播，已写入 Artifact 保留为孤立
  记录，不是已提交版本；
- 历史写入后不可变：提交后修改返回对象不影响已存历史。

本文件使用真实 ``MemoryArtifactBuilder`` + ``ArtifactStore``（临时目录）与
内存版 ``MidTermStoragePort``，不 mock 被测提交链本身。
"""

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from hivememory.core.models import (
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import MemoryEventLog, MemoryEventType
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationContext,
    GenerationOutcome,
    GenerationRequest,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.stores import ArtifactStore
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from tests.helpers.memory import make_memory_identity_scope, make_memory_metadata

FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


class _InMemoryMidTermPort:
    """内存版中期存储：记录 upsert 调用，支持按 memory_id 注入写入失败。"""

    def __init__(self) -> None:
        self._atoms: dict[WorkspaceMemoryKey, MemoryAtom] = {}
        self.upsert_calls: list[MemoryAtom] = []
        self.fail_upsert_for: set[str] = set()
        self.upsert_error: Exception = RuntimeError("canonical 写入失败（测试注入）")

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        if str(memory.id) in self.fail_upsert_for:
            raise self.upsert_error
        self.upsert_calls.append(memory)
        self._atoms[self._key_of(memory)] = memory.model_copy(deep=True)

    def _key_of(self, memory: MemoryAtom) -> WorkspaceMemoryKey:
        return WorkspaceMemoryKey(
            workspace_identity=memory.workspace_identity,
            memory_id=memory.id,
        )

    def _make_key(self, identity_scope: IdentityScope, memory_id) -> WorkspaceMemoryKey:
        return WorkspaceMemoryKey(
            workspace_identity=identity_scope.workspace_identity,
            memory_id=memory_id,
        )

    async def get(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return self._atoms.get(self._make_key(identity_scope, memory_id))

    async def get_by_alias(
        self,
        identity_scope: IdentityScope,
        alias: str,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return None

    async def get_for_mutation(
        self, identity_scope: IdentityScope, memory_id: UUID
    ) -> MemoryAtom | None:
        return self.get(identity_scope, memory_id)

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        return self._atoms.get(key)

    async def patch_payload(self, key: WorkspaceMemoryKey, patch) -> MemoryAtom | None:
        atom = self._atoms.get(key)
        if atom is None:
            return None
        for path, value in patch.items():
            _, section, field = path.split(".", 2)
            target = atom.meta.lifecycle if section == "lifecycle" else getattr(atom.meta, section)
            setattr(target, field, value)
        return atom

    async def delete(self, identity_scope: IdentityScope, memory_id: UUID) -> bool:
        return self._atoms.pop(self._make_key(identity_scope, memory_id), None) is not None

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        return self._atoms.pop(key, None) is not None

    async def batch_delete(self, identity_scope: IdentityScope, ids) -> int:
        return 0

    async def search(
        self,
        identity_scope,
        query,
        top_k,
        filters=None,
        mode="dense",
        score_threshold=0.0,
        *,
        enforce_actor_visibility=True,
    ):
        return []

    async def scroll(
        self, identity_scope=None, filters=None, limit=100, *, enforce_actor_visibility=True
    ):
        return []

    async def count(self, identity_scope=None, filters=None) -> int:
        return len(self._atoms)

    async def list_all_for_maintenance(self, limit: int = 10000):
        return list(self._atoms.values())

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="mid_term", healthy=True)


class _StubGenerationEngine:
    """返回预置 outcome 的最小引擎替身（Familiar 只消费 process 结果）。"""

    def __init__(self, outcomes: list[GenerationOutcome]) -> None:
        self._outcomes = outcomes
        self.received_now: datetime | None = None

    async def process(self, request, *, identity_scope, now=None):
        self.received_now = now
        return self._outcomes


class _FailingPutStore(ArtifactStore):
    """版本记录写入阶段即失败的 ArtifactStore（模拟版本存储不可用）。"""

    def __init__(self, root) -> None:
        super().__init__(FilesystemArtifactStorageAdapter(root_dir=str(root)))
        self.put_calls = 0

    async def put(self, artifact):
        self.put_calls += 1
        raise RuntimeError("版本存储写入失败（测试注入）")


def _atom(content: str, *, version: int = 1) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1", version=version),
        index=IndexLayer(
            title="History memory",
            summary="A memory used to verify history commit consistency.",
            tags=["t1"],
            memory_type=MemoryType.FACT,
            alias="history_memory",
        ),
        payload=PayloadLayer(content=content),
    )


def _update_outcome(before: MemoryAtom, new_content: str) -> GenerationOutcome:
    """模拟引擎的 UPDATE 纯计算结果：内容合并 + 修改前完整原子快照。"""
    atom = before.model_copy(deep=True)
    atom.payload.content = new_content
    return GenerationOutcome(
        atom=atom,
        duplicate_decision=DuplicateDecision.UPDATE,
        memory_before_snapshot=before.model_copy(deep=True),
        changelog="merge to new content",
    )


def _spec(identity_scope: IdentityScope) -> MemoryGenerationTaskSpec:
    return MemoryGenerationTaskSpec(
        identity_scope=identity_scope,
        topic_id="t1",
        label="history-test",
        source=MemoryGenerationSource.WRITE,
        request=GenerationRequest(context=GenerationContext()),
        interaction_input=None,
    )


def _artifact_engine(tmp_path) -> ArtifactEngine:
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    return ArtifactEngine.from_store(store=store)


def _familiar(outcomes: list[GenerationOutcome], artifact_engine, mid_term=None):
    mid_term = mid_term or _InMemoryMidTermPort()
    library = type("Library", (), {"mid_term": mid_term})()
    familiar = MemoryGenerationFamiliar(
        generation_engine=_StubGenerationEngine(outcomes),
        memory_library=library,
        artifact_engine=artifact_engine,
        now=lambda: FIXED_NOW,
    )
    return familiar, mid_term


@pytest.mark.asyncio
async def test_update_snapshot_captures_versioned_atom_before_own_refs(tmp_path):
    """快照捕获时点：含分配后的版本/内容时间，不含自身 ref 与 VERSIONED 事件。"""
    identity_scope = make_memory_identity_scope()
    before = _atom("v1 content")
    # 预置一条创建期事件：§5.1 要求快照保留既有 refs/events。
    before.payload.artifacts.events.append(
        MemoryEventLog(event_type=MemoryEventType.CREATED, at=FIXED_NOW)
    )
    mid_term = _InMemoryMidTermPort()
    await mid_term.upsert(before)

    outcome = _update_outcome(before, "v2 content")
    familiar, mid_term = _familiar([outcome], _artifact_engine(tmp_path), mid_term)
    await familiar.execute(_spec(identity_scope))

    key = WorkspaceMemoryKey(
        workspace_identity=identity_scope.workspace_identity, memory_id=before.id
    )
    canonical = mid_term._atoms[key]
    assert canonical.meta.version == 2
    assert canonical.payload.content == "v2 content"

    # canonical 恰有一条 VERSIONED 事件并指向本次版本记录。
    versioned = [
        e for e in canonical.payload.artifacts.events if e.event_type == MemoryEventType.VERSIONED
    ]
    assert len(versioned) == 1
    version_ref = versioned[0].artifact_refs[0]

    # 读取版本记录本体（经 ArtifactStore 公共读取路径）。
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    version_data = await store.get(identity_scope, version_ref)
    assert version_data["version_number"] == 2
    assert version_data["snapshot_after"]["meta"]["version"] == 2
    assert version_data["snapshot_after"]["payload"]["content"] == "v2 content"
    # 快照不含本次 VERSIONED 事件（捕获先于自身事件追加，§5.1），
    # 但保留捕获时点已有的创建期事件。
    snapshot_events = version_data["snapshot_after"]["payload"]["artifacts"]["events"]
    assert not any(e["event_type"] == "versioned" for e in snapshot_events)
    assert [e["event_type"] for e in snapshot_events] == ["created"]
    # before 快照保存真实修改前内容。
    assert version_data["snapshot_before"]["payload"]["content"] == "v1 content"


@pytest.mark.asyncio
async def test_canonical_publish_failure_leaves_orphan_version_not_head(tmp_path):
    """canonical 写入失败：错误传播；已写入版本记录成为孤立记录，非 canonical head。"""
    identity_scope = make_memory_identity_scope()
    before = _atom("v1 content")
    mid_term = _InMemoryMidTermPort()
    await mid_term.upsert(before)
    mid_term.fail_upsert_for.add(str(before.id))

    outcome = _update_outcome(before, "v2 content")
    familiar, mid_term = _familiar([outcome], _artifact_engine(tmp_path), mid_term)

    with pytest.raises(RuntimeError, match="canonical 写入失败"):
        await familiar.execute(_spec(identity_scope))

    # 旧 canonical 仍有效：内容/版本未变，且没有新的 VERSIONED 事件。
    key = WorkspaceMemoryKey(
        workspace_identity=identity_scope.workspace_identity, memory_id=before.id
    )
    canonical = mid_term._atoms[key]
    assert canonical.meta.version == 1
    assert canonical.payload.content == "v1 content"
    assert not any(
        e.event_type == MemoryEventType.VERSIONED for e in canonical.payload.artifacts.events
    )

    # 已写入的版本记录保留为孤立记录：可被 Artifact 存储列出，但不被
    # canonical 引用（不是已提交版本）。
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    orphan_refs = await store.list_by_memory(identity_scope, str(before.id))
    assert orphan_refs, "已写入的版本记录应保留为孤立 Artifact"
    assert all(
        ref.artifact_id not in {r.artifact_id for r in canonical.payload.artifacts.refs}
        for ref in orphan_refs
    ), "孤立记录不得成为 canonical head 的关联引用"


@pytest.mark.asyncio
async def test_version_record_failure_aborts_before_canonical_publish(tmp_path):
    """版本存储写入失败：不发布 canonical，upsert 未被调用。"""
    identity_scope = make_memory_identity_scope()
    before = _atom("v1 content")
    mid_term = _InMemoryMidTermPort()
    await mid_term.upsert(before)
    mid_term.upsert_calls.clear()  # 只统计本次提交的发布尝试。

    failing_store = _FailingPutStore(tmp_path / "failing_artifacts")
    outcome = _update_outcome(before, "v2 content")
    familiar, mid_term = _familiar(
        [outcome], ArtifactEngine.from_store(store=failing_store), mid_term
    )

    with pytest.raises(RuntimeError, match="版本存储写入失败"):
        await familiar.execute(_spec(identity_scope))

    assert mid_term.upsert_calls == [], "版本记录失败后不得发布 canonical"
    assert failing_store.put_calls >= 1


@pytest.mark.asyncio
async def test_cancelled_canonical_publish_propagates(tmp_path):
    """提交被取消：CancelledError 原样传播，不吞成普通失败。"""
    identity_scope = make_memory_identity_scope()
    before = _atom("v1 content")
    mid_term = _InMemoryMidTermPort()
    await mid_term.upsert(before)
    mid_term.fail_upsert_for.add(str(before.id))
    mid_term.upsert_error = asyncio.CancelledError()

    outcome = _update_outcome(before, "v2 content")
    familiar, _ = _familiar([outcome], _artifact_engine(tmp_path), mid_term)

    with pytest.raises(asyncio.CancelledError):
        await familiar.execute(_spec(identity_scope))


@pytest.mark.asyncio
async def test_history_immutable_after_post_commit_mutation(tmp_path):
    """历史写入后不可变：提交后修改持有对象不影响已存历史。"""
    identity_scope = make_memory_identity_scope()
    before = _atom("v1 content")
    mid_term = _InMemoryMidTermPort()
    await mid_term.upsert(before)

    outcome = _update_outcome(before, "v2 content")
    familiar, mid_term = _familiar([outcome], _artifact_engine(tmp_path), mid_term)
    await familiar.execute(_spec(identity_scope))

    key = WorkspaceMemoryKey(
        workspace_identity=identity_scope.workspace_identity, memory_id=before.id
    )
    canonical = mid_term._atoms[key]
    versioned = [
        e for e in canonical.payload.artifacts.events if e.event_type == MemoryEventType.VERSIONED
    ][0]
    version_ref = versioned.artifact_refs[0]

    # 提交后修改 canonical 原子（内容、标题、动态状态）。
    canonical.payload.content = "mutated after commit"
    canonical.index.title = "mutated title"
    canonical.meta.lifecycle.vitality_score = 1.0

    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    version_data = await store.get(identity_scope, version_ref)
    assert version_data["snapshot_after"]["payload"]["content"] == "v2 content"
    assert version_data["snapshot_after"]["index"]["title"] == "History memory"
    assert version_data["snapshot_after"]["meta"]["lifecycle"]["vitality_score"] != 1.0


@pytest.mark.asyncio
async def test_noop_version_store_fails_content_creation(tmp_path):
    """版本存储未配置（NoOp builder）：内容 create 明确失败，不静默成功。"""
    identity_scope = make_memory_identity_scope()
    draft = _atom("new content")

    familiar, mid_term = _familiar(
        [
            GenerationOutcome(
                atom=draft,
                duplicate_decision=DuplicateDecision.CREATE,
            )
        ],
        ArtifactEngine.noop(),
    )
    with pytest.raises(RuntimeError, match="版本存储未产生"):
        await familiar.execute(_spec(identity_scope))

    assert mid_term.upsert_calls == []
