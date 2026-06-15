"""
MemoryArtifactBuilder 单元测试

覆盖：
- build_for_create: v1 VersionArtifact 先写, CreationArtifact 后写, initial_version_ref 正确
- build_for_create: MemoryCreationArtifact 不含 alias/title/tags
- build_for_create: snapshot_after 保存全量可变字段
- build_for_update: snapshot_before/after 正确, 返回 ArtifactRef
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from hivememory.core.models.artifact import (
    ArtifactRef, ArtifactType,
    MemoryCreationArtifact, MemoryVersionArtifact, MemoryVersionSnapshot,
)
from hivememory.core.models.pending import Identity, PendingAtomMaterializeTask, UpdateFocus
from hivememory.engines.artifacts.memory import MemoryArtifactBuilder, MemoryCreationBundle
from hivememory.engines.generation.models import DuplicateDecision, GenerationContext, MemoryGenerationResult
from hivememory.patchouli.services.memory_generation_tasks import MemoryGenerationTaskController


def _ref(artifact_type: ArtifactType = ArtifactType.MEMORY_VERSION) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(uuid4()),
        artifact_type=artifact_type,
        uri="/tmp/fake.json",
        sha256="abc123",
        created_at=datetime.now(),
    )


def _make_atom():
    from hivememory.core.models.memory import MemoryAtom, IndexLayer, PayloadLayer, MetaData, Artifacts
    from hivememory.core.models import MemoryType, Identity
    atom = MagicMock()
    atom.id = uuid4()
    atom.meta = MagicMock()
    atom.meta.version = 1
    atom.index = MagicMock()
    atom.index.alias = "my-alias"
    atom.index.title = "Test Title"
    atom.index.summary = "Test summary"
    atom.index.tags = ["tag1", "tag2"]
    atom.index.memory_type = MagicMock()
    atom.index.memory_type.value = "FACT"
    atom.payload = MagicMock()
    atom.payload.content = "Initial content"
    return atom


def _make_context() -> GenerationContext:
    ctx = MagicMock(spec=GenerationContext)
    ctx.model_dump.return_value = {"turns": [], "state_summary": ""}
    return ctx


@pytest.fixture
def store():
    """Mock ArtifactStore that records put_json calls."""
    store = MagicMock()
    call_order = []

    async def put_json(artifact, *, namespace=None):
        call_order.append(type(artifact).__name__)
        ref = _ref(artifact.artifact_type)
        return ref

    store.put_json = AsyncMock(side_effect=put_json)
    store._call_order = call_order
    return store


@pytest.fixture
def builder(store):
    return MemoryArtifactBuilder(store)


# ── build_for_create ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_for_create_returns_bundle(builder):
    atom = _make_atom()
    bundle = await builder.build_for_create(
        memory=atom,
        context=_make_context(),
        source_intent="WRITE",
        source_artifact_refs=[],
    )
    assert isinstance(bundle, MemoryCreationBundle)
    assert bundle.creation_ref is not None
    assert bundle.initial_version_ref is not None


@pytest.mark.asyncio
async def test_build_for_create_v1_written_before_creation(builder, store):
    """v1 MemoryVersionArtifact 必须先于 MemoryCreationArtifact 写入。"""
    atom = _make_atom()
    await builder.build_for_create(
        memory=atom, context=_make_context(),
        source_intent="ARCHIVE", source_artifact_refs=[],
    )
    assert store._call_order == ["MemoryVersionArtifact", "MemoryCreationArtifact"]


@pytest.mark.asyncio
async def test_build_for_create_initial_version_ref_set(builder, store):
    """MemoryCreationArtifact.initial_version_ref 指向 v1 ref。"""
    written = []

    async def capture(artifact, *, namespace=None):
        written.append(artifact)
        return _ref(artifact.artifact_type)

    store.put_json = AsyncMock(side_effect=capture)
    atom = _make_atom()
    bundle = await builder.build_for_create(
        memory=atom, context=_make_context(),
        source_intent="WRITE", source_artifact_refs=[],
    )
    creation: MemoryCreationArtifact = written[1]
    assert creation.initial_version_ref is not None
    assert creation.initial_version_ref.artifact_id == bundle.initial_version_ref.artifact_id


@pytest.mark.asyncio
async def test_build_for_create_snapshot_captures_all_fields(builder, store):
    """v1 snapshot_after 包含所有可变字段。"""
    captured = []

    async def capture(artifact, *, namespace=None):
        captured.append(artifact)
        return _ref(artifact.artifact_type)

    store.put_json = AsyncMock(side_effect=capture)
    atom = _make_atom()
    await builder.build_for_create(
        memory=atom, context=_make_context(),
        source_intent="WRITE", source_artifact_refs=[],
    )
    v1: MemoryVersionArtifact = captured[0]
    snap = v1.snapshot_after
    assert snap.content == "Initial content"
    assert snap.alias == "my-alias"
    assert snap.title == "Test Title"
    assert snap.tags == ["tag1", "tag2"]
    assert snap.memory_type == "FACT"
    assert v1.snapshot_before is None


# ── build_for_update ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_for_update_returns_ref(builder):
    atom = _make_atom()
    atom.meta.version = 2
    snapshot_before = MemoryVersionSnapshot(content="old content", title="Old Title")
    ref = await builder.build_for_update(
        memory_after=atom,
        snapshot_before=snapshot_before,
        update_source="UPDATE",
        changelog="Updated reason",
    )
    assert ref is not None
    assert ref.artifact_type == ArtifactType.MEMORY_VERSION


@pytest.mark.asyncio
async def test_build_for_update_snapshot_before_and_after(builder, store):
    captured = []

    async def capture(artifact, *, namespace=None):
        captured.append(artifact)
        return _ref(artifact.artifact_type)

    store.put_json = AsyncMock(side_effect=capture)
    atom = _make_atom()
    atom.meta.version = 3
    snapshot_before = MemoryVersionSnapshot(content="old", title="Old")
    await builder.build_for_update(
        memory_after=atom,
        snapshot_before=snapshot_before,
        update_source="MERGE",
    )
    v: MemoryVersionArtifact = captured[0]
    assert v.snapshot_before.content == "old"
    assert v.snapshot_after.content == "Initial content"
    assert v.version_number == 3
    assert v.update_source == "MERGE"


@pytest.mark.asyncio
async def test_update_provenance_uses_update_intent_and_interaction_ref(store):
    builder = MemoryArtifactBuilder(store)
    artifact_engine = MagicMock()
    artifact_engine.memory = builder
    controller = MemoryGenerationTaskController(
        storage=MagicMock(),
        generation_engine=MagicMock(),
        artifact_engine=artifact_engine,
    )
    atom = _make_atom()
    atom.payload.artifacts.full_history = [{"reason": "merged new facts"}]
    atom.payload.artifacts.refs = []
    atom.payload.artifacts.provenance = []
    interaction_ref = ArtifactRef(
        artifact_id="interaction-1",
        artifact_type=ArtifactType.INTERACTION,
    )
    result = MemoryGenerationResult(
        atom=atom,
        duplicate_decision=DuplicateDecision.UPDATE,
        memory_before_snapshot=MemoryVersionSnapshot(content="old"),
    )

    await controller._build_memory_artifacts(
        [result],
        _make_context(),
        "UPDATE",
        interaction_ref,
    )

    provenance = atom.payload.artifacts.provenance[-1]
    assert provenance.action == "updated"
    assert provenance.source_intent == "UPDATE"
    assert provenance.source_artifacts == [interaction_ref]
    assert atom.payload.artifacts.full_history[-1]["artifact_refs"]


@pytest.mark.asyncio
async def test_run_mode_c_passes_update_source_intent():
    atom = _make_atom()
    storage = MagicMock()
    storage.get_memory = AsyncMock(return_value=atom)
    controller = MemoryGenerationTaskController(
        storage=storage,
        generation_engine=MagicMock(),
    )
    controller._run_generation = AsyncMock(return_value=[])
    interaction_ref = ArtifactRef(
        artifact_id="interaction-1",
        artifact_type=ArtifactType.INTERACTION,
    )
    task = PendingAtomMaterializeTask(
        pending_alias="draft_update",
        intent_id="intent_update",
        source_verb="UPDATE",
        identity=Identity(user_id="u1"),
        focus=UpdateFocus(
            instruction="merge",
            base_uuid=str(atom.id),
            base_alias="fact_test",
        ),
    )

    await controller._run_mode_c(task, _make_context(), interaction_ref)

    _, kwargs = controller._run_generation.call_args
    assert kwargs["source_intent"] == "UPDATE"
    assert kwargs["interaction_ref"] == interaction_ref
