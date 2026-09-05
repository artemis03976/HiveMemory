"""Artifact 复合归属、受控 legacy 读取与 builder 组合测试。"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import (
    ActorIdentity,
    MemoryCreationArtifact,
    MemoryEventLog,
    MemoryEventType,
    WorkspaceIdentity,
)
from hivememory.core.models.artifact import ArtifactRef, InteractionArtifact
from hivememory.core.models.memory import Artifacts
from hivememory.engines.artifacts.document import (
    DocumentArtifactBuilder,
    NoOpDocumentArtifactBuilder,
)
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.artifacts.interaction import InteractionArtifactBuilder
from hivememory.engines.artifacts.memory import MemoryArtifactBuilder
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.stores import ArtifactStore
from hivememory.system.config.patchouli import ArtifactComponentConfig, ArtifactConfig
from tests.helpers.memory import make_memory_identity_scope
from tests.helpers.workspace import make_identity_scope


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))


def _make_artifact(
    workspace: WorkspaceIdentity,
    artifact_id: str = "test-id",
    *,
    topic_id: str = "topic-1",
) -> InteractionArtifact:
    return InteractionArtifact(
        artifact_id=artifact_id,
        created_at=datetime(2026, 6, 14, 12, 0, 0),
        workspace_identity=workspace,
        owner_agent_id="agent-1",
        topic_id=topic_id,
        captured_at=datetime(2026, 6, 14, 12, 0, 0),
    )


def _identity_scope(*, user_id: str = "u1", workspace_id: str = "main_workspace"):
    return make_identity_scope(
        actor_identity=ActorIdentity(user_id=user_id, agent_id="agent-1"),
        workspace_id=workspace_id,
        interaction_id=f"i-{user_id}-{workspace_id}",
    )


@pytest.mark.asyncio
async def test_put_and_get_roundtrip_uses_workspace_scoped_ref(store):
    identity_scope = _identity_scope()
    ref = await store.put(_make_artifact(identity_scope.workspace_identity))

    data = await store.get(identity_scope, ref)

    assert data["artifact_id"] == "test-id"
    assert data["topic_id"] == "topic-1"
    assert data["workspace_identity"]["workspace_id"] == "main_workspace"
    assert ref.workspace_identity == identity_scope.workspace_identity


@pytest.mark.asyncio
async def test_artifact_reads_require_a_complete_identity_scope(store):
    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        await store.get(None, "test-id")


@pytest.mark.asyncio
async def test_same_artifact_id_is_independent_between_workspaces(store):
    main = _identity_scope(workspace_id="main_workspace")
    isolated = _identity_scope(workspace_id="isolation_workspace")
    await store.put(_make_artifact(main.workspace_identity, topic_id="main-topic"))
    await store.put(
        _make_artifact(isolated.workspace_identity, topic_id="isolated-topic")
    )

    main_data = await store.get(main, "test-id")
    isolated_data = await store.get(isolated, "test-id")

    assert main_data["topic_id"] == "main-topic"
    assert isolated_data["topic_id"] == "isolated-topic"


@pytest.mark.asyncio
async def test_same_artifact_id_is_independent_between_owners(store):
    first = _identity_scope(user_id="u1")
    second = _identity_scope(user_id="u2")
    await store.put(_make_artifact(first.workspace_identity, topic_id="u1-topic"))
    await store.put(_make_artifact(second.workspace_identity, topic_id="u2-topic"))

    assert (await store.get(first, "test-id"))["topic_id"] == "u1-topic"
    assert (await store.get(second, "test-id"))["topic_id"] == "u2-topic"


@pytest.mark.asyncio
async def test_cross_workspace_ref_is_projected_as_not_found(store):
    main = _identity_scope()
    isolated = _identity_scope(workspace_id="isolation_workspace")
    ref = await store.put(_make_artifact(main.workspace_identity))

    with pytest.raises(FileNotFoundError, match="artifact not found"):
        await store.get(isolated, ref)
    assert await store.exists(isolated, ref.artifact_id) is False
    assert (await store.verify(isolated, ref)).ok is False


@pytest.mark.asyncio
async def test_ref_uri_never_selects_a_different_physical_file(store, tmp_path):
    main = _identity_scope()
    isolated = _identity_scope(workspace_id="isolation_workspace")
    main_ref = await store.put(_make_artifact(main.workspace_identity, topic_id="main"))
    isolated_ref = await store.put(
        _make_artifact(isolated.workspace_identity, topic_id="isolated")
    )
    forged = main_ref.model_copy(update={"uri": isolated_ref.uri})

    data = await store.get(main, forged)

    assert data["topic_id"] == "main"
    assert str(tmp_path) in main_ref.uri


@pytest.mark.asyncio
async def test_append_only_rejects_different_content_in_same_scope(store):
    identity_scope = _identity_scope()
    await store.put(_make_artifact(identity_scope.workspace_identity, topic_id="first"))

    with pytest.raises(ValueError, match="append-only"):
        await store.put(_make_artifact(identity_scope.workspace_identity, topic_id="second"))


@pytest.mark.asyncio
async def test_same_content_replay_is_idempotent(store):
    identity_scope = _identity_scope()
    artifact = _make_artifact(identity_scope.workspace_identity)

    first = await store.put(artifact)
    replay = await store.put(artifact.model_copy(deep=True))

    assert replay.artifact_id == first.artifact_id
    assert replay.sha256 == first.sha256
    assert await store.exists(identity_scope, first.artifact_id)


@pytest.mark.asyncio
async def test_list_by_memory_is_workspace_scoped(store):
    main = _identity_scope()
    isolated = _identity_scope(workspace_id="isolation_workspace")
    for identity_scope, artifact_id in ((main, "main-artifact"), (isolated, "isolated-artifact")):
        await store.put(
            MemoryCreationArtifact(
                artifact_id=artifact_id,
                workspace_identity=identity_scope.workspace_identity,
                owner_agent_id=identity_scope.actor_identity.agent_id,
                memory_id="memory-1",
                source_intent="WRITE",
            )
        )

    refs = await store.list_by_memory(main, "memory-1")

    assert [ref.artifact_id for ref in refs] == ["main-artifact"]
    assert refs[0].workspace_identity == main.workspace_identity


@pytest.mark.asyncio
async def test_tampered_content_fails_get_and_verify(store):
    identity_scope = _identity_scope()
    ref = await store.put(_make_artifact(identity_scope.workspace_identity))
    path = Path(ref.uri)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["topic_id"] = "tampered"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        await store.get(identity_scope, ref)
    assert (await store.verify(identity_scope, ref)).ok is False


def _write_legacy_artifact(root: Path, *, artifact_id: str, owner_user_id: str) -> None:
    path = root / "interaction" / "2025" / "01" / f"{artifact_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "artifact_id": artifact_id,
        "artifact_type": "interaction",
        "created_at": "2025-01-01T00:00:00",
        "owner_user_id": owner_user_id,
        "topic_id": "legacy-topic",
        "content_hash": None,
    }
    data["content_hash"] = hashlib.sha256(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_legacy_artifact_is_visible_only_in_owner_main_workspace(tmp_path):
    _write_legacy_artifact(tmp_path, artifact_id="legacy-1", owner_user_id="u1")
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))
    main = _identity_scope(user_id="u1")
    isolated = _identity_scope(user_id="u1", workspace_id="isolation_workspace")

    data = await store.get(main, "legacy-1")

    assert data["topic_id"] == "legacy-topic"
    assert await store.exists(isolated, "legacy-1") is False


@pytest.mark.asyncio
async def test_partial_workspace_legacy_record_is_rejected(tmp_path):
    path = tmp_path / "old" / "partial.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_id": "partial-1",
                "artifact_type": "interaction",
                "owner_user_id": "u1",
                "workspace_id": "isolation_workspace",
                "content_hash": None,
            }
        ),
        encoding="utf-8",
    )
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))

    with pytest.raises(FileNotFoundError, match="artifact not found"):
        await store.get(_identity_scope(user_id="u1"), "partial-1")


def test_old_artifacts_payload_ignores_removed_legacy_fields():
    """v0.4 旧字段已被结构化 artifact 体系替代，反序列化时忽略。"""
    old = {
        "raw_source_url": "https://example.com",
        "file_path": "/tmp/doc.md",
        "context_ref": [{"session_id": "s1", "msg_id": "m1"}],
        "full_history": [{"timestamp": "2026-01-01", "content": "v1", "reason": "init"}],
    }
    artifacts = Artifacts.model_validate(old)

    assert not hasattr(artifacts, "raw_source_url")
    assert not hasattr(artifacts, "file_path")
    assert not hasattr(artifacts, "context_ref")
    assert not hasattr(artifacts, "full_history")
    assert artifacts.refs == []
    assert artifacts.events == []
    assert artifacts.cold_archive_uri is None
    assert artifacts.cold_archive_hash is None
    assert artifacts.revival_keys == []


@pytest.mark.asyncio
async def test_artifact_builder_and_engine_preserve_workspace(tmp_path):
    identity_scope = _identity_scope()
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))
    engine = ArtifactEngine.from_store(store)
    ref = await engine.document.build_and_store(
        source_type="markdown",
        source_uri="memory://source",
        content_hash=None,
        retrieved_at=datetime(2026, 1, 1),
        workspace_identity=identity_scope.workspace_identity,
        owner_agent_id=identity_scope.actor_identity.agent_id,
    )

    assert ref is not None
    stored = await store.get(identity_scope, ref)
    assert stored["workspace_identity"] == identity_scope.workspace_identity.model_dump()


def test_artifacts_payload_deserializes_scoped_refs_and_events_as_models():
    workspace = WorkspaceIdentity(
        owner_user_id="u1",
        workspace_key="main_workspace",
        workspace_id="main_workspace",
    )
    payload = {
        "refs": [
            {
                "artifact_id": "interaction-1",
                "artifact_type": "interaction",
                "workspace_identity": workspace.model_dump(),
                "uri": "/tmp/interaction.json",
                "sha256": "abc",
            }
        ],
        "events": [
            {
                "event_type": "versioned",
                "artifact_refs": [
                    {
                        "artifact_id": "interaction-1",
                        "artifact_type": "interaction",
                        "workspace_identity": workspace.model_dump(),
                    }
                ],
            }
        ],
    }

    artifacts = Artifacts.model_validate(payload)

    assert isinstance(artifacts.refs[0], ArtifactRef)
    assert artifacts.refs[0].workspace_identity == workspace
    assert isinstance(artifacts.events[0], MemoryEventLog)
    assert artifacts.events[0].event_type == MemoryEventType.VERSIONED
    assert isinstance(artifacts.events[0].artifact_refs[0], ArtifactRef)


def test_artifact_engine_holds_all_builders(tmp_path):
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))
    engine = ArtifactEngine.from_store(store)

    assert isinstance(engine.interaction, InteractionArtifactBuilder)
    assert isinstance(engine.document, DocumentArtifactBuilder)
    assert isinstance(engine.memory, MemoryArtifactBuilder)


def test_artifact_engine_from_store_honors_component_config(tmp_path):
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))
    config = ArtifactConfig(document=ArtifactComponentConfig(enabled=False))

    engine = ArtifactEngine.from_store(store, config=config)

    assert isinstance(engine.interaction, InteractionArtifactBuilder)
    assert isinstance(engine.memory, MemoryArtifactBuilder)
    assert isinstance(engine.document, NoOpDocumentArtifactBuilder)


@pytest.mark.asyncio
async def test_noop_artifact_engine_returns_empty_results():
    engine = ArtifactEngine.noop()
    context = make_memory_identity_scope()

    interaction_ref = await engine.interaction.build_and_store(
        topic_id="topic-1",
        blocks=[],
        identity_scope=context,
    )
    memory_bundle = await engine.memory.build_for_create(
        memory=object(),
        context=object(),
        source_intent="WRITE",
        source_artifact_refs=[],
    )
    version_ref = await engine.memory.build_for_update(
        memory_after=object(),
        update_source="UPDATE",
    )

    assert interaction_ref is None
    assert memory_bundle.refs == []
    assert version_ref is None
