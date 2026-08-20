"""Memory Artifact Builder 的 Workspace 归属与 provenance 行为测试。"""

from datetime import datetime
from uuid import uuid4

import pytest

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.models.artifact import (
    ArtifactType,
    InteractionArtifact,
    MemoryVersionSnapshot,
)
from hivememory.engines.artifacts.memory import MemoryArtifactBuilder, MemoryCreationBundle
from hivememory.engines.generation.models import GenerationContext
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.stores import ArtifactStore
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_context


@pytest.fixture
def access():
    return make_access_context(
        user_id="u1",
        agent_id="source-agent",
        workspace_id="main_workspace",
    )


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))


def _make_atom(access, *, memory_id=None, source_agent_id="source-agent") -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=make_memory_metadata(
            source_agent_id=source_agent_id,
            user_id=access.workspace_identity.owner_user_id,
            workspace_id=access.workspace_identity.workspace_id,
        ),
        index=IndexLayer(
            title="Test Title",
            summary="A test memory summary with enough detail",
            tags=["tag1", "tag2"],
            memory_type=MemoryType.FACT,
            alias="my-alias",
        ),
        payload=PayloadLayer(content="Initial content"),
    )


@pytest.mark.asyncio
async def test_build_for_create_persists_scoped_artifacts_and_links_initial_version(
    store,
    access,
):
    atom = _make_atom(access)
    builder = MemoryArtifactBuilder(store)

    bundle = await builder.build_for_create(
        memory=atom,
        context=GenerationContext(),
        source_intent="WRITE",
        source_artifact_refs=[],
    )

    assert isinstance(bundle, MemoryCreationBundle)
    version = await store.get(access, bundle.initial_version_ref)
    creation = await store.get(access, bundle.creation_ref)
    assert version["artifact_type"] == ArtifactType.MEMORY_VERSION.value
    assert creation["artifact_type"] == ArtifactType.MEMORY_CREATION.value
    assert version["workspace_identity"] == access.workspace_identity.model_dump()
    assert creation["initial_version_ref"]["artifact_id"] == bundle.initial_version_ref.artifact_id
    assert version["owner_agent_id"] == "source-agent"
    assert creation["owner_agent_id"] == "source-agent"
    assert version["snapshot_after"]["content"] == "Initial content"
    assert version["snapshot_after"]["alias"] == "my-alias"
    assert version["snapshot_after"]["title"] == "Test Title"
    assert set(version["snapshot_after"]["tags"]) == {"tag1", "tag2"}
    assert version["snapshot_after"]["memory_type"] == "FACT"
    assert creation["title"] == ""
    assert "alias" not in creation
    assert "tags" not in creation


@pytest.mark.asyncio
async def test_build_for_update_keeps_memory_provenance_and_scope(store, access):
    atom = _make_atom(access)
    atom.meta.version = 3
    builder = MemoryArtifactBuilder(store)

    ref = await builder.build_for_update(
        memory_after=atom,
        snapshot_before=MemoryVersionSnapshot(content="old", title="Old"),
        update_source="MERGE",
        changelog="Updated reason",
    )

    data = await store.get(access, ref)
    assert data["artifact_type"] == ArtifactType.MEMORY_VERSION.value
    assert data["version_number"] == 3
    assert data["update_source"] == "MERGE"
    assert data["workspace_identity"] == access.workspace_identity.model_dump()
    assert data["owner_agent_id"] == atom.meta.source_agent_id


@pytest.mark.asyncio
async def test_builder_rejects_source_ref_from_another_workspace(store, access):
    other = make_access_context(
        user_id="u1",
        agent_id="other-agent",
        workspace_id="isolation_workspace",
    )
    source = await store.put(
        InteractionArtifact(
            artifact_id="source-artifact",
            workspace_identity=other.workspace_identity,
            owner_agent_id="other-agent",
            topic_id="other-topic",
            created_at=datetime(2026, 1, 1),
        )
    )
    atom = _make_atom(access)
    builder = MemoryArtifactBuilder(store)

    with pytest.raises(WorkspaceMismatchError, match="workspace.mismatch"):
        await builder.build_for_create(
            memory=atom,
            context=GenerationContext(),
            source_intent="WRITE",
            source_artifact_refs=[source],
        )

    assert await store.list_by_memory(access, str(atom.id)) == []
