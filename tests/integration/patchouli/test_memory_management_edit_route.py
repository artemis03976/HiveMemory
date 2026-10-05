"""真实公开/本地总线、记忆编辑服务与存储之间的编辑路由契约。"""

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.engines.artifacts import ArtifactEngine
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.application.memory_management_service import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.memory_library import (
    ArtifactStore,
    LongTermMemoryStore,
    MemoryLibrary,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.memory_library.adapters.artifact import FilesystemArtifactStorageAdapter
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope

COMMIT_NOW = datetime(2026, 10, 4, 12, tzinfo=UTC)


class _DeterministicEmbedding:
    """只替换协作边界外的 embedding provider，避免下载与网络请求。"""

    def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


class _UnusedGenerationEngine:
    """外部管理编辑不执行生成计算。"""

    async def process(self, *args, **kwargs):
        raise AssertionError("external memory edit must not run generation")


@pytest_asyncio.fixture
async def edit_route(tmp_path):
    """绑定真实路由 handler，使用 Qdrant 内存模式和临时 Artifact 存储。"""
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(location=":memory:")
    qdrant.collection_name = "memory_management_edit_route"
    qdrant.vector_dimension = 2
    qdrant.embedding_service = _DeterministicEmbedding()
    await qdrant.client.create_collection(
        collection_name=qdrant.collection_name,
        vectors_config={"dense_text": VectorParams(size=2, distance=Distance.COSINE)},
    )
    mid_term = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))
    artifact_store = ArtifactStore(
        FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts"))
    )
    library = MemoryLibrary(
        short_term=ShortTermMemoryStore(),
        mid_term=mid_term,
        long_term=LongTermMemoryStore(
            FileBasedStorageAdapter(archive_dir=str(tmp_path / "archive"), compress=False)
        ),
        artifact_store=artifact_store,
    )
    familiar = MemoryGenerationFamiliar(
        generation_engine=_UnusedGenerationEngine(),
        memory_library=library,
        artifact_engine=ArtifactEngine.from_store(artifact_store),
        now=lambda: COMMIT_NOW,
    )
    local_bus = PatchouliBus()
    local_bus.register(PatchouliLocalRoutes.MEMORY_UPDATE, familiar.update_external_memory)
    application = MemoryManagementService(bus=local_bus)
    global_bus = GlobalSystemBus()
    global_bus.register(PatchouliRoutes.MEMORY_UPDATE, application.update_memory)
    try:
        yield global_bus, mid_term
    finally:
        global_bus.unregister(PatchouliRoutes.MEMORY_UPDATE)
        local_bus.unregister(PatchouliLocalRoutes.MEMORY_UPDATE)
        await qdrant.client.close()


@pytest.mark.asyncio
async def test_public_memory_edit_persists_content_through_real_route_handlers(edit_route):
    """管理编辑只需归属，参数拆分后仍经真实 handler 写入内容与版本。"""
    global_bus, mid_term = edit_route
    scope = make_identity_scope(user_id="u1", agent_id="manager-agent")
    policy = MemoryAccessPolicy(visibility=MemoryVisibility.PRIVATE, target_agent_id="author-agent")
    memory = MemoryAtom(
        meta=make_memory_metadata(
            user_id="u1", source_agent_id="author-agent", access_policy=policy
        ),
        index=IndexLayer(
            title="Original title",
            summary="Memory edited through the public management route.",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="original content"),
    )
    await mid_term.upsert(memory)

    await global_bus.request(
        PatchouliRoutes.MEMORY_UPDATE,
        str(memory.id),
        identity_scope=scope,
        title="Edited title",
        content="edited content",
    )
    persisted = await mid_term.get_by_key(
        WorkspaceMemoryKey(workspace_identity=scope.workspace_identity, memory_id=memory.id)
    )

    assert persisted.index.title == "Edited title"
    assert persisted.payload.content == "edited content"
    assert persisted.meta.version == 2
    assert persisted.meta.updated_at == COMMIT_NOW
    assert persisted.meta.access_policy == policy
    assert persisted.meta.provenance.source_agent_id == "author-agent"
