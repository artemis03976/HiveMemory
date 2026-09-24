"""真实 Qdrant 内存模式下验证 Memory 的 Workspace 复合寻址与读取隔离。"""

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    build_internal_identity_scope,
)
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from tests.helpers.memory import make_memory_metadata


class _DeterministicEmbedding:
    """只替换测试边界外的 embedding provider。"""

    def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


@pytest_asyncio.fixture
async def memory_store():
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(location=":memory:")
    qdrant.collection_name = "memory_workspace_isolation"
    qdrant.vector_dimension = 2
    qdrant.embedding_service = _DeterministicEmbedding()
    await qdrant.client.create_collection(
        collection_name=qdrant.collection_name,
        vectors_config={
            "dense_text": VectorParams(size=2, distance=Distance.COSINE),
        },
    )
    store = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))
    try:
        yield store, qdrant
    finally:
        await qdrant.client.close()


def _identity_scope(
    workspace_id: str,
    *,
    user_id: str = "u1",
    agent_id: str = "agent-a",
):
    return build_internal_identity_scope(
        ActorIdentity(user_id=user_id, agent_id=agent_id, team_id="team-a"),
        workspace_id,
    )


def _memory(identity_scope, *, memory_id: UUID, content: str, alias: str) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id,
        meta=make_memory_metadata(
            user_id=identity_scope.actor_identity.user_id,
            source_agent_id=identity_scope.actor_identity.agent_id,
            team_id=identity_scope.actor_identity.team_id,
            workspace_id=identity_scope.workspace_identity.workspace_id,
        ),
        index=IndexLayer(
            title="Collision memory",
            summary="Two workspaces intentionally reuse every opaque identifier.",
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


@pytest.mark.asyncio
async def test_same_uuid_and_alias_are_independent_between_workspaces(memory_store) -> None:
    """捕获 Qdrant 仍以裸 UUID/alias 全局寻址、导致覆盖或串读的缺陷。"""
    store, _ = memory_store
    main = _identity_scope("main_workspace")
    isolation = _identity_scope("isolation_workspace")
    shared_id = uuid4()
    await store.upsert(
        _memory(main, memory_id=shared_id, content="main content", alias="fact_collision")
    )
    await store.upsert(
        _memory(
            isolation,
            memory_id=shared_id,
            content="isolation content",
            alias="fact_collision",
        )
    )

    main_by_id = await store.get(main, shared_id)
    isolation_by_id = await store.get(isolation, shared_id)
    main_by_alias = await store.get_by_alias(main, "fact_collision")
    isolation_by_alias = await store.get_by_alias(isolation, "fact_collision")

    assert main_by_id.payload.content == "main content"
    assert isolation_by_id.payload.content == "isolation content"
    assert main_by_alias.payload.content == "main content"
    assert isolation_by_alias.payload.content == "isolation content"


@pytest.mark.asyncio
async def test_public_memory_is_not_visible_from_another_workspace(memory_store) -> None:
    """捕获 PUBLIC policy 在 ownership hard filter 之前生效的缺陷。"""
    store, _ = memory_store
    main = _identity_scope("main_workspace")
    isolation = _identity_scope("isolation_workspace")
    memory_id = uuid4()
    await store.upsert(_memory(main, memory_id=memory_id, content="main only", alias="fact_public"))

    assert await store.get(isolation, memory_id) is None
    assert await store.get_by_alias(isolation, "fact_public") is None
