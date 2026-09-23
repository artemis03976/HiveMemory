"""真实 Qdrant 内存模式下验证 Memory 的 Workspace 复合寻址与兼容读取。"""

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    build_internal_identity_scope,
)
from hivememory.core.mtp.exceptions import StorageReadError
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


@pytest.mark.asyncio
async def test_legacy_record_without_schema_version_is_no_longer_readable(memory_store) -> None:
    """legacy v1 解释分支已删除：缺 schema_version 的记录 fail closed 而非被解释。

    存量 legacy 数据由迁移工具转换为 canonical v2
    （scripts/migrate_v1_memory_and_artifacts.py）；未迁移记录不再被任何
    Workspace 解释读取，也不得被第二 Workspace 的 compatibility-read 召回。
    """
    store, qdrant = memory_store
    main = _identity_scope("main_workspace")
    isolation = _identity_scope("isolation_workspace")
    memory_id = uuid4()
    legacy_payload = {
        "id": str(memory_id),
        "meta": {
            "source_agent_id": "agent-a",
            "user_id": "u1",
            "visibility": "PUBLIC",
        },
        "index": {
            "title": "Legacy main memory",
            "summary": "Legacy record has no Workspace projection and stays main-only.",
            "tags": [],
            "memory_type": "FACT",
            "alias": "fact_legacy",
        },
        "payload": {"content": "legacy main"},
        "relations": {},
    }
    await qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=str(memory_id),
                vector={"dense_text": [0.25, 0.75]},
                payload=legacy_payload,
            )
        ],
    )

    with pytest.raises(StorageReadError):
        await store.get(main, memory_id)
    with pytest.raises(StorageReadError):
        await store.get(isolation, memory_id)


@pytest.mark.asyncio
async def test_legacy_uuid_cleanup_preserves_same_id_in_foreign_workspace(memory_store) -> None:
    """捕获 v2 写入无条件回收异域同名 UUID 点的越权缺陷。

    归属核验无法解码时按"宁可不清理"处理：异域旧点不被删除，保留等待
    迁移工具处理，而不是被当前 Workspace 的写入顺手回收。
    """
    store, qdrant = memory_store
    current = _identity_scope("isolation_workspace", user_id="u1")
    foreign_user_id = "u2"
    memory_id = uuid4()
    legacy_payload = {
        "id": str(memory_id),
        "meta": {
            "source_agent_id": "foreign-agent",
            "user_id": foreign_user_id,
            "visibility": "PUBLIC",
        },
        "index": {
            "title": "Foreign legacy memory",
            "summary": "A colliding legacy UUID remains owned by another user.",
            "tags": [],
            "memory_type": "FACT",
        },
        "payload": {"content": "foreign legacy content"},
        "relations": {},
    }
    await qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=str(memory_id),
                vector={"dense_text": [0.25, 0.75]},
                payload=legacy_payload,
            )
        ],
    )

    await store.upsert(
        _memory(
            current,
            memory_id=memory_id,
            content="current workspace content",
            alias="fact_current",
        )
    )

    remaining = await qdrant.client.retrieve(
        collection_name=qdrant.collection_name,
        ids=[str(memory_id)],
        with_payload=True,
        with_vectors=False,
    )
    assert len(remaining) == 1
    assert remaining[0].payload["meta"]["user_id"] == foreign_user_id

    # 删除当前 Workspace 记录后，异域旧点同样不被顺手回收。
    assert await store.delete(current, memory_id) is True
    remaining_after_delete = await qdrant.client.retrieve(
        collection_name=qdrant.collection_name,
        ids=[str(memory_id)],
        with_payload=True,
        with_vectors=False,
    )
    assert len(remaining_after_delete) == 1
    assert remaining_after_delete[0].payload["meta"]["user_id"] == foreign_user_id
