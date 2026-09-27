"""Memory alias 同 Workspace 唯一性不变量的集成测试（A2 §8 D-4 第二层）。

真实协作边界：``MidTermMemoryStore`` + ``QdrantStorageAdapter`` +
``QdrantMemoryStore``（Qdrant ``:memory:``），归档/复活路径叠加真实
``MemoryLibrary`` 与文件冷存储，外部编辑路径叠加真实
``MemoryGenerationFamiliar`` 与 ``ArtifactEngine``（临时目录）。只替换
进程外的 embedding provider 与未参与的生成引擎。

保护的行为：
- 同 Workspace 重名写入被拒绝且不产生写入；PRIVATE 记忆同样占用 alias；
- alias 归属跟随当前 canonical 记录：保留原 alias 的更新可提交，改名后旧
  alias 被释放；
- revive 时 alias 已被新记忆占用则显式失败，归档记录保持原状；
- 存量重名（绕过受控写入）使精确 alias 查询与 alias 批量读取 fail closed；
- 外部创建/编辑撞名在写版本 Artifact 之前失败，不留下孤立历史记录。
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
    WorkspaceMemoryKey,
    build_internal_identity_scope,
)
from hivememory.engines.artifacts import ArtifactEngine
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import (
    ArtifactStore,
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.memory import make_memory_metadata

FIXED_NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)


class _DeterministicEmbedding:
    """只替换测试边界外的 embedding provider。"""

    def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


@pytest_asyncio.fixture
async def qdrant_mid_term():
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(location=":memory:")
    qdrant.collection_name = "memory_alias_uniqueness"
    qdrant.vector_dimension = 2
    qdrant.embedding_service = _DeterministicEmbedding()
    await qdrant.client.create_collection(
        collection_name=qdrant.collection_name,
        vectors_config={"dense_text": VectorParams(size=2, distance=Distance.COSINE)},
    )
    try:
        yield MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False)), qdrant
    finally:
        await qdrant.client.close()


def _scope(workspace_id: str = "main_workspace", *, agent_id: str = "agent-a"):
    return build_internal_identity_scope(
        ActorIdentity(user_id="u1", agent_id=agent_id, team_id="team-a"),
        workspace_id,
    )


def _memory(scope, *, alias: str, content: str = "content") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(
            user_id=scope.actor_identity.user_id,
            source_agent_id=scope.actor_identity.agent_id,
            team_id=scope.actor_identity.team_id,
            workspace_id=scope.workspace_identity.workspace_id,
        ),
        index=IndexLayer(
            title="Alias uniqueness memory",
            summary="Memory used to verify workspace-scoped alias uniqueness.",
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


@pytest.mark.asyncio
async def test_duplicate_alias_in_same_workspace_is_rejected_without_write(qdrant_mid_term):
    """第二条同名记忆写入被拒绝：既有记录不变，新记录不落库。"""
    store, _ = qdrant_mid_term
    scope = _scope()
    first = _memory(scope, alias="fact_shared", content="first")
    second = _memory(scope, alias="fact_shared", content="second")
    await store.upsert(first)

    with pytest.raises(MemoryAliasConflictError) as exc_info:
        await store.upsert(second)

    assert exc_info.value.details["conflicting_memory_id"] == str(first.id)
    assert await store.get(scope, second.id) is None
    resolved = await store.get_by_alias(scope, "fact_shared")
    assert resolved.id == first.id


@pytest.mark.asyncio
async def test_private_memory_still_occupies_alias(qdrant_mid_term):
    """对写入方不可见的 PRIVATE 记忆同样占用 alias。

    捕获占用查询叠加 actor 读取策略、只看见可见子集而放行重名的缺陷。
    """
    store, _ = qdrant_mid_term
    owner_scope = _scope(agent_id="agent-b")
    private = _memory(owner_scope, alias="fact_private")
    private.meta.access_policy = MemoryAccessPolicy(
        visibility=MemoryVisibility.PRIVATE,
        target_agent_id="agent-b",
    )
    await store.upsert(private)

    with pytest.raises(MemoryAliasConflictError):
        await store.upsert(_memory(_scope(agent_id="agent-a"), alias="fact_private"))


@pytest.mark.asyncio
async def test_alias_ownership_follows_current_canonical_record(qdrant_mid_term):
    """保留原 alias 的更新可提交；改名后旧 alias 被释放，可由其他记忆使用。"""
    store, _ = qdrant_mid_term
    scope = _scope()
    memory = _memory(scope, alias="fact_old", content="v1")
    await store.upsert(memory)

    memory.payload.content = "v2"
    await store.upsert(memory)
    memory.index.alias = "fact_new"
    await store.upsert(memory)
    successor = _memory(scope, alias="fact_old", content="successor")
    await store.upsert(successor)

    assert (await store.get_by_alias(scope, "fact_old")).id == successor.id
    renamed = await store.get_by_alias(scope, "fact_new")
    assert (renamed.id, renamed.payload.content) == (memory.id, "v2")


@pytest.mark.asyncio
async def test_revive_fails_when_alias_was_taken_during_archive(qdrant_mid_term, tmp_path):
    """归档释放 alias；复活时撞名显式失败，归档记录与新记忆保持原状。"""
    store, _ = qdrant_mid_term
    scope = _scope()
    library = MemoryLibrary(
        short_term=ShortTermMemoryStore(),
        mid_term=store,
        long_term=LongTermMemoryStore(
            FileBasedStorageAdapter(archive_dir=str(tmp_path / "archive"), compress=False)
        ),
    )
    archived = _memory(scope, alias="fact_revived", content="archived")
    await store.upsert(archived)
    key = WorkspaceMemoryKey.from_identity_scope(scope, archived.id)
    await library.archive(key)
    newcomer = _memory(scope, alias="fact_revived", content="newcomer")
    await store.upsert(newcomer)

    with pytest.raises(MemoryAliasConflictError):
        await library.revive(scope, archived.id)

    assert await library.long_term.is_archived(key) is True
    assert await store.get(scope, archived.id) is None
    assert (await store.get_by_alias(scope, "fact_revived")).id == newcomer.id


@pytest.mark.asyncio
async def test_exact_alias_lookup_fails_closed_on_existing_duplicates(qdrant_mid_term):
    """绕过受控写入的存量重名使精确查询 fail closed，不按存储顺序任取其一。"""
    store, qdrant = qdrant_mid_term
    scope = _scope()
    await qdrant.upsert_memory(_memory(scope, alias="fact_legacy"), use_sparse=False)
    await qdrant.upsert_memory(_memory(scope, alias="fact_legacy"), use_sparse=False)

    with pytest.raises(MemoryAliasConflictError) as exc_info:
        await store.get_by_alias(scope, "fact_legacy")

    assert exc_info.value.details["reason"] == "ambiguous_alias"


@pytest.mark.asyncio
async def test_alias_batch_read_propagates_ambiguity_instead_of_empty_result(qdrant_mid_term):
    """alias 批量读取遇到多义 alias 时传播结构化错误，不被兜底伪装为"未找到"。"""
    store, qdrant = qdrant_mid_term
    scope = _scope()
    await qdrant.upsert_memory(_memory(scope, alias="fact_legacy"), use_sparse=False)
    await qdrant.upsert_memory(_memory(scope, alias="fact_legacy"), use_sparse=False)
    familiar = RetrievalFamiliar(
        engine=object(),
        memory_library=type("Library", (), {"mid_term": store})(),
    )

    with pytest.raises(MemoryAliasConflictError):
        await familiar.retrieve_by_aliases(["fact_legacy"], scope)


class _UnusedGenerationEngine:
    """外部创建/编辑路径不调用生成引擎。"""

    async def process(self, *args, **kwargs):  # pragma: no cover - 不应被调用
        raise AssertionError("external create/update must not run generation")


def _familiar(store, artifact_store) -> MemoryGenerationFamiliar:
    return MemoryGenerationFamiliar(
        generation_engine=_UnusedGenerationEngine(),
        memory_library=type("Library", (), {"mid_term": store})(),
        artifact_engine=ArtifactEngine.from_store(store=artifact_store),
        now=lambda: FIXED_NOW,
    )


def _artifact_store(tmp_path) -> ArtifactStore:
    return ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))


@pytest.mark.asyncio
async def test_external_create_conflict_fails_before_version_artifact(qdrant_mid_term, tmp_path):
    """外部创建撞名在写版本 Artifact 前失败：不留下孤立历史记录。

    捕获只依赖 upsert 兜底、冲突前已写入版本记录的缺陷。
    """
    store, _ = qdrant_mid_term
    scope = _scope()
    artifact_store = _artifact_store(tmp_path)
    familiar = _familiar(store, artifact_store)
    await store.upsert(_memory(scope, alias="fact_manual"))
    duplicate = _memory(scope, alias="fact_manual")

    with pytest.raises(MemoryAliasConflictError):
        await familiar.create_external_memory(scope, duplicate)

    assert await artifact_store.list_by_memory(scope, str(duplicate.id)) == []
    assert await store.get(scope, duplicate.id) is None


@pytest.mark.asyncio
async def test_external_content_edit_on_duplicated_alias_fails_before_version_artifact(
    qdrant_mid_term, tmp_path
):
    """未改 alias 的内容编辑同样先校验：存量重名时在写版本记录前失败。

    捕获只在 alias 变化时预检、而 upsert 兜底在版本记录之后才拒绝的缺陷。
    """
    store, qdrant = qdrant_mid_term
    scope = _scope()
    edited = _memory(scope, alias="fact_legacy", content="v1")
    await qdrant.upsert_memory(edited, use_sparse=False)
    await qdrant.upsert_memory(_memory(scope, alias="fact_legacy"), use_sparse=False)
    artifact_store = _artifact_store(tmp_path)
    familiar = _familiar(store, artifact_store)

    with pytest.raises(MemoryAliasConflictError):
        await familiar.update_external_memory(edited.id, identity_scope=scope, content="v2")

    assert await artifact_store.list_by_memory(scope, str(edited.id)) == []
    assert (await store.get(scope, edited.id)).payload.content == "v1"
