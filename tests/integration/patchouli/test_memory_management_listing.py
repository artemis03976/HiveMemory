"""管理列表经真实 local bus、RetrievalFamiliar 与 Qdrant 内存模式的翻页与类型排除。"""

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.core.models import IdentityScope, IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.memory_library.library import MemoryLibrary
from hivememory.memory_library.stores import MidTermMemoryStore
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


class _DeterministicEmbedding:
    """只替换测试边界外的 embedding provider。"""

    async def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


@pytest_asyncio.fixture
async def listing():
    """真实的管理服务 → local bus → RetrievalFamiliar → 中期库门面 → Qdrant 内存模式。"""
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(location=":memory:")
    qdrant.collection_name = "memory_management_listing"
    qdrant.embedding_service = _DeterministicEmbedding()
    await qdrant.client.create_collection(
        collection_name=qdrant.collection_name,
        vectors_config={"dense_text": VectorParams(size=2, distance=Distance.COSINE)},
    )
    store = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))
    local_bus = PatchouliBus()
    familiar = RetrievalFamiliar(
        engine=object(),  # 列表不经检索引擎
        memory_library=MemoryLibrary(short_term=object(), mid_term=store, long_term=object()),
    )
    local_bus.register(PatchouliLocalRoutes.MEMORY_LIST, familiar.list_memories)
    try:
        yield store, MemoryManagementService(bus=local_bus)
    finally:
        await qdrant.client.close()


async def _seed(
    store: MidTermMemoryStore,
    scope: IdentityScope,
    memory_type: MemoryType = MemoryType.FACT,
) -> MemoryAtom:
    atom = MemoryAtom(
        meta=make_memory_metadata(
            user_id=scope.actor_identity.user_id,
            source_agent_id="a1",
            workspace_id=scope.workspace_identity.workspace_id,
        ),
        index=IndexLayer(title="Listed memory", summary="s", memory_type=memory_type),
        payload=PayloadLayer(content="c"),
    )
    await store.upsert(atom)
    return atom


@pytest.mark.asyncio
async def test_pages_reach_every_memory_of_the_workspace_once(listing) -> None:
    """跟随 next_cursor 翻页取得本 Workspace 的全部记忆，每条一次，总数是库内总量。

    捕获只返回第一页、游标没有传到存储（反复取回第一页）或计数越过 Workspace 边界的缺陷。
    """
    store, service = listing
    scope = make_identity_scope(workspace_id="main_workspace")
    seeded = {(await _seed(store, scope)).id for _ in range(5)}
    await _seed(store, make_identity_scope(workspace_id="other_workspace"))

    collected, totals, cursor = [], set(), None
    for _ in range(4):  # 5 条、每页 2 条应在 3 页内取完；多出一页的余量让缺陷以断言失败而不是挂起
        page = await service.list_memories(
            identity_scope=scope, limit=2, cursor=cursor, refresh_vitality=False
        )
        collected.extend(atom.id for atom in page.memories)
        totals.add(page.total)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert cursor is None
    assert sorted(collected) == sorted(seeded)
    assert totals == {5}


@pytest.mark.asyncio
async def test_excluded_types_are_neither_listed_nor_counted(listing) -> None:
    """排除的类型在存储侧过滤：列表不含它们，总数也不计入。

    捕获排除改回取回后过滤（页内条数少于请求、总数仍计入 Profile）或排除条件没有下传的缺陷。
    """
    store, service = listing
    scope = make_identity_scope(workspace_id="main_workspace")
    facts = {(await _seed(store, scope)).id for _ in range(3)}
    for _ in range(2):
        await _seed(store, scope, MemoryType.AGENT_PROFILE)

    page = await service.list_memories(
        identity_scope=scope,
        limit=10,
        exclude_types=[MemoryType.AGENT_PROFILE.value],
        refresh_vitality=False,
    )

    assert {atom.id for atom in page.memories} == facts
    assert page.total == 3
