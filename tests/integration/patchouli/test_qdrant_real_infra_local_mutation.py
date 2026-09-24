"""真实 Qdrant 服务上的受控局部更新与向量保留证据（A2-P MVL-4 待验收项）。

需要独立 Qdrant 服务（开发环境经 ``scripts/hivememory-dev.sh start`` 拉起，
默认 ``http://127.0.0.1:6333``；可用 ``HIVEMEMORY_TEST_QDRANT_URL`` 覆盖）。
服务不可达时整文件跳过，不影响 CI。

锁定契约（A2-P §4.1/§9）：
- ``patch_payload``：Qdrant 局部 payload 更新只改 ``meta.lifecycle`` /
  ``meta.access_policy``，向量与其余字段原样保留，不改内容版本；
- ``upsert(recompute_vectors=False)``：embedding 不重算，向量逐位一致，
  payload 整体替换（含 ``payload.agent_config``）；
- 旧整数 schema ``2`` 记录在 mutation 入口被拒绝（兼容窗口只读）。
"""

import os
import socket
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceMemoryKey,
)
from hivememory.engines.retrieval.memory_codec import MemorySchemaReadOnlyError
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from tests.helpers.memory import make_memory_metadata

QDRANT_URL = os.environ.get("HIVEMEMORY_TEST_QDRANT_URL", "http://127.0.0.1:6333")
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def _service_reachable(url: str) -> bool:
    """快速探测 Qdrant HTTP 端口；不可达时跳过真实基础设施用例。"""
    host = url.split("//")[-1].split(":")[0]
    port = int(url.rsplit(":", 1)[-1].rstrip("/"))
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.real_infra,
    pytest.mark.skipif(
        not _service_reachable(QDRANT_URL),
        reason=f"独立 Qdrant 服务不可达: {QDRANT_URL}",
    ),
]


class _DeterministicEmbedding:
    """测试边界外的确定性 embedding provider（2 维，便于断言逐位一致）。"""

    def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


@pytest_asyncio.fixture
async def store():
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    qdrant.collection_name = "a2p_real_infra_local_mutation"
    qdrant.vector_dimension = 2
    qdrant.embedding_service = _DeterministicEmbedding()
    await qdrant.client.create_collection(
        collection_name=qdrant.collection_name,
        vectors_config={
            "dense_text": VectorParams(size=2, distance=Distance.COSINE),
        },
    )
    mid_term = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))
    try:
        yield mid_term, qdrant
    finally:
        await qdrant.client.delete_collection(collection_name=qdrant.collection_name)
        await qdrant.client.close()


def _memory(content: str) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="Real infra memory",
            summary="Verifies local payload mutation on a standalone Qdrant.",
            tags=["t1"],
            memory_type=MemoryType.FACT,
            alias="real_infra_memory",
        ),
        payload=PayloadLayer(content=content),
    )


def _identity_scope():
    from tests.helpers.memory import make_memory_identity_scope

    return make_memory_identity_scope(user_id="u1", agent_id="a1")


async def _dense_vectors(qdrant: QdrantMemoryStore, memory: MemoryAtom):
    key = WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)
    points = await qdrant.client.retrieve(
        collection_name=qdrant.collection_name,
        ids=[qdrant._point_id(key)],
        with_payload=True,
        with_vectors=True,
    )
    assert points, "point 未写入"
    return points[0].vector["dense_text"], points[0].payload


@pytest.mark.asyncio
async def test_patch_payload_preserves_vectors_and_non_target_fields(store):
    """patch_payload：向量逐位一致、其余字段不动、仅 lifecycle 白名单字段更新。"""
    mid_term, qdrant = store
    identity_scope = _identity_scope()
    atom = _memory("content before patch")

    await mid_term.upsert(atom)
    vectors_before, payload_before = await _dense_vectors(qdrant, atom)
    assert payload_before["schema_version"] == "2.1"
    updated_at_before = payload_before["meta"]["updated_at"]

    result = await mid_term.patch_payload(
        WorkspaceMemoryKey(workspace_identity=identity_scope.workspace_identity, memory_id=atom.id),
        {
            "meta.lifecycle.access_count": 7,
            "meta.lifecycle.last_accessed_at": FIXED_NOW,
        },
    )

    assert result is not None
    assert result.meta.lifecycle.access_count == 7
    assert result.meta.lifecycle.last_accessed_at == FIXED_NOW

    vectors_after, payload_after = await _dense_vectors(qdrant, atom)
    # 向量逐位一致：局部更新未触碰向量。
    assert vectors_after == vectors_before
    # 白名单字段已更新。
    assert payload_after["meta"]["lifecycle"]["access_count"] == 7
    # 非目标字段（内容时间、内容、版本、策略）原样保留。
    assert payload_after["meta"]["updated_at"] == updated_at_before
    assert payload_after["meta"]["version"] == payload_before["meta"]["version"]
    assert payload_after["payload"]["content"] == "content before patch"
    assert payload_after["meta"]["access_policy"] == payload_before["meta"]["access_policy"]


@pytest.mark.asyncio
async def test_upsert_without_recompute_preserves_vectors_on_real_qdrant(store):
    """recompute_vectors=False：embedding 不重算，向量逐位一致且 payload 更新。"""
    mid_term, qdrant = store
    atom = _memory("agent config carrier")
    atom.payload.agent_config = {"model_name": "v1"}

    await mid_term.upsert(atom)
    vectors_before, _ = await _dense_vectors(qdrant, atom)

    updated = atom.model_copy(deep=True)
    updated.payload.agent_config = {"model_name": "v2"}
    await mid_term.upsert(updated, recompute_vectors=False)

    vectors_after, payload_after = await _dense_vectors(qdrant, atom)
    assert vectors_after == vectors_before
    assert payload_after["payload"]["agent_config"] == {"model_name": "v2"}
    # payload 其余键同步替换（同一提交的完整 payload）。
    assert payload_after["payload"]["content"] == "agent config carrier"


@pytest.mark.asyncio
async def test_legacy_schema_memory_rejected_at_mutation_entry_on_real_qdrant(store):
    """旧整数 schema 记录在 mutation 入口被拒（兼容窗口只读，fail closed）。"""
    mid_term, qdrant = store
    atom = _memory("legacy carrier")
    await mid_term.upsert(atom)

    # 直接写入一条旧整数 schema 2 的原始 payload（平铺字段布局）。
    legacy_id = uuid4()
    legacy_payload = {
        "schema_version": 2,
        "id": str(legacy_id),
        "meta": {
            "source_agent_id": "legacy-agent",
            "workspace_identity": {
                "owner_user_id": "u1",
                "workspace_key": "main_workspace",
                "workspace_id": "main_workspace",
            },
            "owner_user_id": "u1",
            "workspace_key": "main_workspace",
            "workspace_id": "main_workspace",
            "access_policy": {"visibility": "PUBLIC"},
            "created_at": "2026-09-06T00:00:00+00:00",
            "updated_at": "2026-09-06T00:00:00+00:00",
            "version": 1,
        },
        "index": {
            "title": "Legacy",
            "summary": "Legacy v2 record for real infra gate verification.",
            "memory_type": "FACT",
            "tags": [],
        },
        "payload": {"content": "legacy"},
        "relations": {},
    }
    key = WorkspaceMemoryKey(workspace_identity=atom.workspace_identity, memory_id=legacy_id)
    from qdrant_client.models import PointStruct

    await qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=qdrant._point_id(key), vector={"dense_text": [0.1, 0.9]}, payload=legacy_payload
            ),
        ],
    )

    identity_scope = _identity_scope()
    with pytest.raises(MemorySchemaReadOnlyError):
        await mid_term.get_for_mutation(identity_scope, legacy_id)
    with pytest.raises(MemorySchemaReadOnlyError):
        await mid_term.patch_payload(
            key,
            {"meta.lifecycle.access_count": 1},
        )
