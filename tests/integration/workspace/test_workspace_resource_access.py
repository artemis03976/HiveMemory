"""Workspace 资源服务的集成测试：admission + 真实 Qdrant 存储 + 资源读取。

被测协作边界（全部真实）：LocalTrustedAdmissionService → WorkspaceMemoryService /
ProfileResourceService → MidTermMemoryStore → QdrantStorageAdapter → 内存模式
Qdrant。只有 embedding provider 用确定性 fake（进程外部依赖）。

保护父计划 5.6/5.7 与 WRX-1 验收：完整入口链先 admission 后资源 policy；
两个 Workspace 的同 alias/同 profile 坐标互不串扰；跨 Workspace 引用
fail closed；PRIVATE 可见性在 owner 硬边界内继续生效。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.core.errors import (
    AdmissionDeniedError,
    ResourceNotFoundError,
)
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
)
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    ProfileResourceService,
    WorkspaceMemoryService,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
ISOLATED = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class _DeterministicEmbedding:
    """只替换进程外部的 embedding provider（确定性二维向量）。"""

    def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


@pytest_asyncio.fixture
async def mid_term_store():
    qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
    qdrant.client = AsyncQdrantClient(location=":memory:")
    qdrant.collection_name = "workspace_resource_access"
    qdrant.vector_dimension = 2
    qdrant.embedding_service = _DeterministicEmbedding()
    await qdrant.client.create_collection(
        collection_name=qdrant.collection_name,
        vectors_config={"dense_text": VectorParams(size=2, distance=Distance.COSINE)},
    )
    store = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))
    try:
        yield store
    finally:
        await qdrant.client.close()


async def _seed(store: MidTermMemoryStore, workspace, *, alias, content, agent_id="a1"):
    from hivememory.core.models import MemoryAccessPolicy, MemoryAtom, MetaData

    atom = MemoryAtom(
        meta=MetaData(
            workspace_identity=workspace,
            source_agent_id=agent_id,
            access_policy=MemoryAccessPolicy.public(),
        ),
        index=IndexLayer(
            title=f"title-{alias}",
            summary=f"summary-{alias}",
            tags=[],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )
    await store.upsert(atom)
    return atom


def _admission() -> LocalTrustedAdmissionService:
    return LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )


async def _context(
    admission: LocalTrustedAdmissionService,
    workspace,
    operation,
    *,
    agent_id="a1",
):
    return await admission.admit(
        CallerPrincipal("local-process:test"),
        ActorIdentity(user_id="u1", agent_id=agent_id),
        workspace,
        operation,
    )


@pytest.mark.asyncio
async def test_same_alias_isolated_between_two_workspaces(mid_term_store):
    """两个 Workspace 的同名 alias 各自命中各自内容，互不串扰。"""
    await _seed(mid_term_store, MAIN, alias="fact_shared", content="main content")
    await _seed(mid_term_store, ISOLATED, alias="fact_shared", content="isolated content")
    service = WorkspaceMemoryService(mid_term_store, retrieval=None)
    admission = _admission()

    main_snapshot = await service.read_memory_by_alias(
        await _context(admission, MAIN, WorkspaceOperation.RESOURCE_READ),
        "fact_shared",
    )
    isolated_snapshot = await service.read_memory_by_alias(
        await _context(admission, ISOLATED, WorkspaceOperation.RESOURCE_READ),
        "fact_shared",
    )

    assert main_snapshot.content == "main content"
    assert isolated_snapshot.content == "isolated content"
    assert main_snapshot.workspace_identity == MAIN
    assert isolated_snapshot.workspace_identity == ISOLATED


@pytest.mark.asyncio
async def test_cross_workspace_uuid_read_fails_closed(mid_term_store):
    """另一 Workspace 的 canonical uuid 在本 Workspace 不可见（not found）。"""
    seeded = await _seed(mid_term_store, MAIN, alias="fact_main_only", content="secret")
    service = WorkspaceMemoryService(mid_term_store, retrieval=None)
    admission = _admission()

    with pytest.raises(ResourceNotFoundError):
        await service.read_memory(
            await _context(admission, ISOLATED, WorkspaceOperation.RESOURCE_READ),
            str(seeded.id),
        )


@pytest.mark.asyncio
async def test_unknown_principal_is_denied_before_touching_resources(mid_term_store):
    """未注册 principal 在 admission 层被拒绝，未到达资源存储。"""
    admission = _admission()

    with pytest.raises(AdmissionDeniedError):
        await admission.admit(
            CallerPrincipal("local-process:stranger"),
            ActorIdentity(user_id="u1", agent_id="a1"),
            MAIN,
            WorkspaceOperation.RESOURCE_READ,
        )


@pytest.mark.asyncio
async def test_canonical_update_is_visible_on_next_read(mid_term_store):
    """canonical 更新后下一次读取立即返回新内容与新 revision。"""
    seeded = await _seed(mid_term_store, MAIN, alias="fact_evolving", content="v1")
    service = WorkspaceMemoryService(mid_term_store, retrieval=None)
    admission = _admission()
    context = await _context(admission, MAIN, WorkspaceOperation.RESOURCE_READ)

    before = await service.read_memory_by_alias(context, "fact_evolving")
    updated = seeded.model_copy(deep=True)
    updated.payload = PayloadLayer(content="v2")
    updated.meta = updated.meta.model_copy(update={"version": seeded.meta.version + 1})
    await mid_term_store.upsert(updated)

    after = await service.read_memory_by_alias(context, "fact_evolving")

    assert before.content == "v1"
    assert after.content == "v2"
    assert after.source_revision == seeded.meta.version + 1


@pytest.mark.asyncio
async def test_profile_and_memory_services_share_same_admission_boundary(mid_term_store):
    """profile 与 memory 服务消费同一 admission 签发的上下文，语义一致。"""
    profile_atom = MemoryAtom(
        meta=MetaData(
            workspace_identity=MAIN,
            source_agent_id="a1",
            access_policy=MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="a1",
            ),
        ),
        index=IndexLayer(
            title="profile",
            summary="profile summary",
            tags=[],
            memory_type=MemoryType.AGENT_PROFILE,
            alias="agent_config",
        ),
        payload=PayloadLayer(
            content="persona",
            artifacts={"agent_config": {"model_name": "gpt-test"}},
        ),
    )
    await mid_term_store.upsert(profile_atom)

    profile_service = ProfileResourceService(mid_term_store)
    admission = _admission()

    owner_ctx = await _context(admission, MAIN, WorkspaceOperation.PROFILE_READ, agent_id="a1")
    profile = await profile_service.read_profile(owner_ctx, "agent_config")
    assert profile.source_kind == "atom"

    # alias 路径的存储预过滤合并不可见与缺失：对其他 agent 统一 not found
    other_ctx = await _context(admission, MAIN, WorkspaceOperation.PROFILE_READ, agent_id="a2")
    with pytest.raises(ResourceNotFoundError):
        await profile_service.read_profile(other_ctx, "agent_config")
