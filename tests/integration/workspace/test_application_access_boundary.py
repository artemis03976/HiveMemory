"""统一访问边界与公开 application 链路的集成测试（WRX-1 核心验收）。

真实协作边界（父计划 11.2/5.7.2）：GlobalSystemBus + PatchouliBridge +
PatchouliPublicApi（真实 application 服务）+ Patchouli local bus（真实
familiar/controller/coordinator handler）+ 真实 Qdrant ``:memory:`` 存储 +
真实 InteractionSubmissionQueue。只替换进程外依赖：embedding 用确定性
二维向量、生成执行（LLM）与 Topic 会话读取用确定性实现。不创建
Alice/PendingAtomRuntime/MTP（headless）。

保护父计划 5.6/5.7 与 WRX-1 验收：
- admission → operation → ownership/visibility → domain policy 顺序；
- 裸 scope/错误 grant 被拒绝；管理操作不泄漏给 Agent；
- 交互提交不隐含检索权；无 Alice 的真实提交/结果可用；
- Profile source 语义与归属任务投影经公开路由可达。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ResourceNotFoundError,
    ScopeRequiredError,
)
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    LogicalBlock,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    TurnRecord,
)
from hivememory.core.protocol.models import InteractionPayload, RetrievalRequest
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    InteractionSubmissionService,
    MemoryIntent,
    MemoryIntentSubmissionService,
    MemoryManagementService,
    MemoryTaskManagementService,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.control.memory_generation.controller import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.control.memory_generation.coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationResult
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class _DeterministicEmbedding:
    """进程外 embedding 依赖的确定性替换。"""

    def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


class _FakeTopicData:
    topic_title = "boundary topic"
    topic_summary = "boundary topic summary"
    state_summary = "boundary state summary"

    def recent_blocks(self, limit):
        return [LogicalBlock(turn=TurnRecord(user_query="question", assistant_final_text="answer"))]


class _FakeRetrievalEngine:
    """检索引擎边界的确定性替换：返回预置 atoms。"""

    def __init__(self, atoms):
        self._atoms = atoms

    async def retrieve(self, query, top_k):
        class _Result:
            memories = list(self._atoms)
            memories_count = len(self._atoms)
            latency_ms = 0.1

        return _Result()


@pytest_asyncio.fixture
async def wired():
    """装配真实 bridge/application/local-route 协作链。"""
    qdrant = QdrantMemoryStoreStub()
    await qdrant.setup()

    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()

    familiar = RetrievalFamiliar(
        engine=_FakeRetrievalEngine([]),
        memory_library=MemoryLibrary(
            short_term=object(),  # 读取用例不触达短期库
            mid_term=qdrant.store,
            long_term=object(),
        ),
        local_bus=local_bus,
    )

    controller = MemoryGenerationTaskController(bus=local_bus)
    coordinator = MemoryGenerationCoordinator(bus=local_bus)
    await controller.start()
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_GET,
        familiar.get_memory,
    )
    local_bus.register(
        PatchouliLocalRoutes.GET_AGENT_PROFILE_SNAPSHOT,
        familiar.get_agent_profile_snapshot,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
        controller.submit_generation,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_GET,
        controller.get_task,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_WAIT,
        controller.wait_task,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
        controller.submit_generation_many,
    )
    local_bus.register(
        PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
        coordinator.submit_active,
    )
    local_bus.register(
        PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
        AsyncMock(side_effect=lambda spec: _settlement_results(spec.pending_alias)),
    )
    local_bus.register(
        PatchouliLocalRoutes.TOPIC_GET,
        AsyncMock(return_value=_FakeTopicData()),
    )

    async def _apply(
        payload, *, identity_scope, target_topic_id, interaction_id=None, asset_refs=()
    ):
        return target_topic_id

    queue = InteractionSubmissionQueue(_apply)

    memory_service = MemoryManagementService(bus=local_bus)
    profile_service = AgentProfileManagementService(bus=local_bus)
    task_service = MemoryTaskManagementService(bus=local_bus)
    interactions = InteractionSubmissionService(interaction_queue=queue)
    memory_intents = MemoryIntentSubmissionService(bus=local_bus)

    # 本验收不触达 run 编排/Topic/模型就绪用例，入口以 MagicMock 占位
    public_api = PatchouliPublicApi(
        chat=MagicMock(),
        memory=memory_service,
        memory_tasks=task_service,
        agent_profiles=profile_service,
        interactions=interactions,
        memory_intents=memory_intents,
        topics=MagicMock(),
        readiness=MagicMock(),
    )
    bridge = PatchouliBridge(
        local_bus=local_bus,
        global_bus=global_bus,
        public_api=public_api,
    )
    bridge.mount()

    admission = LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )
    try:
        yield _Wired(
            global_bus=global_bus,
            store=qdrant.store,
            controller=controller,
            admission=admission,
        )
    finally:
        await controller.stop()
        await qdrant.close()


class _Wired:
    """测试组合的已装配组件集合。"""

    def __init__(self, *, global_bus, store, controller, admission):
        self.global_bus = global_bus
        self.store = store
        self.controller = controller
        self.admission = admission

    async def context(self, operation, workspace=MAIN):
        actor = make_identity_scope(
            user_id="u1", agent_id="a1", workspace_id=workspace.workspace_id
        ).actor_identity
        return await self.admission.admit(
            CallerPrincipal("local-process:test"),
            actor,
            workspace,
            operation,
        )


class QdrantMemoryStoreStub:
    """真实 Qdrant ``:memory:`` 存储的测试装配。"""

    def __init__(self):
        self.client = None
        self.store = None

    async def setup(self):
        from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore

        qdrant = QdrantMemoryStore.__new__(QdrantMemoryStore)
        qdrant.client = AsyncQdrantClient(location=":memory:")
        qdrant.collection_name = "workspace_access_boundary"
        qdrant.vector_dimension = 2
        qdrant.embedding_service = _DeterministicEmbedding()
        await qdrant.client.create_collection(
            collection_name=qdrant.collection_name,
            vectors_config={"dense_text": VectorParams(size=2, distance=Distance.COSINE)},
        )
        self.client = qdrant.client
        self.store = MidTermMemoryStore(QdrantStorageAdapter(qdrant, use_sparse=False))

    async def close(self):
        await self.client.close()


def _settlement_results(pending_alias: str):
    settlement = PendingAtomSettlement(
        pending_alias=pending_alias,
        intent_id=f"intent_{pending_alias}",
        resolution=PendingAtomResolution.CREATED,
        canonical_alias="memory_alias",
        canonical_uuid=str(uuid4()),
    )
    return [
        MemoryGenerationResult(
            canonical_alias="memory_alias",
            settlement=settlement,
        )
    ]


async def _seed_public_fact(store, *, alias, content, workspace=MAIN, agent_id="a1"):
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


# ---------------------------------------------------------------------------
# 验收用例
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_principal_denied_before_application(wired):
    """未注册 principal 在 admission 被拒绝，请求不进入 application。"""
    with pytest.raises(AdmissionDeniedError):
        actor = ActorIdentity(user_id="u1", agent_id="a1")
        await wired.admission.admit(
            CallerPrincipal("local-process:stranger"),
            actor,
            MAIN,
            WorkspaceOperation.RESOURCE_READ,
        )


@pytest.mark.asyncio
async def test_agent_read_grant_cannot_reach_management_get(wired):
    """resource.read grant 调用管理 GET 路由：application 入口 OperationDenied。"""
    await _seed_public_fact(wired.store, alias="fact_mgmt", content="managed")
    context = await wired.context(WorkspaceOperation.RESOURCE_READ)

    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            str(uuid4()),
            identity_scope=context.identity_scope,
            access=context,
        )


@pytest.mark.asyncio
async def test_management_grant_reads_via_owner_semantics(wired):
    """management.memory grant 走管理 GET：owner-management 语义读取成功。"""
    atom = await _seed_public_fact(wired.store, alias="fact_owned", content="managed content")
    context = await wired.context(WorkspaceOperation.MANAGEMENT_MEMORY)

    result = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_GET,
        str(atom.id),
        identity_scope=context.identity_scope,
        access=context,
    )

    assert result is not None
    assert result.payload.content == "managed content"


@pytest.mark.asyncio
async def test_bare_scope_without_access_rejected_on_read_route(wired):
    """新公开读路由不接受裸 scope（无迁移路径）。"""
    with pytest.raises(ScopeRequiredError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_READ,
            str(uuid4()),
            identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        )


@pytest.mark.asyncio
async def test_actor_visible_read_enforces_visibility_and_workspace(wired):
    """Actor-visible 点读：可见 atom 读取；PRIVATE 他者与跨 Workspace 不可见。"""
    public_atom = await _seed_public_fact(wired.store, alias="fact_public", content="visible")
    private_atom = MemoryAtom(
        meta=MetaData(
            workspace_identity=MAIN,
            source_agent_id="a1",
            access_policy=MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="a1",
            ),
        ),
        index=IndexLayer(
            title="title-secret",
            summary="summary-secret",
            tags=[],
            memory_type=MemoryType.FACT,
            alias="fact_secret",
        ),
        payload=PayloadLayer(content="secret"),
    )
    await wired.store.upsert(private_atom)
    foreign = await _seed_public_fact(
        wired.store, alias="fact_foreign_ws", content="other ws", workspace=OTHER
    )

    read_context = await wired.context(WorkspaceOperation.RESOURCE_READ)

    from hivememory.core.models import ActorIdentity as _Actor

    other_agent_access = await wired.admission.admit(
        CallerPrincipal("local-process:test"),
        _Actor(user_id="u1", agent_id="a2"),
        MAIN,
        WorkspaceOperation.RESOURCE_READ,
    )

    visible = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(public_atom.id),
        access=read_context,
    )
    assert visible is not None
    assert visible.payload.content == "visible"

    # PRIVATE 只对目标 Agent 可见：其他 Agent 视为不存在
    secret = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(private_atom.id),
        access=other_agent_access,
    )
    assert secret is None

    # 跨 Workspace 的 canonical uuid 不可见
    foreign = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(foreign.id),
        access=other_agent_access,
    )
    assert foreign is None


@pytest.mark.asyncio
async def test_profile_snapshot_route_with_profile_read_grant(wired):
    """profile.read 走 snapshot 路由取得 source 归属；管理 grant 被拒绝。"""
    profile_atom = MemoryAtom(
        meta=MetaData(
            workspace_identity=MAIN,
            source_agent_id="a1",
            access_policy=MemoryAccessPolicy.public(),
        ),
        index=IndexLayer(
            title="profile",
            summary="profile summary text",
            tags=[],
            memory_type=MemoryType.AGENT_PROFILE,
            alias="agent_config",
        ),
        payload=PayloadLayer(
            content="persona",
            artifacts={"agent_config": {"model_name": "gpt-test"}},
        ),
    )
    await wired.store.upsert(profile_atom)
    profile_context = await wired.context(WorkspaceOperation.PROFILE_READ)

    snapshot = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE_SNAPSHOT,
        "agent_config",
        identity_scope=profile_context.identity_scope,
        access=profile_context,
    )
    assert snapshot.source_kind == "atom"
    assert snapshot.source_atom_uuid == str(profile_atom.id)

    management_context = await wired.context(WorkspaceOperation.MANAGEMENT_MEMORY)
    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE_SNAPSHOT,
            "agent_config",
            identity_scope=management_context.identity_scope,
            access=management_context,
        )


@pytest.mark.asyncio
async def test_interaction_submit_does_not_imply_search(wired):
    """交互提交用例接纳收据；同一 grant 调用检索被拒绝（能力不互相推导）。"""
    submit_context = await wired.context(WorkspaceOperation.INTERACTION_SUBMIT)
    payload = InteractionPayload(
        user_message="boundary question",
        mtp_traces=[],
        assistant_final_text="boundary answer",
        turn_events=[],
    )
    receipt = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_INTERACTION_SUBMIT,
        payload=payload,
        access=submit_context,
        requested_topic_id="topic_boundary",
        interaction_id="interaction_boundary_1",
    )
    assert receipt.interaction_id == "interaction_boundary_1"

    retrieval_request = RetrievalRequest(
        semantic_query="boundary",
        identity_scope=submit_context.identity_scope,
    )
    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            request=retrieval_request,
            access=submit_context,
        )


@pytest.mark.asyncio
async def test_memory_intent_submit_and_observe_result_without_alice(wired):
    """意图提交 → 真实生成链 → task.observe 观察真实结果；跨 scope 拒绝。"""
    intent_context = await wired.context(WorkspaceOperation.MEMORY_INTENT_SUBMIT)
    submission = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_INTENT_SUBMIT,
        intent=MemoryIntent(
            kind="write",
            topic_id="topic_boundary",
            content="remember the boundary fact",
            title="Boundary Note",
        ),
        access=intent_context,
    )
    assert submission.accepted is True
    assert submission.task_id is not None

    final = await wired.controller.wait_task(submission.task_id)
    assert final.status.value == "completed"

    observe_context = await wired.context(WorkspaceOperation.TASK_OBSERVE)
    result = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
        submission.task_id,
        access=observe_context,
    )
    assert result.status.value == "completed"
    assert result.canonical_alias == "memory_alias"
    assert result.identity_scope == intent_context.identity_scope
    assert result.submitted_by == "local-process:test"

    other_context = await wired.context(WorkspaceOperation.TASK_OBSERVE, workspace=OTHER)
    with pytest.raises(ResourceNotFoundError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
            submission.task_id,
            access=other_context,
        )
