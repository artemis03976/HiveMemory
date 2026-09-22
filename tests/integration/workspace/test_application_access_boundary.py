"""统一认证网关、共享行为检查与公开 application 链路的集成测试（A1 验收）。

真实协作边界：GlobalSystemBus + PatchouliBridge + PatchouliPublicApi（真实
application 服务 + 共享行为检查）+ 统一认证网关（真实两类注册表）+
Patchouli local bus（真实 familiar/controller/coordinator handler）+ 真实
Qdrant ``:memory:`` 存储 + 真实 InteractionSubmissionQueue。只替换进程外
依赖：embedding 用确定性二维向量、生成执行（LLM）与 Topic 会话读取用
确定性实现。不创建 Alice/PendingAtomRuntime/MTP（headless）。

保护 A1 计划第 5 节验收证据：
- 两项认证在统一网关一次完成；三类调用侧共用同一业务路由与行为检查；
- 同一 principal 多 Actor、同一 Actor 多 Workspace 的许可互不串扰；
- 空白名单可进入但资源动作全拒绝；operation 互不隐含；
- 行为许可与资源 owner 规则分别生效；方法授权先于资源访问与副作用；
- context 与单次 operation 解耦：同一 context 先后执行不同获准操作；
- 到期 context 拒绝、重新认证恢复；授权拒绝不产生副作用、不包装成
  服务不可用。
"""

from __future__ import annotations

import json
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
    TopicManagementService,
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
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.access import CallerPrincipal
from hivememory.workspace import WorkspaceOperation
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")

READ = WorkspaceOperation.RESOURCE_READ
SEARCH = WorkspaceOperation.RESOURCE_SEARCH
INTENT = WorkspaceOperation.MEMORY_INTENT_SUBMIT
INTERACT = WorkspaceOperation.INTERACTION_SUBMIT
OBSERVE = WorkspaceOperation.TASK_OBSERVE
MANAGE_MEMORY = WorkspaceOperation.MANAGEMENT_MEMORY
MANAGE_TASK = WorkspaceOperation.MANAGEMENT_TASK


def _access_records() -> list:
    """两类注册表共用的访问矩阵：同 owner 多 Actor、同 Actor 多 Workspace。"""
    return [
        # MAIN：a1 全量基线操作；a2 仅读取；a3 空白名单；a4 仅观察；a5 仅 Memory 管理
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=MAIN.workspace_id,
            agent_id="a1",
            allowed_operations={READ, SEARCH, INTERACT, INTENT, OBSERVE, MANAGE_TASK},
        ),
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=MAIN.workspace_id,
            agent_id="a2",
            allowed_operations={READ},
        ),
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=MAIN.workspace_id,
            agent_id="a3",
            allowed_operations=frozenset(),
        ),
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=MAIN.workspace_id,
            agent_id="a4",
            allowed_operations={OBSERVE},
        ),
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=MAIN.workspace_id,
            agent_id="a5",
            allowed_operations={MANAGE_MEMORY},
        ),
        # OTHER：a1 的白名单与 MAIN 不同；a2 无准入记录；a4 仅有观察
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=OTHER.workspace_id,
            agent_id="a1",
            allowed_operations={READ, INTENT},
        ),
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=OTHER.workspace_id,
            agent_id="a4",
            allowed_operations={OBSERVE},
        ),
    ]


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


class FakeClock:
    """可控单调时钟：仅 TTL 到期用例推进。"""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest_asyncio.fixture
async def wired():
    """装配真实 bridge/application/local-route/统一网关协作链。"""
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
        PatchouliLocalRoutes.MEMORY_RETRIEVE,
        familiar.retrieve_async,
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

    clock = FakeClock()
    access = make_access_composition(
        _access_records(),
        default_workspace=MAIN,
        context_ttl_seconds=60,
        clock=clock,
    )

    memory_service = MemoryManagementService(bus=local_bus, access_guard=access.guard)
    profile_service = AgentProfileManagementService(bus=local_bus, access_guard=access.guard)
    task_service = MemoryTaskManagementService(bus=local_bus, access_guard=access.guard)
    interactions = InteractionSubmissionService(
        interaction_queue=queue,
        access_guard=access.guard,
    )
    memory_intents = MemoryIntentSubmissionService(bus=local_bus, access_guard=access.guard)
    topics = TopicManagementService(bus=local_bus, access_guard=access.guard)

    # 本验收不触达 run 编排/模型就绪用例，入口以 MagicMock 占位
    public_api = PatchouliPublicApi(
        chat=MagicMock(),
        memory=memory_service,
        memory_tasks=task_service,
        agent_profiles=profile_service,
        interactions=interactions,
        memory_intents=memory_intents,
        topics=topics,
        readiness=MagicMock(),
    )
    bridge = PatchouliBridge(
        local_bus=local_bus,
        global_bus=global_bus,
        public_api=public_api,
    )
    bridge.mount()

    # System 管理门面：与外部 adapter 同一全局总线，验证 context 传播
    system_memory = MemoryApplicationService(global_bus=global_bus, config=MagicMock())
    system_tasks = MemoryTaskApplicationService(global_bus=global_bus)

    try:
        yield _Wired(
            global_bus=global_bus,
            store=qdrant.store,
            controller=controller,
            queue=queue,
            access=access,
            clock=clock,
            system_memory=system_memory,
            system_tasks=system_tasks,
        )
    finally:
        await queue.stop()
        await controller.stop()
        await qdrant.close()


class _Wired:
    """测试组合的已装配组件集合。"""

    def __init__(
        self,
        *,
        global_bus,
        store,
        controller,
        queue,
        access: AccessTestComposition,
        clock: FakeClock,
        system_memory,
        system_tasks,
    ):
        self.global_bus = global_bus
        self.store = store
        self.controller = controller
        self.queue = queue
        self.access = access
        self.clock = clock
        self.system_memory = system_memory
        self.system_tasks = system_tasks


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
async def test_unregistered_source_denied_before_workspace_admission(wired):
    """未登记来源在系统接入层拒绝，不进入 Workspace 准入（证据 1）。"""
    with pytest.raises(AdmissionDeniedError) as exc_info:
        await wired.access.gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal("local-process:stranger"),
            actor=ActorIdentity(user_id="u1", agent_id="a1"),
            workspace=MAIN,
        )
    assert exc_info.value.details["reason"] == "unknown_principal"


@pytest.mark.asyncio
async def test_same_owner_actors_have_different_admission_across_workspaces(wired):
    """同一 owner：a2 在 MAIN 获准、在 OTHER 无准入记录；权限互不串扰（证据 1/2）。"""
    main_context = await wired.access.authenticate(agent_id="a2", workspace=MAIN)
    assert main_context.identity_scope.workspace_identity == MAIN

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await wired.access.authenticate(agent_id="a2", workspace=OTHER)
    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
async def test_empty_whitelist_admits_entry_but_denies_every_resource_action(wired):
    """空白名单 Actor 可进入，但资源动作全部拒绝（证据 3）。"""
    context = await wired.access.authenticate(agent_id="a3", workspace=MAIN)

    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_READ,
            str(uuid4()),
            access=context,
        )


@pytest.mark.asyncio
async def test_read_only_actor_cannot_submit_intent_or_reach_management(wired):
    """仅有 read 的 Actor 无法写；也不能借读取进入管理路由（证据 3）。"""
    read_context = await wired.access.authenticate(agent_id="a2", workspace=MAIN)

    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_INTENT_SUBMIT,
            intent=MemoryIntent(kind="write", topic_id="t", content="x"),
            access=read_context,
        )
    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            str(uuid4()),
            identity_scope=read_context.identity_scope,
            access=read_context,
        )


@pytest.mark.asyncio
async def test_observe_operation_cannot_cancel_task(wired):
    """task.observe 不授予取消：取消绑定 management.task（证据 3）。"""
    intent_context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    submission = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_INTENT_SUBMIT,
        intent=MemoryIntent(
            kind="write",
            topic_id="topic_boundary",
            content="observe cancel boundary",
            title="Observe Note",
        ),
        access=intent_context,
    )
    assert submission.accepted is True

    observe_context = await wired.access.authenticate(agent_id="a4", workspace=MAIN)
    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL,
            submission.task_id,
            access=observe_context,
        )


@pytest.mark.asyncio
async def test_actor_visible_read_enforces_visibility_and_workspace(wired):
    """行为许可与资源 owner 规则分别生效（证据 4）：PUBLIC 可读；PRIVATE
    只对目标 Agent；跨 Workspace 不可见；管理语义独立验证。"""
    public_atom = await _seed_public_fact(wired.store, alias="fact_public", content="visible")
    private_atom = MemoryAtom(
        meta=MetaData(
            workspace_identity=MAIN,
            source_agent_id="a1",
            access_policy=MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id="a2",
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

    a1_context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)

    visible = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(public_atom.id),
        access=a1_context,
    )
    assert visible is not None
    assert visible.payload.content == "visible"

    # 有 read 行为许可但资源 PRIVATE 只对 a2：对 a1 按不存在呈现
    secret = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(private_atom.id),
        access=a1_context,
    )
    assert secret is None

    # 跨 Workspace 的 canonical uuid 不可见
    outside = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(foreign.id),
        access=a1_context,
    )
    assert outside is None


@pytest.mark.asyncio
async def test_management_memory_reads_with_owner_semantics_independently(wired):
    """management.memory 的 owner-management 读取独立生效（证据 4）。"""
    atom = await _seed_public_fact(wired.store, alias="fact_owned", content="managed content")
    manage_context = await wired.access.authenticate(agent_id="a5", workspace=MAIN)

    result = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_GET,
        str(atom.id),
        identity_scope=manage_context.identity_scope,
        access=manage_context,
    )
    assert result is not None
    assert result.payload.content == "managed content"


@pytest.mark.asyncio
async def test_single_context_reused_across_different_permitted_operations(wired):
    """同一有效 context 先后执行 read/search/intent；未获准的管理操作仍拒绝（证据 6）。"""
    context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)

    public_atom = await _seed_public_fact(wired.store, alias="fact_reuse", content="reuse")
    read = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_READ,
        str(public_atom.id),
        access=context,
    )
    assert read is not None

    retrieval = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
        request=RetrievalRequest(
            semantic_query="reuse",
            identity_scope=context.identity_scope,
        ),
        access=context,
    )
    assert retrieval is not None

    # 换操作不重建身份，但方法所需的 operation 不在白名单内时仍拒绝
    with pytest.raises(OperationDeniedError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            str(public_atom.id),
            identity_scope=context.identity_scope,
            access=context,
        )


@pytest.mark.asyncio
async def test_interaction_submit_reaches_real_queue_via_global_route(wired):
    """interaction.submit 经真实全局路由与内存队列接纳；收据可用（无 Alice）。"""
    context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    receipt = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_INTERACTION_SUBMIT,
        payload=InteractionPayload(
            user_message="boundary question",
            mtp_traces=[],
            assistant_final_text="boundary answer",
            turn_events=[],
        ),
        access=context,
        requested_topic_id="topic_boundary",
        interaction_id="interaction_boundary_1",
    )
    assert receipt.interaction_id == "interaction_boundary_1"
    assert receipt.work_id.startswith("interaction:")
    record = await wired.queue.runtime.get(receipt.work_id)
    persisted = json.loads(record.item.payload)
    assert persisted["correlation"] == {}
    assert persisted["identity_scope"] == context.identity_scope.model_dump(mode="json")


@pytest.mark.asyncio
async def test_delayed_interaction_rechecks_access_and_accepted_work_survives_close(wired):
    """过期授权不能进入队列；已接纳交互只携带 scope，关闭网关后仍可应用。"""
    context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    payload = InteractionPayload(
        user_message="delayed question",
        mtp_traces=[],
        assistant_final_text="delayed answer",
        turn_events=[],
    )
    wired.clock.now += 60
    with pytest.raises(ScopeRequiredError) as exc_info:
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_INTERACTION_SUBMIT,
            access=context,
            payload=payload,
            interaction_id="delayed_interaction",
        )
    assert exc_info.value.details["reason"] == "context_expired"
    assert not await wired.queue.is_accepted("delayed_interaction")

    renewed = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    receipt = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_INTERACTION_SUBMIT,
        access=renewed,
        payload=payload,
        interaction_id="delayed_interaction",
        requested_topic_id="topic_delayed",
    )
    wired.access.gateway.close()
    await wired.queue.start()
    outcome = await wired.queue.wait(receipt.interaction_id, timeout=2)
    assert outcome.state.value == "succeeded"
    assert outcome.topic_id == "topic_delayed"


@pytest.mark.asyncio
async def test_delayed_intent_cannot_create_a_task_with_expired_access(wired):
    context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    wired.clock.now += 60
    with pytest.raises(ScopeRequiredError) as exc_info:
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_INTENT_SUBMIT,
            access=context,
            intent=MemoryIntent(kind="write", topic_id="topic_boundary", content="delayed"),
        )
    assert exc_info.value.details["reason"] == "context_expired"
    assert await wired.controller.list_tasks() == []


@pytest.mark.asyncio
async def test_intent_submit_and_observe_result_without_alice(wired):
    """意图提交 → 真实生成链 → task.observe 观察真实结果；跨 scope 拒绝（证据 8）。"""
    intent_context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
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

    # 任务归属按权威 identity_scope 投影判断：提交者自己的观察 context 可见
    observe_context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    result = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
        submission.task_id,
        access=observe_context,
    )
    assert result.status.value == "completed"
    assert result.canonical_alias == "memory_alias"
    assert result.identity_scope == intent_context.identity_scope

    # OTHER Workspace 的观察 Actor：归属不一致统一 not found，不泄漏存在性
    other_context = await _other_observe_context(wired)
    with pytest.raises(ResourceNotFoundError):
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_TASK_GET,
            submission.task_id,
            access=other_context,
        )


@pytest.mark.asyncio
async def test_system_management_facade_propagates_context_via_global_bus(wired):
    """System 管理门面与 adapter 走同一全局总线，context 完整传播（证据 5）。"""
    atom = await _seed_public_fact(wired.store, alias="fact_system", content="system path")
    manage_context = await wired.access.authenticate(agent_id="a5", workspace=MAIN)

    found = await wired.system_memory.get_memory(
        atom.id,
        identity_scope=manage_context.identity_scope,
        access=manage_context,
    )
    assert found is not None
    assert found.payload.content == "system path"

    # 无 permission 的 context 经同一 System 门面在 application 入口被拒
    read_context = await wired.access.authenticate(agent_id="a2", workspace=MAIN)
    with pytest.raises(OperationDeniedError):
        await wired.system_memory.get_memory(
            atom.id,
            identity_scope=read_context.identity_scope,
            access=read_context,
        )


@pytest.mark.asyncio
async def test_access_error_propagates_through_system_feedback_facade(wired):
    """访问错误沿 System 门面原语义传播，不被包装成服务不可用（A1 第 3.4 节）。"""
    read_context = await wired.access.authenticate(agent_id="a2", workspace=MAIN)

    # System 门面的 record_feedback 曾把 RuntimeError 分支包装为
    # MemoryLifecycleUnavailableError；行为授权拒绝必须以原错误冒出。
    with pytest.raises(OperationDeniedError):
        await wired.system_memory.record_feedback(
            uuid4(),
            identity_scope=read_context.identity_scope,
            positive=True,
            source="boundary",
            access=read_context,
        )


@pytest.mark.asyncio
async def test_system_task_facade_propagates_observe_context(wired):
    """System 任务门面补齐访问参数：观察经真实总线到达 application 检查。"""
    intent_context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    submission = await wired.global_bus.request(
        GlobalRoutes.PATCHOULI_MEMORY_INTENT_SUBMIT,
        intent=MemoryIntent(
            kind="write",
            topic_id="topic_boundary",
            content="system task facade",
            title="System Task Note",
        ),
        access=intent_context,
    )
    # 任务归属按权威 identity_scope 投影判断：观察 context 与提交者同 scope
    observe_context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)

    task = await wired.system_tasks.get_memory_task(
        submission.task_id,
        access=observe_context,
    )
    assert task is not None
    assert task.task_id == submission.task_id

    # OTHER Workspace 的观察 Actor：归属不一致按 not found 拒绝
    with pytest.raises(ResourceNotFoundError):
        await wired.system_tasks.get_memory_task(
            submission.task_id,
            access=await _other_observe_context(wired),
        )


async def _other_observe_context(wired):
    """OTHER Workspace 的观察 context（共享注册矩阵中的 OTHER/a4）。"""
    return await wired.access.authenticate(agent_id="a4", workspace=OTHER)


@pytest.mark.asyncio
async def test_expired_context_rejected_and_reauthentication_restores_access(wired):
    """超过认证有效区间后旧 context 拒绝；重新认证恢复（证据 7）。"""
    context = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    public_atom = await _seed_public_fact(wired.store, alias="fact_ttl", content="ttl")
    assert (
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_READ,
            str(public_atom.id),
            access=context,
        )
        is not None
    )

    wired.clock.now += 61

    with pytest.raises(ScopeRequiredError) as exc_info:
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_READ,
            str(public_atom.id),
            access=context,
        )
    assert exc_info.value.details["reason"] == "context_expired"

    renewed = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    assert (
        await wired.global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_READ,
            str(public_atom.id),
            access=renewed,
        )
        is not None
    )
