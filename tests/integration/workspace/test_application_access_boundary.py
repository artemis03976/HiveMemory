"""统一认证网关、共享操作授权与公开 application 链路的集成测试。

真实协作边界：GlobalSystemBus + PatchouliBridge + PatchouliPublicApi（真实
application 服务）+ 统一认证网关（真实两类注册表）+ Patchouli local bus
（真实 familiar/controller/coordinator handler）+ 真实 Qdrant ``:memory:``
存储 + 真实 InteractionSubmissionQueue。只替换进程外依赖：embedding 用确
定性二维向量、生成执行（LLM）与 Topic 会话读取用确定性实现。不创建
Alice/PendingAtomRuntime/MTP（headless）。

保护的契约：
- 两项认证在统一网关一次完成，得到工作 actor；未登记 principal、不匹配
  adapter 与无准入记录的 Actor 分别以稳定 reason 拒绝；
- actor 能力执行 operation 检查，root 能力只向保留的 system actor 开放，
  Patchouli 公开路由只接收授权点组装的 ``IdentityScope``；
- 同一 principal 多 Actor、同一 Actor 多 Workspace 的许可互不串扰；
- 空白名单可进入但资源动作全拒绝；operation 互不隐含；
- 工作 actor 与单次 operation 解耦：同一工作 actor 先后执行不同获准操作；
- 交互提交由系统以 root scope 执行，参与的 Agent 记为发起者；System 停止
  先关网关（拒绝新认证），已接纳的交互照常应用；
  授权拒绝不产生副作用、不包装成服务不可用。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import CallerPrincipal, WorkingActor, WorkspaceOperation
from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ResourceNotFoundError,
)
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    InteractionSubmissionService,
    MemoryIntentSubmissionService,
    MemoryManagementService,
    MemoryTaskManagementService,
    TopicManagementService,
)
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_intent import MemoryIntent
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
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
from hivememory.workspace.contracts import ResolveReferencesRequest, RetrieveRequest
from tests.helpers.memory import make_memory_metadata
from tests.helpers.sessions import make_interaction_payload, make_logical_block
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
    make_workspace_read_view,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")

READ = WorkspaceOperation.MEMORY_READ
SEARCH = WorkspaceOperation.MEMORY_SEARCH
INTENT = WorkspaceOperation.MEMORY_WRITE


def _access_records() -> list:
    """两类注册表共用的访问矩阵：同 owner 多 Actor、同 Actor 多 Workspace。"""
    return [
        # MAIN：a1 可读、检索和提交；a2 仅读取；a3 空白名单。
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=MAIN.workspace_id,
            agent_id="a1",
            allowed_operations={READ, SEARCH, INTENT},
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
        # OTHER：a1 的白名单与 MAIN 不同；a2 无准入记录。
        make_actor_access_record(
            owner_user_id="u1",
            workspace_id=OTHER.workspace_id,
            agent_id="a1",
            allowed_operations={READ, INTENT},
        ),
        # root 在每个 Workspace 独立准入，不配置 actor operation。
        *[
            make_actor_access_record(
                owner_user_id="u1", workspace_id=workspace.workspace_id, agent_id="system"
            )
            for workspace in (MAIN, OTHER)
        ],
    ]


class _DeterministicEmbedding:
    """进程外 embedding 依赖的确定性替换。"""

    async def encode(self, *, dense_texts, sparse_texts=None):
        return [0.25, 0.75]


class _FakeTopicData:
    topic_title = "boundary topic"
    topic_summary = "boundary topic summary"
    state_summary = "boundary state summary"

    def recent_blocks(self, limit):
        return [make_logical_block("question", "answer")]


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
    coordinator = MemoryGenerationCoordinator(bus=local_bus, recent_block_limit=5)
    await controller.start()
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_GET,
        familiar.get_memory,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_LIST,
        familiar.list_memories,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_RETRIEVE,
        familiar.retrieve_async,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
        familiar.retrieve_by_aliases_async,
    )
    local_bus.register(
        PatchouliLocalRoutes.GET_AGENT_PROFILE,
        familiar.get_agent_profile,
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
        PatchouliLocalRoutes.MEMORY_TASK_LIST,
        controller.list_tasks,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_WAIT,
        controller.wait_task,
    )
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_ADMIT_GENERATION,
        controller.admit_generation,
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
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id=None,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
    ):
        return target_topic_id

    queue = InteractionSubmissionQueue(_apply)

    access = make_access_composition(_access_records(), default_workspace=MAIN)

    # Patchouli 公开路由是授权点以下的资源 owner：不注入共享行为检查，
    # 只接收授权点组装的 IdentityScope。
    memory_service = MemoryManagementService(bus=local_bus)
    profile_service = AgentProfileManagementService(bus=local_bus)
    task_service = MemoryTaskManagementService(bus=local_bus)
    interactions = InteractionSubmissionService(interaction_queue=queue)
    memory_intents = MemoryIntentSubmissionService(bus=local_bus)
    topics = TopicManagementService(bus=local_bus)

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

    # actor 读取走共享读取视图；root 管理直接经同一全局总线访问资源 owner。
    runtime = make_workspace_read_view(global_bus)
    actor_memory = ActorMemoryService(
        global_bus=global_bus,
        operation_authorizer=access.authorizer,
        memory_reader=runtime.memory,
        threads=access.threads,
    )
    root_memory = RootMemoryService(global_bus=global_bus, operation_authorizer=access.authorizer)
    root_profiles = RootAgentProfileService(
        global_bus=global_bus, operation_authorizer=access.authorizer
    )
    root_tasks = RootMemoryTaskService(
        global_bus=global_bus, operation_authorizer=access.authorizer
    )

    try:
        yield _Wired(
            global_bus=global_bus,
            store=qdrant.store,
            controller=controller,
            queue=queue,
            access=access,
            actor_memory=actor_memory,
            root_memory=root_memory,
            root_profiles=root_profiles,
            root_tasks=root_tasks,
        )
    finally:
        await queue.stop()
        await controller.stop()
        runtime.close()
        access.gateway.close()
        bridge.unmount()
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
        actor_memory,
        root_memory,
        root_profiles,
        root_tasks,
    ):
        self.global_bus = global_bus
        self.store = store
        self.controller = controller
        self.queue = queue
        self.access = access
        self.actor_memory = actor_memory
        self.root_memory = root_memory
        self.root_profiles = root_profiles
        self.root_tasks = root_tasks


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
        meta=make_memory_metadata(
            user_id=workspace.owner_user_id,
            source_agent_id=agent_id,
            workspace_id=workspace.workspace_id,
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


async def _submit_intent(wired, actor: WorkingActor):
    """以操作授权者组装的可信 scope 提交一条 WRITE 意图（a1 获准）。"""
    scope = wired.access.authorizer.authorize_agent(actor, INTENT, MAIN)
    return await wired.global_bus.request(
        PatchouliRoutes.MEMORY_INTENT_SUBMIT,
        intent=MemoryIntent(
            kind="write",
            topic_id="topic_boundary",
            intent_id=f"intent_{uuid4().hex[:12]}",
            pending_alias=f"draft_boundary_note_{uuid4().hex[:4]}",
            content="remember the boundary fact",
            title="Boundary Note",
        ),
        identity_scope=scope,
    )


# ---------------------------------------------------------------------------
# 验收用例
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_source_denied_before_workspace_admission(wired):
    """未登记来源在系统接入层拒绝，不进入 Workspace 准入。"""
    with pytest.raises(AdmissionDeniedError) as exc_info:
        await wired.access.gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal("local-process:stranger"),
            credential=None,
            actor=ActorIdentity(user_id="u1", agent_id="a1"),
            workspace=MAIN,
        )
    assert exc_info.value.details["reason"] == "unknown_principal"


@pytest.mark.asyncio
async def test_mismatched_adapter_denied_at_principal_authentication(wired):
    """来源 adapter 与接入登记不匹配时在第一项认证拒绝。"""
    with pytest.raises(AdmissionDeniedError) as exc_info:
        await wired.access.authenticate(agent_id="a1", adapter="http")
    assert exc_info.value.details["reason"] == "adapter_mismatch"


@pytest.mark.asyncio
async def test_same_owner_actors_have_different_admission_across_workspaces(wired):
    """同一 owner：a2 在 MAIN 获准、在 OTHER 无准入记录；权限互不串扰。

    授权返回的 scope 携带准入的 workspace。
    """
    main_actor = await wired.access.authenticate(agent_id="a2", workspace=MAIN)
    scope = wired.access.authorizer.authorize_agent(main_actor, READ, MAIN)
    assert scope.workspace_identity == MAIN

    with pytest.raises(AdmissionDeniedError) as exc_info:
        await wired.access.authenticate(agent_id="a2", workspace=OTHER)
    assert exc_info.value.details["reason"] == "actor_not_admitted"


@pytest.mark.asyncio
async def test_empty_whitelist_admits_entry_but_denies_every_resource_action(wired):
    """空白名单 Actor 可进入，但资源动作在授权点全部拒绝。

    actor 读取在 backing 调用前检查 operation；意图提交路径在组装可信
    scope 后经公开路由到达真实生成链，缺少许可时不能登记生成任务。
    """
    actor = await wired.access.authenticate(agent_id="a3", workspace=MAIN)

    with pytest.raises(OperationDeniedError) as read_error:
        (
            await wired.actor_memory.resolve_references(
                ResolveReferencesRequest((str(str(uuid4())),)), actor=actor
            )
        )[0].atom
    assert read_error.value.details["reason"] == "operation_not_allowed"
    with pytest.raises(OperationDeniedError) as intent_error:
        wired.access.authorizer.authorize_agent(actor, INTENT, MAIN)
    assert intent_error.value.details["reason"] == "operation_not_allowed"


@pytest.mark.asyncio
async def test_read_only_actor_cannot_submit_intent_or_reach_management(wired):
    """仅有 read 的 Actor 无法写；也不能借读取进入管理路由。"""
    read_actor = await wired.access.authenticate(agent_id="a2", workspace=MAIN)

    with pytest.raises(OperationDeniedError):
        wired.access.authorizer.authorize_agent(read_actor, INTENT, MAIN)
    with pytest.raises(OperationDeniedError) as root_error:
        await wired.root_memory.get_memory(uuid4(), target_workspace=MAIN, actor=read_actor)
    assert root_error.value.details["reason"] == "root_required"


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", ["a1", "a3"], ids=["actor-operations", "empty-whitelist"])
async def test_non_root_actor_cannot_observe_or_cancel_task(wired, agent_id):
    """actor 操作白名单无论是否为空，都不授予任务观察与取消的 root 能力。"""
    intent_actor = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    submission = await _submit_intent(wired, intent_actor)
    assert submission.accepted is True
    final = await wired.controller.wait_task(submission.task_id)
    assert final.status.value == "completed"

    agent_actor = await wired.access.authenticate(agent_id=agent_id, workspace=MAIN)
    for method in (wired.root_tasks.get_memory_task, wired.root_tasks.cancel_memory_task):
        with pytest.raises(OperationDeniedError) as exc_info:
            await method(
                submission.task_id,
                target_workspace=MAIN,
                actor=agent_actor,
            )
        assert exc_info.value.details["reason"] == "root_required"
    with pytest.raises(OperationDeniedError) as list_error:
        await wired.root_tasks.list_memory_tasks(target_workspace=MAIN, actor=agent_actor)
    assert list_error.value.details["reason"] == "root_required"

    # 被拒的取消不改变权威任务终态：system 经 root API 仍观察到 completed。
    root_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)
    task = await wired.root_tasks.get_memory_task(
        submission.task_id,
        target_workspace=MAIN,
        actor=root_actor,
    )
    assert task.status.value == "completed"


@pytest.mark.asyncio
async def test_actor_visible_read_enforces_visibility_and_workspace(wired):
    """行为许可与资源 owner 规则分别生效：PUBLIC 可读；PRIVATE
    只对目标 Agent；跨 Workspace 不可见；管理语义独立验证。"""
    public_atom = await _seed_public_fact(wired.store, alias="fact_public", content="visible")
    private_atom = MemoryAtom(
        meta=make_memory_metadata(
            user_id=MAIN.owner_user_id,
            source_agent_id="a1",
            workspace_id=MAIN.workspace_id,
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

    a1_actor = await wired.access.authenticate(agent_id="a1", workspace=MAIN)

    visible = (
        await wired.actor_memory.resolve_references(
            ResolveReferencesRequest((public_atom.index.alias,)), actor=a1_actor
        )
    )[0].atom
    assert visible.payload.content == "visible"

    # 有 read 行为许可但资源 PRIVATE 只对 a2：对 a1 按不存在呈现
    secret = (
        await wired.actor_memory.resolve_references(
            ResolveReferencesRequest((private_atom.index.alias,)), actor=a1_actor
        )
    )[0].atom
    assert secret is None

    # 跨 Workspace 的正式 alias 不可见
    outside = (
        await wired.actor_memory.resolve_references(
            ResolveReferencesRequest((foreign.index.alias,)), actor=a1_actor
        )
    )[0].atom
    assert outside is None


@pytest.mark.asyncio
async def test_management_memory_reads_with_owner_semantics_independently(wired):
    """system 的 root 管理读取按 Workspace 归属独立生效，不要求 actor 操作。"""
    atom = await _seed_public_fact(wired.store, alias="fact_owned", content="managed content")
    atom.meta.access_policy = MemoryAccessPolicy(
        visibility=MemoryVisibility.PRIVATE, target_agent_id="a1"
    )
    await wired.store.upsert(atom)
    manage_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)

    found = await wired.root_memory.get_memory(atom.id, target_workspace=MAIN, actor=manage_actor)
    assert found.payload.content == "managed content"


@pytest.mark.asyncio
async def test_management_profile_list_reads_with_owner_semantics(wired):
    """root Profile 列表与 Memory 管理读取同为 owner-management：PRIVATE/TEAM Profile 也列出。

    system 不是任何受限 Profile 的目标；若列表按 system 的可见性过滤，管理视图只剩 PUBLIC。
    """
    policies = {
        "agent_public": MemoryAccessPolicy.public(),
        "agent_private": MemoryAccessPolicy(
            visibility=MemoryVisibility.PRIVATE, target_agent_id="a1"
        ),
        "agent_team": MemoryAccessPolicy(visibility=MemoryVisibility.TEAM, target_team_id="t1"),
    }
    for alias, policy in policies.items():
        atom = await _seed_public_fact(wired.store, alias=alias, content=alias)
        atom.index.memory_type = MemoryType.AGENT_PROFILE
        atom.meta.access_policy = policy
        await wired.store.upsert(atom)
    await _seed_public_fact(wired.store, alias="fact_not_profile", content="fact")
    manage_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)

    profiles = await wired.root_profiles.list_agent_profiles(
        target_workspace=MAIN, actor=manage_actor
    )

    assert sorted(profile.index.alias for profile in profiles) == sorted(policies)


@pytest.mark.asyncio
async def test_single_working_actor_reused_across_different_permitted_operations(wired):
    """同一工作 actor 可先后执行 read/search；actor 操作不授予 root 管理能力。"""
    actor = await wired.access.authenticate(agent_id="a1", workspace=MAIN)

    public_atom = await _seed_public_fact(wired.store, alias="fact_reuse", content="reuse")
    read = (
        await wired.actor_memory.resolve_references(
            ResolveReferencesRequest((public_atom.index.alias,)), actor=actor
        )
    )[0].atom
    assert read.payload.content == "reuse"

    # 检索参数由调用方提交：检索请求由能力层用授权返回的可信 scope 构造，
    # 替身引擎无命中。
    retrieval = await wired.actor_memory.retrieve(
        RetrieveRequest(semantic_query="reuse"), actor=actor
    )
    assert retrieval == []

    # 换 actor 操作不重建身份，调用 root API 仍要求保留的 system actor。
    with pytest.raises(OperationDeniedError) as root_error:
        await wired.root_memory.get_memory(public_atom.id, target_workspace=MAIN, actor=actor)
    assert root_error.value.details["reason"] == "root_required"


@pytest.mark.asyncio
async def test_interaction_submit_reaches_real_queue_with_participant_as_initiator(wired):
    """交互由系统以 root scope 经真实全局路由提交，参与的 Agent 记为发起者。

    参与者 a3 是空白名单的 Agent，交互提交不要求它持有 actor 操作；队列记录的
    发起者是参与者而不是执行提交的 system，归属取 root scope 的 Workspace。队列
    尚未启动，提交以零时限等待，接纳后以 ``pending`` 交回。
    """
    system_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)
    scope = wired.access.authorizer.authorize_root(system_actor, MAIN)
    participant = ActorIdentity(user_id="u1", agent_id="a3")
    result = await wired.global_bus.request(
        PatchouliRoutes.INTERACTION_SUBMIT,
        payload=make_interaction_payload("boundary question", "boundary answer"),
        identity_scope=scope,
        participant=participant,
        requested_topic_id="topic_boundary",
        interaction_id="interaction_boundary_1",
        timeout_seconds=0,
    )
    assert (result.interaction_id, result.state) == ("interaction_boundary_1", "pending")
    record = await wired.queue.runtime.get("interaction:interaction_boundary_1")
    persisted = json.loads(record.item.payload)
    assert persisted["correlation"] == {}
    assert persisted["belong_to"] == MAIN.model_dump(mode="json")
    assert persisted["from_actor"] == participant.model_dump(mode="json")


@pytest.mark.asyncio
async def test_gateway_close_rejects_new_authentication_and_accepted_work_survives(wired):
    """网关关闭只拒绝新认证：已接纳的交互携带独立的归属与发起者，关闭后仍可应用。

    System 停止先关闭网关；同一 interaction_id 的重放在关闭前照常被接纳。队列尚未
    启动，两次提交都以零时限等待并以 ``pending`` 交回。
    """
    system_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)
    scope = wired.access.authorizer.authorize_root(system_actor, MAIN)
    participant = ActorIdentity(user_id="u1", agent_id="a1")
    payload = make_interaction_payload("delayed question", "delayed answer")
    first = await wired.global_bus.request(
        PatchouliRoutes.INTERACTION_SUBMIT,
        payload=payload,
        identity_scope=scope,
        participant=participant,
        interaction_id="delayed_interaction",
        requested_topic_id="topic_delayed",
        timeout_seconds=0,
    )
    replay = await wired.global_bus.request(
        PatchouliRoutes.INTERACTION_SUBMIT,
        payload=payload,
        identity_scope=scope,
        participant=participant,
        interaction_id="delayed_interaction",
        requested_topic_id="topic_delayed",
        timeout_seconds=0,
    )
    assert (first.state, replay.state) == ("pending", "pending")

    wired.access.gateway.close()
    with pytest.raises(AdmissionDeniedError) as auth_error:
        await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    assert auth_error.value.details["reason"] == "authentication_gateway_closed"

    await wired.queue.start()
    outcome = await wired.queue.wait(first.interaction_id, timeout=2)
    assert outcome.state.value == "succeeded"
    assert outcome.topic_id == "topic_delayed"


@pytest.mark.asyncio
async def test_intent_submit_and_observe_result_without_alice(wired):
    """actor 意图进入真实生成链，system 经 root API 观察结果；跨 Workspace 拒绝。"""
    intent_actor = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    submission = await _submit_intent(wired, intent_actor)
    assert submission.accepted is True
    final = await wired.controller.wait_task(submission.task_id)
    assert final.status.value == "completed"

    # 观察独立于提交者的 actor 操作，按 root 授权与权威 belong_to 判断归属。
    observe_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)
    result = await wired.root_tasks.get_memory_task(
        submission.task_id, target_workspace=MAIN, actor=observe_actor
    )
    assert result.status.value == "completed"
    assert result.canonical_alias == "memory_alias"
    authorized = wired.access.authorizer.authorize_agent(intent_actor, INTENT, MAIN)
    assert result.belong_to == authorized.workspace_identity
    assert result.from_actor == authorized.actor_identity

    # OTHER Workspace 的 root：归属不一致统一 not found，不泄漏存在性。
    other_actor = await _other_observe_actor(wired)
    with pytest.raises(ResourceNotFoundError):
        await wired.root_tasks.get_memory_task(
            submission.task_id, target_workspace=OTHER, actor=other_actor
        )


@pytest.mark.asyncio
async def test_system_management_facade_propagates_authorized_scope_via_global_bus(wired):
    """System 管理门面与 adapter 走同一全局总线，授权组装的 scope 完整传播。"""
    atom = await _seed_public_fact(wired.store, alias="fact_system", content="system path")
    manage_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)

    found = await wired.root_memory.get_memory(atom.id, target_workspace=MAIN, actor=manage_actor)
    assert found.payload.content == "system path"

    # actor 的读取许可不能进入 System root 管理门面。
    read_actor = await wired.access.authenticate(agent_id="a2", workspace=MAIN)
    with pytest.raises(OperationDeniedError) as exc_info:
        await wired.root_memory.get_memory(atom.id, target_workspace=MAIN, actor=read_actor)
    assert exc_info.value.details["reason"] == "root_required"


@pytest.mark.asyncio
async def test_access_error_propagates_through_system_feedback_facade(wired):
    """访问错误沿 System 门面原语义传播，不被包装成服务不可用。"""
    read_actor = await wired.access.authenticate(agent_id="a2", workspace=MAIN)

    # root 反馈必须保留授权拒绝，不能误报生命周期服务不可用。
    with pytest.raises(OperationDeniedError) as exc_info:
        await wired.root_memory.record_feedback(
            uuid4(),
            target_workspace=MAIN,
            positive=True,
            source="boundary",
            actor=read_actor,
        )
    assert exc_info.value.details["reason"] == "root_required"


@pytest.mark.asyncio
async def test_system_task_facade_lists_only_resident_workspace_tasks(wired):
    """System 任务门面的观察列表按驻留 Workspace 过滤。"""
    intent_actor = await wired.access.authenticate(agent_id="a1", workspace=MAIN)
    submission = await _submit_intent(wired, intent_actor)
    await wired.controller.wait_task(submission.task_id)

    observe_actor = await wired.access.authenticate(agent_id="system", workspace=MAIN)
    resident = await wired.root_tasks.list_memory_tasks(target_workspace=MAIN, actor=observe_actor)
    assert [task.task_id for task in resident] == [submission.task_id]

    # OTHER Workspace 的 root：列表为空，不泄漏其他 Workspace 的任务。
    other_actor = await _other_observe_actor(wired)
    assert await wired.root_tasks.list_memory_tasks(target_workspace=OTHER, actor=other_actor) == []


async def _other_observe_actor(wired):
    """OTHER Workspace 独立准入的 system 工作 actor，不借 MAIN 的授权切换目标。"""
    return await wired.access.authenticate(agent_id="system", workspace=OTHER)
