"""真实能力层、读取视图、Patchouli 管理链与失效事件协作；仅替换向量后端。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import RunBinding, WorkspaceAccessContext
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import (
    AgentProfile,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryChangeEvent,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomStatus,
    WorkspaceMemoryKey,
    WriteFocus,
)
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.patchouli.application.agent_profile_management_service import (
    AgentProfileManagementService,
)
from hivememory.patchouli.application.memory_management_service import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_change_publisher import MemoryChangePublisher
from hivememory.patchouli.memory_library.adapters.artifact import FilesystemArtifactStorageAdapter
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import (
    ArtifactStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.cache import AtomCache, ProfileCache, ProfileCacheEntry, WorkspaceEpochs
from hivememory.workspace.cache.invalidation import CacheInvalidator
from hivememory.workspace.capability.actor import ActorAgentService, ActorMemoryService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.contracts import (
    ResolveReferencesRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.contracts.errors import PendingUpdateNotAllowedError
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
    make_workspace_runtime,
)

WORKSPACE = make_workspace_identity(owner_user_id="u1")


class _UnusedPort:
    """只占位本测试不使用的服务，误调用时立即失败，不提供伪成功。"""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise AssertionError(f"本协作测试不应调用 {name}")

        return fail


class _VectorStore:
    """向量后端替身；只保存完整原子副本，不替换 adapter、store 或管理行为。"""

    def __init__(self) -> None:
        self.atoms: dict[WorkspaceMemoryKey, MemoryAtom] = {}

    async def upsert_memory(self, atom, **kwargs):
        key = WorkspaceMemoryKey(workspace_identity=atom.workspace_identity, memory_id=atom.id)
        self.atoms[key] = atom.model_copy(deep=True)

    async def get_memory(self, key):
        atom = self.atoms.get(key)
        return atom.model_copy(deep=True) if atom is not None else None

    async def get_memory_by_alias(self, alias, **kwargs):
        return next(
            (
                atom.model_copy(deep=True)
                for atom in self.atoms.values()
                if atom.index.alias == alias
            ),
            None,
        )

    async def get_memory_ids_by_alias(self, alias, *, query_filter, limit):
        # 当前集成链只有一个 Workspace；归属和 policy 仍由真实 adapter 校验。
        return [atom.id for atom in self.atoms.values() if atom.index.alias == alias][:limit]


@dataclass
class _Chain:
    """每个测试独立装配的协作边界。"""

    bus: GlobalSystemBus
    store: MidTermMemoryStore
    runtime: WorkspaceRuntime
    memories: ActorMemoryService
    profiles: ActorAgentService
    root_memories: RootMemoryService
    composition: AccessTestComposition
    writer: WorkspaceAccessContext
    reader: WorkspaceAccessContext
    manager: WorkspaceAccessContext


@pytest_asyncio.fixture
async def chain(tmp_path):
    """真实 RPC 与通知链，未参与本场景的生成/生命周期服务只作失败占位。"""
    local_bus = PatchouliBus()
    bus = GlobalSystemBus()
    install_intent_admission(bus)
    runtime = make_workspace_runtime(global_bus=bus)
    runtime.subscribe(bus)
    store = MidTermMemoryStore(
        QdrantStorageAdapter(_VectorStore()), change_publisher=MemoryChangePublisher(local_bus)
    )
    library = MemoryLibrary(ShortTermMemoryStore(), store, _UnusedPort())
    retrieval = RetrievalFamiliar(_UnusedPort(), library)
    artifacts = ArtifactStore(FilesystemArtifactStorageAdapter(str(tmp_path / "artifacts")))
    generation = MemoryGenerationFamiliar(
        generation_engine=_UnusedPort(),
        memory_library=library,
        artifact_engine=ArtifactEngine.from_store(artifacts),
    )
    local_bus.register(PatchouliLocalRoutes.MEMORY_GET, retrieval.get_memory)
    local_bus.register(
        PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES, retrieval.retrieve_by_aliases_async
    )
    local_bus.register(PatchouliLocalRoutes.GET_AGENT_PROFILE, retrieval.get_agent_profile)
    local_bus.register(PatchouliLocalRoutes.MEMORY_UPDATE, generation.update_external_memory)
    placeholder = _UnusedPort()
    bridge = PatchouliBridge(
        local_bus=local_bus,
        global_bus=bus,
        public_api=PatchouliPublicApi(
            memory=MemoryManagementService(bus=local_bus),
            agent_profiles=AgentProfileManagementService(bus=local_bus),
            chat=placeholder,
            memory_tasks=placeholder,
            interactions=placeholder,
            memory_intents=placeholder,
            topics=placeholder,
            readiness=placeholder,
        ),
    )
    bridge.mount()
    install_intent_admission(bus)
    composition = make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id=agent)
            for agent in ("writer", "reader", "system", "custom_agent", "omni_doll")
        ],
        default_workspace=WORKSPACE,
    )
    result = _Chain(
        bus=bus,
        store=store,
        runtime=runtime,
        composition=composition,
        memories=ActorMemoryService(
            bus,
            operation_authorizer=composition.authorizer,
            memory_reader=runtime.aliases,
            threads=composition.threads,
        ),
        root_memories=RootMemoryService(bus, operation_authorizer=composition.authorizer),
        profiles=ActorAgentService(
            operation_authorizer=composition.authorizer,
            profile_reader=runtime.profiles,
        ),
        writer=await composition.authenticate_thread(
            agent_id="writer", process_id="writer_process", register_process=True
        ),
        reader=await composition.authenticate_thread(
            agent_id="reader", process_id="reader_process", register_process=True
        ),
        manager=await composition.authenticate(agent_id="system"),
    )
    try:
        yield result
    finally:
        runtime.close()
        bridge.unmount()


def _atom(
    alias: str = "canonical", *, profile: bool = False, visibility: str = "PUBLIC"
) -> MemoryAtom:
    """创建可被真实存储与能力层读取的测试原子；PRIVATE 只对 writer 可见。"""
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="writer", user_id="u1", visibility=visibility),
        index=IndexLayer(
            title="缓存原子",
            summary="验证管理提交后的读取视图",
            alias=alias,
            memory_type=MemoryType.AGENT_PROFILE if profile else MemoryType.FACT,
        ),
        payload=PayloadLayer(
            content="old content", agent_config={"model_name": "old-model"} if profile else None
        ),
    )


@pytest.mark.asyncio
async def test_management_update_invalidates_actor_atom_read_before_return(chain):
    """管理更新返回时旧原子与 alias 已失效，后续能力层回读新内容。"""
    atom = _atom()
    await chain.store.upsert(atom)
    first = await chain.memories.resolve_references(
        ResolveReferencesRequest(("canonical",)), access=chain.reader
    )
    assert first[0].atom.payload.content == "old content"

    await chain.root_memories.update_memory(
        atom.id, content="new content", target_workspace=WORKSPACE, access=chain.manager
    )
    assert chain.runtime.stats()["atom_size"] == 0
    second = await chain.memories.resolve_references(
        ResolveReferencesRequest(("canonical",)), access=chain.reader
    )

    assert second[0].atom.payload.content == "new content"
    assert second[0].atom.meta.version == 2


@pytest.mark.asyncio
async def test_profile_source_update_invalidates_profile_configuration(chain):
    """Profile 的源原子管理变更后，下一次进程阶段解析取得新配置。"""
    atom = _atom("custom_agent", profile=True)
    await chain.store.upsert(atom)
    process_access = await chain.composition.authenticate(
        agent_id="custom_agent", binding=RunBinding.for_task_process("first_profile_process")
    )
    allocator = CPUAllocator(
        operation_authorizer=chain.composition.authorizer,
        profile_reader=chain.runtime.profiles,
        asset_reader=InMemoryWorkspaceAssetStore(),
    )
    first = await allocator.resolve_agent_profile(target_workspace=WORKSPACE, access=process_access)
    assert first.model_name == "old-model"

    await chain.root_memories.update_memory(
        atom.id,
        agent_config={"model_name": "new-model"},
        target_workspace=WORKSPACE,
        access=chain.manager,
    )
    assert chain.runtime.stats()["profile_size"] == 0
    next_process_access = await chain.composition.authenticate(
        agent_id="custom_agent", binding=RunBinding.for_task_process("next_profile_process")
    )
    second = await allocator.resolve_agent_profile(
        target_workspace=WORKSPACE, access=next_process_access
    )

    assert second.model_name == "new-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_cache", [False, True], ids=["canonical", "cached"])
async def test_cpu_profile_without_actor_operations_rejects_other_agents_private_source(
    chain, warm_cache: bool
):
    """阶段读取不检查 actor 操作，但冷读与缓存命中都保持 PRIVATE 可见性。"""
    atom = _atom("custom_agent", profile=True, visibility="PRIVATE")
    await chain.store.upsert(atom)
    if warm_cache:
        # writer 有权读取此 PRIVATE 源原子，先填充共享读取视图以覆盖命中路径。
        await chain.runtime.profiles.get(
            "custom_agent", scope=make_identity_scope(user_id="u1", agent_id="writer")
        )
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="custom_agent", allowed_operations=frozenset()
            )
        ],
        default_workspace=WORKSPACE,
    )
    access = await composition.authenticate(
        agent_id="custom_agent", binding=RunBinding.for_task_process("private_profile_process")
    )
    allocator = CPUAllocator(
        operation_authorizer=composition.authorizer,
        profile_reader=chain.runtime.profiles,
        asset_reader=InMemoryWorkspaceAssetStore(),
    )

    with pytest.raises(ResourceNotFoundError) as excinfo:
        await allocator.resolve_agent_profile(target_workspace=WORKSPACE, access=access)

    assert excinfo.value.details == {"resource": "agent_profile", "agent_alias": "custom_agent"}


@pytest.mark.asyncio
async def test_write_ack_is_shared_then_global_settlement_redirects(chain):
    """WRITE 登记可跨进程回读，结算事件经登记更新为 canonical redirect。"""
    pending = await chain.memories.submit_write_intent(
        SubmitWriteIntentRequest(WriteFocus(content="intent body")),
        access=await chain.composition.authenticate_thread(
            agent_id="writer", process_id="first_process", register_process=True
        ),
    )
    before = await chain.memories.resolve_references(
        ResolveReferencesRequest((pending.pending_alias,)), access=chain.reader
    )
    assert (before[0].kind, before[0].pending.focus.content) == ("pending", "intent body")
    atom = _atom()
    await chain.store.upsert(atom)
    await chain.bus.publish(
        GlobalEvents.PENDING_ATOM_SETTLED,
        settlement=PendingAtomSettlement(
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="canonical",
            canonical_uuid=str(atom.id),
        ),
    )
    after = await chain.memories.resolve_references(
        ResolveReferencesRequest((pending.pending_alias,)), access=chain.reader
    )

    assert (after[0].kind, after[0].canonical_uuid, after[0].atom.payload.content) == (
        "redirect",
        str(atom.id),
        "old content",
    )


@pytest.mark.asyncio
async def test_update_rejects_pending_and_missing_bases_without_registering(chain):
    """UPDATE 基础不合法时不产生新意图，pending 与缺失分别表达领域错误。"""
    pending = await chain.memories.submit_write_intent(
        SubmitWriteIntentRequest(WriteFocus(content="pending base")),
        access=chain.writer,
    )
    with pytest.raises(PendingUpdateNotAllowedError):
        await chain.memories.submit_update_intent(
            SubmitUpdateIntentRequest(pending.pending_alias, "revise"),
            access=chain.writer,
        )
    with pytest.raises(ResourceNotFoundError):
        await chain.memories.submit_update_intent(
            SubmitUpdateIntentRequest("missing", "revise"),
            access=chain.writer,
        )

    assert chain.runtime.intents.size == 1
    assert (
        chain.runtime.intents.get(pending.pending_alias, WORKSPACE).status
        == PendingAtomStatus.MATERIALIZING
    )


@pytest.mark.asyncio
async def test_update_intent_on_private_base_is_indistinguishable_from_missing(chain):
    """读不到基础的 Actor 既读不到 UPDATE 意图，也不能借 UPDATE 错误类型探知它存在。"""
    atom = _atom("private_base", visibility="PRIVATE")
    await chain.store.upsert(atom)
    pending = await chain.memories.submit_update_intent(
        SubmitUpdateIntentRequest("private_base", "revise", "secret revision"),
        access=chain.writer,
    )

    hidden = await chain.memories.resolve_references(
        ResolveReferencesRequest((pending.pending_alias,)), access=chain.reader
    )
    visible = await chain.memories.resolve_references(
        ResolveReferencesRequest((pending.pending_alias,)), access=chain.writer
    )
    with pytest.raises(ResourceNotFoundError):
        await chain.memories.submit_update_intent(
            SubmitUpdateIntentRequest(pending.pending_alias, "revise again"),
            access=chain.reader,
        )

    assert (hidden[0].kind, hidden[0].pending) == ("not_found", None)
    assert (visible[0].kind, visible[0].pending.focus.content) == ("pending", "secret revision")
    assert chain.runtime.intents.size == 1


@pytest.mark.asyncio
async def test_update_success_invalidates_base_and_claims_canonical_uuid(chain):
    """UPDATE 基础由能力层确定 UUID，ACK 后基础缓存失效且物化请求引用该原子。"""
    atom = _atom()
    await chain.store.upsert(atom)
    await chain.memories.resolve_references(
        ResolveReferencesRequest(("canonical",)), access=chain.writer
    )

    pending = await chain.memories.submit_update_intent(
        SubmitUpdateIntentRequest("canonical", "revise", "replacement"),
        access=chain.writer,
    )
    assert chain.runtime.stats()["atom_size"] == 0
    task = pending

    assert (task.pending_alias, task.focus.base_uuid, task.focus.content) == (
        pending.pending_alias,
        str(atom.id),
        "replacement",
    )


@pytest.mark.asyncio
async def test_update_rejects_settled_handle_but_accepts_formal_alias(chain):
    """结算句柄可 READ redirect，UPDATE 仍要求正式 atom alias。"""
    pending = await chain.memories.submit_write_intent(
        SubmitWriteIntentRequest(WriteFocus(content="shared")),
        access=chain.writer,
    )
    atom = _atom()
    await chain.store.upsert(atom)
    await chain.bus.publish(
        GlobalEvents.PENDING_ATOM_SETTLED,
        settlement=PendingAtomSettlement(
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="canonical",
            canonical_uuid=str(atom.id),
        ),
    )

    with pytest.raises(ResourceNotFoundError):
        await chain.memories.submit_update_intent(
            SubmitUpdateIntentRequest(pending.pending_alias, "revise"),
            access=await chain.composition.authenticate_thread(
                agent_id="writer", process_id="second_process"
            ),
        )
    assert chain.runtime.intents.size == 1
    revision = await chain.memories.submit_update_intent(
        SubmitUpdateIntentRequest("canonical", "revise"),
        access=await chain.composition.authenticate_thread(
            agent_id="writer", process_id="second_process", register_process=True
        ),
    )

    assert (revision.source_verb, revision.focus.base_uuid) == ("UPDATE", str(atom.id))


@pytest.mark.asyncio
async def test_subscriber_failure_after_invalidation_preserves_commit_and_advanced_epoch(chain):
    """失效后订阅者抛异常，缓存已清理、代次已推进，存储提交仍正常返回。"""
    atom = _atom("profile_source")
    await chain.store.upsert(atom)
    atoms = AtomCache(4)
    profiles = ProfileCache(4)
    epochs = WorkspaceEpochs()
    atoms.put(atom)
    profiles.put(
        WORKSPACE,
        "profile_source",
        ProfileCacheEntry(
            AgentProfile(model_name="old-model"), MemoryAccessPolicy.public(), atom.id, 1
        ),
    )

    class BrokenInvalidator(CacheInvalidator):
        """模拟失效后出现观测代码错误，验证失效不能被回滚。"""

        async def on_changed(self, *, payload: MemoryChangeEvent) -> None:
            await super().on_changed(payload=payload)
            raise RuntimeError("after invalidation")

    subscriber = BrokenInvalidator(atom_cache=atoms, profile_cache=profiles, epochs=epochs)
    subscriber.subscribe(chain.bus)
    try:
        updated = await chain.root_memories.update_memory(
            atom.id, content="committed", target_workspace=WORKSPACE, access=chain.manager
        )
        read = (
            await chain.memories.resolve_references(
                ResolveReferencesRequest((atom.index.alias,)), access=chain.reader
            )
        )[0].atom

        assert (updated.payload.content, read.payload.content) == ("committed", "committed")
        assert (
            atoms.get_by_alias(WORKSPACE, "profile_source"),
            profiles.get(WORKSPACE, "profile_source"),
            epochs.current(WORKSPACE),
        ) == (None, None, 1)
    finally:
        subscriber.unsubscribe()


@pytest.mark.asyncio
async def test_registry_unsubscribe_stops_terminal_events_and_resubscribe_is_idempotent(chain):
    """重复装配不重复执行，取消订阅后全局失败事件不再改登记。"""
    pending = await chain.memories.submit_write_intent(
        SubmitWriteIntentRequest(WriteFocus(content="shared")),
        access=chain.writer,
    )
    chain.runtime.unsubscribe()
    await chain.bus.publish(
        GlobalEvents.PENDING_ATOM_FAILED,
        pending_alias=pending.pending_alias,
        intent_id=pending.intent_id,
    )
    assert (
        chain.runtime.intents.get(pending.pending_alias, WORKSPACE).status
        == PendingAtomStatus.MATERIALIZING
    )
    chain.runtime.subscribe(chain.bus)
    chain.runtime.subscribe(chain.bus)
    await chain.bus.publish(
        GlobalEvents.PENDING_ATOM_FAILED,
        pending_alias=pending.pending_alias,
        intent_id=pending.intent_id,
    )

    assert (
        chain.runtime.intents.get(pending.pending_alias, WORKSPACE).status
        == PendingAtomStatus.FAILED
    )
