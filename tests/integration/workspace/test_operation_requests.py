"""真实操作入口、线程 context 与能力层协作，验证分派、权限和撤销语义。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import RunBinding, WorkspaceOperation
from hivememory.core.errors import OperationDeniedError, ResourceNotFoundError, ScopeRequiredError
from hivememory.core.models import (
    ActorIdentity,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomStatus,
    ResolvedAgentProfile,
    UpdateFocus,
    WriteFocus,
)
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.memory_intent import MemoryIntent
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import (
    DeriveCallThreadRequest,
    OperationRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
    ThreadEndedError,
)
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_composition, make_actor_access_record


class _OperationHarness:
    """真实入口、总线 backing 与读取视图；资源 owner 仅在公共 RPC 边界替换。"""

    def __init__(self, *, access) -> None:
        operation_authorizer = access.authorizer
        self.bus = GlobalSystemBus()
        self.admission = install_intent_admission(self.bus)
        self.memories: dict[str, MemoryAtom] = {}
        self.profile_source = MemoryAtom(
            meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
            index=IndexLayer(
                alias="call_target", title="CALL 目标", memory_type=MemoryType.AGENT_PROFILE
            ),
            payload=PayloadLayer(
                content="CALL 目标人设", agent_config={"model_name": "call-model"}
            ),
        )
        self.citations: list[dict] = []
        self.owner_calls: list[str] = []
        self.backing = BusCanonicalReadBackend(self.bus)
        self.runtime = WorkspaceRuntime(backing=self.backing, atom_capacity=32, profile_capacity=16)
        self.registry = self.runtime.intents
        for route, handler in (
            (PatchouliRoutes.MEMORY_READ, self._read),
            (PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, self._retrieve_by_aliases),
            (PatchouliRoutes.MEMORY_RETRIEVE, self._retrieve),
            (PatchouliRoutes.GET_AGENT_PROFILE, self._get_profile),
            (PatchouliRoutes.RECORD_MEMORY_CITATION, self._record_citation),
        ):
            self.bus.register(route, handler)
        self.entry = WorkspaceOperationEntry(
            ActorMemoryService(
                self.bus,
                operation_authorizer=operation_authorizer,
                memory_reader=self.runtime.aliases,
                threads=access.threads,
            ),
            agent=ActorAgentService(
                operation_authorizer=operation_authorizer,
                profile_reader=self.runtime.profiles,
            ),
            threads=access.threads,
        )

    async def _read(self, memory_id, *, identity_scope):
        self.owner_calls.append(PatchouliRoutes.MEMORY_READ)
        return next((atom for atom in self.memories.values() if atom.id == memory_id), None)

    async def _retrieve_by_aliases(self, aliases, *, identity_scope):
        self.owner_calls.append(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES)
        return [self.memories[alias] for alias in aliases if alias in self.memories]

    async def _retrieve(self, request):
        self.owner_calls.append(PatchouliRoutes.MEMORY_RETRIEVE)
        return list(self.memories.values())[: request.top_k]

    async def _get_profile(self, agent_alias, *, identity_scope):
        self.owner_calls.append(PatchouliRoutes.GET_AGENT_PROFILE)
        if agent_alias != self.profile_source.index.alias:
            raise ResourceNotFoundError(
                details={"resource": "agent_profile", "agent_alias": agent_alias}
            )
        return ResolvedAgentProfile(
            profile=AgentProfile.from_atom(self.profile_source),
            access_policy=self.profile_source.meta.access_policy,
            source_memory_id=self.profile_source.id,
            source_version=self.profile_source.meta.version,
        )

    async def _record_citation(self, *, memory_id, source, identity_scope):
        """正式引用的副作用由 RPC 接收端记录，不能由测试伪造服务返回结果。"""
        self.owner_calls.append(PatchouliRoutes.RECORD_MEMORY_CITATION)
        self.citations.append(
            {"memory_id": memory_id, "source": source, "identity_scope": identity_scope}
        )


def _seed_base(harness) -> None:
    """为 READ、RUN 与 UPDATE 准备同一个可见正式原子。"""
    harness.memories["base"] = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            alias="base", title="基础", summary="基础内容", memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="正式基础内容"),
    )


@pytest.fixture
def operation_setup():
    """持久化用内存替身，其余边界使用同一份真实授权与登记。"""
    access = make_access_composition([make_actor_access_record()])
    harness = _OperationHarness(access=access)
    _seed_base(harness)
    yield harness, access
    harness.runtime.close()
    access.gateway.close()
    access.gateway.revoke_all_contexts()


async def _submit(entry, request, *, access):
    """CPU 侧的提交面：CALL 派生走入口的生命周期方法，记忆请求走统一分派。"""
    if isinstance(request, DeriveCallThreadRequest):
        return await entry.derive_child_thread(request, access=access)
    return await entry.execute(request, access=access)


async def _derive_thread(access, process_id="process", *, agent_id="test_agent"):
    """入口身份来自真实认证，不能由请求构造者指定。"""
    context = await access.authenticate(
        agent_id=agent_id, binding=RunBinding.for_task_process(process_id)
    )
    return access.register_main_thread(context).access


@pytest.mark.asyncio
async def test_write_request_registers_once_with_thread_context_actor_workspace_and_process(
    operation_setup,
):
    """错误分派、重复提交或遗失线程解析都会改变登记结果；进程与线程标识来自线程表。"""
    harness, access = operation_setup
    thread_access = await _derive_thread(access)
    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="请求正文", title="请求标题")),
        access=thread_access,
    )
    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert harness.registry.size == 1
    assert stored.focus == WriteFocus(content="请求正文", title="请求标题")
    assert stored.from_actor.user_id == "test_user"
    assert stored.from_actor.agent_id == "test_agent"
    assert stored.belong_to == access.default_workspace
    assert (stored.process_id, stored.thread_id) == ("process", "main")
    assert stored.status == PendingAtomStatus.MATERIALIZING


@pytest.mark.asyncio
async def test_update_request_resolves_base_and_registers_supplied_instruction_once(
    operation_setup,
):
    """UPDATE 必须由能力层解析基础原子并保留指令与内容参数。"""
    harness, access = operation_setup
    thread_access = await _derive_thread(access)
    pending = await harness.entry.execute(
        SubmitUpdateIntentRequest("base", "更新指令", "更新正文"), access=thread_access
    )
    assert harness.registry.size == 1
    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert stored.focus == UpdateFocus(
        base_alias="base",
        base_uuid=str(harness.memories["base"].id),
        instruction="更新指令",
        content="更新正文",
    )
    assert (stored.process_id, stored.thread_id) == ("process", "main")
    assert stored.from_actor.agent_id == "test_agent"
    assert stored.belong_to == access.default_workspace


@pytest.mark.asyncio
async def test_write_and_update_dispatch_every_focus_field_with_registered_identifiers(
    operation_setup,
):
    """派发给 Patchouli 的公共意图逐字段对应登记内容，标识与话题来自登记和进程。"""
    harness, access = operation_setup
    thread_access = await _derive_thread(access)
    write = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="写入正文", title="写入标题", reason="原因")),
        access=thread_access,
    )
    update = await harness.entry.execute(
        SubmitUpdateIntentRequest("base", "更新指令", "更新正文"), access=thread_access
    )
    assert [intent for intent, _scope in harness.admission.submissions] == [
        MemoryIntent(
            kind="write",
            topic_id="test-topic",
            intent_id=write.intent_id,
            pending_alias=write.pending_alias,
            content="写入正文",
            title="写入标题",
            reason="原因",
        ),
        MemoryIntent(
            kind="update",
            topic_id="test-topic",
            intent_id=update.intent_id,
            pending_alias=update.pending_alias,
            instruction="更新指令",
            content="更新正文",
            base_alias="base",
            base_uuid=str(harness.memories["base"].id),
        ),
    ]


@pytest.mark.asyncio
async def test_reference_request_preserves_order_and_missing_items(operation_setup):
    """读取请求交付逐项结果，不因不存在的引用丢失原始顺序。"""
    harness, access = operation_setup
    thread_access = await _derive_thread(access)
    results = await harness.entry.execute(
        ResolveReferencesRequest(("missing", "base", "base")), access=thread_access
    )
    assert [(item.requested_alias, item.kind) for item in results] == [
        ("missing", "not_found"),
        ("base", "atom"),
        ("base", "atom"),
    ]
    assert results[1].atom.payload.content == "正式基础内容"
    assert harness.registry.size == 0


def _request_cases():
    """仅枚举公开请求与它们实际需要的 operation。"""
    return [
        pytest.param(
            SubmitWriteIntentRequest(WriteFocus(content="请求正文")),
            WorkspaceOperation.MEMORY_WRITE,
            id="write",
        ),
        pytest.param(
            SubmitUpdateIntentRequest("base", "更新指令"),
            WorkspaceOperation.MEMORY_UPDATE,
            id="update",
        ),
        pytest.param(
            ResolveReferencesRequest(("base",)), WorkspaceOperation.MEMORY_READ, id="read"
        ),
        pytest.param(ResolveRunTargetRequest("base"), WorkspaceOperation.MEMORY_RUN, id="run"),
        pytest.param(
            RetrieveRequest("查询基础内容"), WorkspaceOperation.MEMORY_SEARCH, id="search"
        ),
        pytest.param(
            DeriveCallThreadRequest("call_target"), WorkspaceOperation.AGENT_CALL, id="call"
        ),
    ]


@pytest.fixture
def single_operation_setup():
    """每个场景只登记一项 actor 操作，验证请求不隐含其他许可。"""
    created = []

    async def make(operation: WorkspaceOperation):
        access = make_access_composition(
            [
                make_actor_access_record(allowed_operations={operation}),
                make_actor_access_record(agent_id="call_target", allowed_operations=[]),
            ]
        )
        harness = _OperationHarness(access=access)
        _seed_base(harness)
        created.append((harness, access))
        return harness, access, await _derive_thread(access)

    yield make
    for harness, access in created:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_write_with_only_memory_write_registers_without_reading(single_operation_setup):
    """WRITE 只需 memory.write：意图带线程 context 绑定的进程与 actor，不读取任何记忆。"""
    harness, access, thread_access = await single_operation_setup(WorkspaceOperation.MEMORY_WRITE)

    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="请求正文")), access=thread_access
    )

    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert stored.focus == WriteFocus(content="请求正文")
    assert stored.status == PendingAtomStatus.MATERIALIZING
    assert stored.process_id == "process"
    assert stored.from_actor == ActorIdentity(user_id="test_user", agent_id="test_agent")
    assert stored.belong_to == access.default_workspace
    assert harness.registry.size == 1
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []


@pytest.mark.asyncio
async def test_update_with_only_memory_update_resolves_base_without_citation(
    single_operation_setup,
):
    """UPDATE 只需 memory.update：基础原子的解析不借用 READ，也不记录正式引用。"""
    harness, access, thread_access = await single_operation_setup(WorkspaceOperation.MEMORY_UPDATE)

    pending = await harness.entry.execute(
        SubmitUpdateIntentRequest("base", "更新指令"), access=thread_access
    )

    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert stored.focus == UpdateFocus(
        base_alias="base",
        base_uuid=str(harness.memories["base"].id),
        instruction="更新指令",
    )
    assert stored.status == PendingAtomStatus.MATERIALIZING
    assert stored.process_id == "process"
    assert stored.from_actor == ActorIdentity(user_id="test_user", agent_id="test_agent")
    assert harness.registry.size == 1
    assert harness.runtime.stats()["cold_reads"] == 1
    assert harness.citations == []


@pytest.mark.asyncio
async def test_read_with_only_memory_read_delivers_and_records_citation(single_operation_setup):
    """READ 只需 memory.read：交付逐项结果并为正式原子记录一次引用。"""
    harness, _access, thread_access = await single_operation_setup(WorkspaceOperation.MEMORY_READ)

    results = await harness.entry.execute(ResolveReferencesRequest(("base",)), access=thread_access)

    assert [(item.requested_alias, item.kind) for item in results] == [("base", "atom")]
    assert results[0].atom.payload.content == "正式基础内容"
    assert [item["memory_id"] for item in harness.citations] == [harness.memories["base"].id]
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 1


@pytest.mark.asyncio
async def test_run_with_only_memory_run_delivers_target_and_records_citation(
    single_operation_setup,
):
    """RUN 只需 memory.run：交付单个执行目标，引用记录与 READ 相同。"""
    harness, _access, thread_access = await single_operation_setup(WorkspaceOperation.MEMORY_RUN)

    result = await harness.entry.execute(ResolveRunTargetRequest("base"), access=thread_access)

    assert (result.requested_alias, result.kind) == ("base", "atom")
    assert result.atom.payload.content == "正式基础内容"
    assert [item["memory_id"] for item in harness.citations] == [harness.memories["base"].id]
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 1


@pytest.mark.asyncio
async def test_search_with_only_memory_search_delivers_without_citation(single_operation_setup):
    """SEARCH 只需 memory.search：检索结果预热读取视图，但检索不是引用。"""
    harness, _access, thread_access = await single_operation_setup(WorkspaceOperation.MEMORY_SEARCH)

    atoms = await harness.entry.execute(RetrieveRequest("查询基础内容"), access=thread_access)

    assert [(atom.index.alias, atom.payload.content) for atom in atoms] == [
        ("base", "正式基础内容")
    ]
    assert harness.citations == []
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 1


@pytest.mark.asyncio
async def test_call_target_with_only_agent_call_delivers_profile(single_operation_setup):
    """CALL 目标读取只需 agent.call：经 Profile 读取视图交付目标图纸，不记录记忆引用。"""
    harness, _access, thread_access = await single_operation_setup(WorkspaceOperation.AGENT_CALL)

    child = await harness.entry.derive_child_thread(
        DeriveCallThreadRequest("call_target"), access=thread_access
    )

    profile = child.descriptor.agent_profile
    assert (profile.agent_id, profile.persona, profile.model_name) == (
        "call_target",
        "CALL 目标人设",
        "call-model",
    )
    assert harness.citations == []
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 1


@pytest.mark.asyncio
async def test_child_thread_ended_through_entry_rejects_later_submissions(single_operation_setup):
    """入口的生命周期方法派生并结束子线程：结束之后以子线程提交的请求按线程已结束拒绝。"""
    harness, _access, thread_access = await single_operation_setup(WorkspaceOperation.AGENT_CALL)
    child = await harness.entry.derive_child_thread(
        DeriveCallThreadRequest("call_target"), access=thread_access
    )

    harness.entry.end_thread(access=child.access)

    with pytest.raises(ThreadEndedError):
        await harness.entry.execute(ResolveReferencesRequest(("base",)), access=child.access)
    assert harness.owner_calls == [PatchouliRoutes.GET_AGENT_PROFILE]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,operation", _request_cases())
async def test_request_without_required_operation_is_denied_before_side_effects(
    operation_request, operation
):
    """逐次授权不可被入口绕过；拒绝不得冷读、登记或撤回现有意图。"""
    access = make_access_composition(
        [make_actor_access_record(allowed_operations=set(WorkspaceOperation) - {operation})]
    )
    harness = _OperationHarness(access=access)
    _seed_base(harness)
    thread_access = await _derive_thread(access)
    prior = harness.registry.register_write(
        WriteFocus(content="保留意图"),
        belong_to=access.default_workspace,
        from_actor=ActorIdentity(user_id="test_user", agent_id="test_agent"),
        process_id="process",
        thread_id="main",
    )
    try:
        with pytest.raises(OperationDeniedError) as error:
            await _submit(harness.entry, operation_request, access=thread_access)
        assert error.value.details["operation"] == operation.value
        assert error.value.details["reason"] == "operation_not_allowed"
        assert harness.registry.size == 1
        assert (
            harness.registry.get(prior.pending_alias, access.default_workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
        assert harness.runtime.stats()["cold_reads"] == 0
        assert harness.citations == []
        assert harness.owner_calls == []
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_request,allowed_operation,denied_request,denied_operation",
    [
        pytest.param(
            ResolveReferencesRequest(("base",)),
            WorkspaceOperation.MEMORY_READ,
            ResolveRunTargetRequest("base"),
            WorkspaceOperation.MEMORY_RUN,
            id="read-not-run",
        ),
        pytest.param(
            ResolveRunTargetRequest("base"),
            WorkspaceOperation.MEMORY_RUN,
            ResolveReferencesRequest(("base",)),
            WorkspaceOperation.MEMORY_READ,
            id="run-not-read",
        ),
    ],
)
async def test_paired_operations_remain_independent_after_an_authorized_request(
    allowed_request, allowed_operation, denied_request, denied_operation
):
    """获准请求先填充读取缓存，不能使另一操作借已缓存的目标绕过权限检查。

    READ 与 RUN 共用同一读取缓存，任一入口若在授权前查缓存，只有热态才能暴露，
    因此两个方向各保留一例。WRITE 与 UPDATE 在授权前不读取任何共享状态，
    其拒绝由冷态逐请求矩阵覆盖。
    """
    access = make_access_composition(
        [make_actor_access_record(allowed_operations={allowed_operation})]
    )
    harness = _OperationHarness(access=access)
    _seed_base(harness)
    thread_access = await _derive_thread(access)
    try:
        await harness.entry.execute(allowed_request, access=thread_access)
        before_calls = list(harness.owner_calls)
        before_citations = list(harness.citations)
        before_size = harness.registry.size
        with pytest.raises(OperationDeniedError) as error:
            await harness.entry.execute(denied_request, access=thread_access)

        assert error.value.details == {
            "operation": denied_operation.value,
            "reason": "operation_not_allowed",
        }
        assert harness.owner_calls == before_calls
        assert harness.citations == before_citations
        assert harness.registry.size == before_size
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_revoked_thread_context_rejects_request_before_dispatch_without_side_effects(
    operation_setup,
):
    """线程已结束（凭据已撤销）时入口在分派前拒绝；各拒绝分支由线程表测试逐项守护。"""
    harness, access = operation_setup
    context = await _derive_thread(access)
    access.gateway.revoke_thread_context(context)
    with pytest.raises(ThreadEndedError):
        await harness.entry.execute(
            SubmitWriteIntentRequest(WriteFocus(content="请求正文")), access=context
        )
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []
    assert harness.owner_calls == []


@pytest.mark.asyncio
async def test_unknown_request_type_is_programming_error(operation_setup):
    """入口不把尚未注册的请求降级为成功或领域拒绝。"""
    harness, access = operation_setup
    thread_access = await _derive_thread(access)
    with pytest.raises(TypeError, match="Unsupported operation request: OperationRequest"):
        await harness.entry.execute(OperationRequest[None](), access=thread_access)
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0


@pytest.mark.asyncio
async def test_wrong_context_type_cannot_submit_or_revoke_thread(operation_setup):
    """非密封对象不能代替线程 context，也不能误撤已派生的线程。"""
    harness, access = operation_setup
    issued = await _derive_thread(access)
    with pytest.raises(ScopeRequiredError):
        await harness.entry.execute(
            SubmitWriteIntentRequest(WriteFocus(content="拒绝")), access=object()
        )
    with pytest.raises(ScopeRequiredError):
        access.gateway.revoke_thread_context(object())
    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="仍有效")), access=issued
    )
    assert harness.registry.size == 1
    assert (
        harness.registry.get(pending.pending_alias, access.default_workspace).focus.content
        == "仍有效"
    )


@pytest.mark.asyncio
async def test_inflight_read_finishes_after_thread_context_revocation(operation_setup):
    """只读操作已通过入口时自然完成，撤销不取消调用方或改变读取结果。"""
    harness, access = operation_setup
    thread_access = await _derive_thread(access)
    started = asyncio.Event()
    release = asyncio.Event()
    original_read = harness.backing.retrieve_by_aliases

    async def delayed_read(aliases, *, scope):
        started.set()
        await release.wait()
        return await original_read(aliases, scope=scope)

    harness.backing.retrieve_by_aliases = delayed_read
    task = asyncio.create_task(
        harness.entry.execute(ResolveReferencesRequest(("base",)), access=thread_access)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        access.gateway.revoke_thread_context(thread_access)
        assert task.cancelling() == 0
        release.set()
        (result,) = await asyncio.wait_for(task, timeout=2)
        assert result.kind == "atom"
        assert result.atom.payload.content == "正式基础内容"
        with pytest.raises(ThreadEndedError, match="revoked"):
            await harness.entry.execute(ResolveReferencesRequest(("base",)), access=thread_access)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _WriteAwaitingBeforeReturn(ActorMemoryService):
    """真实能力服务，只在 WRITE 登记后、返回入口前插入一次等待，验证接纳不随线程撤销回滚。"""

    def __init__(self, *args, started: asyncio.Event, release: asyncio.Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._started = started
        self._release = release

    async def submit_write_intent(self, request, **kwargs):
        pending = await super().submit_write_intent(request, **kwargs)
        self._started.set()
        await self._release.wait()
        return pending


@pytest.mark.asyncio
async def test_write_returning_after_revocation_preserves_admitted_intent(
    operation_setup,
):
    """WRITE 已被接纳后线程撤销，不撤回已派发的意图。"""
    harness, access = operation_setup
    started = asyncio.Event()
    release = asyncio.Event()
    entry = WorkspaceOperationEntry(
        _WriteAwaitingBeforeReturn(
            harness.bus,
            operation_authorizer=access.authorizer,
            memory_reader=harness.runtime.aliases,
            threads=access.threads,
            started=started,
            release=release,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            profile_reader=harness.runtime.profiles,
        ),
        threads=access.threads,
    )
    thread_access = await _derive_thread(access)
    task = asyncio.create_task(
        entry.execute(
            SubmitWriteIntentRequest(WriteFocus(content="迟到的写入")), access=thread_access
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        access.gateway.revoke_thread_context(thread_access)
        assert task.cancelling() == 0
        release.set()
        pending = await asyncio.wait_for(task, timeout=2)
        assert harness.registry.size == 1
        assert (
            harness.registry.get(pending.pending_alias, access.default_workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _end_threads(process, threads):
    """Actor 阶段收尾：结束进程的全部线程。"""
    threads.end_all(process)


def _stop_process(process, threads):
    """进程收到停止：线程随之结束，凭据仍待 Actor 阶段收尾撤销。"""
    process.record.request_stop()


_CROSSING_CASES = [
    pytest.param(_end_threads, id="threads-ended"),
    pytest.param(_stop_process, id="process-stopped"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", _CROSSING_CASES)
async def test_call_finishing_profile_load_after_thread_end_cannot_register_child(
    operation_setup, interrupt
):
    """Profile 冷读跨越线程结束或进程停止时重新判定存活，不签发或登记迟到子线程。"""
    harness, access = operation_setup
    thread = await _derive_thread(access)
    entered, release = asyncio.Event(), asyncio.Event()
    original = harness.backing.get_agent_profile

    async def delayed(alias, *, scope):
        entered.set()
        await release.wait()
        return await original(alias, scope=scope)

    harness.backing.get_agent_profile = delayed
    task = asyncio.create_task(
        harness.entry.derive_child_thread(DeriveCallThreadRequest("call_target"), access=thread)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        process = access.process_table.get("process")
        interrupt(process, access.threads)
        release.set()
        with pytest.raises(ThreadEndedError):
            await asyncio.wait_for(task, timeout=2)
        assert [t.thread_id for t in process.threads.snapshot()] == ["main"]
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", _CROSSING_CASES)
async def test_update_base_read_crossing_thread_end_registers_no_intent(operation_setup, interrupt):
    """UPDATE 的基础冷读跨越线程结束或进程停止时，在登记前的同步段拒绝，不产生意图。"""
    harness, access = operation_setup
    thread = await _derive_thread(access)
    entered, release = asyncio.Event(), asyncio.Event()
    original = harness.backing.retrieve_by_aliases

    async def delayed(aliases, *, scope):
        entered.set()
        await release.wait()
        return await original(aliases, scope=scope)

    harness.backing.retrieve_by_aliases = delayed
    task = asyncio.create_task(
        harness.entry.execute(SubmitUpdateIntentRequest("base", "更新指令"), access=thread)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        interrupt(access.process_table.get("process"), access.threads)
        release.set()
        with pytest.raises(ThreadEndedError):
            await asyncio.wait_for(task, timeout=2)
        assert harness.registry.size == 0
        assert harness.admission.submissions == []
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,operation", _request_cases())
async def test_every_actor_request_is_rejected_after_stop_without_side_effects(
    operation_request, operation
):
    """停止被接受后线程即已结束：六项 actor 请求都以生命周期错误拒绝，且不读取、不登记、不派生。

    即使白名单允许该操作、线程凭据尚未撤销，拒绝也发生在分派之前。
    """
    access = make_access_composition(
        [
            make_actor_access_record(allowed_operations={operation}),
            make_actor_access_record(agent_id="call_target", allowed_operations=[]),
        ]
    )
    harness = _OperationHarness(access=access)
    _seed_base(harness)
    thread_access = await _derive_thread(access)
    process = access.process_table.get("process")
    try:
        process.record.request_stop()
        with pytest.raises(ThreadEndedError):
            await _submit(harness.entry, operation_request, access=thread_access)
        assert harness.registry.size == 0
        assert harness.runtime.stats()["cold_reads"] == 0
        assert harness.owner_calls == []
        assert [t.thread_id for t in process.threads.snapshot()] == ["main"]
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
