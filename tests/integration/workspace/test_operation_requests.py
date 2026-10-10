"""真实操作入口、凭据表与能力层协作，验证分派、权限和吊销语义。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import RunBinding, WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
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
from hivememory.core.mtp.exceptions import AliasNotFoundError
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import (
    CancelIntentsRequest,
    ExecutionCredential,
    ExecutionCredentialRevokedError,
    OperationRequest,
    ResolveCallTargetRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.credentials import ExecutionCredentialRegistry
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_composition, make_actor_access_record


class _OperationHarness:
    """真实入口、总线 backing 与读取视图；资源 owner 仅在公共 RPC 边界替换。"""

    def __init__(self, *, operation_authorizer) -> None:
        self.bus = GlobalSystemBus()
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
        self.credentials = ExecutionCredentialRegistry()
        for route, handler in (
            (GlobalRoutes.PATCHOULI_MEMORY_READ, self._read),
            (GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES, self._retrieve_by_aliases),
            (GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, self._retrieve),
            (GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, self._get_profile),
            (GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION, self._record_citation),
        ):
            self.bus.register(route, handler)
        self.entry = WorkspaceOperationEntry(
            ActorMemoryService(
                self.bus,
                operation_authorizer=operation_authorizer,
                memory_reader=self.runtime.aliases,
            ),
            agent=ActorAgentService(
                operation_authorizer=operation_authorizer,
                profile_reader=self.runtime.profiles,
            ),
            credential_registry=self.credentials,
            intent_registry=self.registry,
        )

    async def _read(self, memory_id, *, identity_scope):
        self.owner_calls.append(GlobalRoutes.PATCHOULI_MEMORY_READ)
        return next((atom for atom in self.memories.values() if atom.id == memory_id), None)

    async def _retrieve_by_aliases(self, aliases, *, identity_scope):
        self.owner_calls.append(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES)
        return [self.memories[alias] for alias in aliases if alias in self.memories]

    async def _retrieve(self, request):
        self.owner_calls.append(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE)
        return list(self.memories.values())[: request.top_k]

    async def _get_profile(self, agent_alias, *, identity_scope):
        self.owner_calls.append(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE)
        if agent_alias != self.profile_source.index.alias:
            raise AliasNotFoundError(
                message_key="mtp.call.profile_not_found", params={"agent_alias": agent_alias}
            )
        return ResolvedAgentProfile(
            profile=AgentProfile.from_atom(self.profile_source),
            access_policy=self.profile_source.meta.access_policy,
            source_memory_id=self.profile_source.id,
            source_version=self.profile_source.meta.version,
        )

    async def _record_citation(self, *, memory_id, source, identity_scope):
        """正式引用的副作用由 RPC 接收端记录，不能由测试伪造服务返回结果。"""
        self.owner_calls.append(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION)
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
    harness = _OperationHarness(operation_authorizer=access.authorizer)
    _seed_base(harness)
    yield harness, access
    harness.runtime.close()
    access.gateway.close()
    access.gateway.revoke_all_contexts()


async def _issue(harness, access, process_id="process", *, agent_id="test_agent"):
    """入口身份来自真实认证，不能由请求构造者指定。"""
    context = await access.authenticate(
        agent_id=agent_id, binding=RunBinding.for_task_process(process_id)
    )
    return harness.credentials.issue(
        access=context,
        target_workspace=access.default_workspace,
        process_id=process_id,
    )


@pytest.mark.asyncio
async def test_write_request_registers_once_with_credential_actor_workspace_and_process(
    operation_setup,
):
    """错误分派、重复提交或遗失凭据绑定都会改变登记结果。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="请求正文", title="请求标题")),
        credential=credential,
    )
    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert harness.registry.size == 1
    assert stored.focus == WriteFocus(content="请求正文", title="请求标题")
    assert stored.from_actor.user_id == "test_user"
    assert stored.from_actor.agent_id == "test_agent"
    assert stored.belong_to == access.default_workspace
    assert stored.process_id == "process"
    assert stored.status == PendingAtomStatus.PENDING


@pytest.mark.asyncio
async def test_update_request_resolves_base_and_registers_supplied_instruction_once(
    operation_setup,
):
    """UPDATE 必须由能力层解析基础原子并保留指令与内容参数。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    pending = await harness.entry.execute(
        SubmitUpdateIntentRequest("base", "更新指令", "更新正文"), credential=credential
    )
    assert harness.registry.size == 1
    stored = harness.registry.get(pending.pending_alias, access.default_workspace)
    assert stored.focus == UpdateFocus(
        base_alias="base",
        base_uuid=str(harness.memories["base"].id),
        instruction="更新指令",
        content="更新正文",
    )
    assert stored.process_id == "process"
    assert stored.from_actor.agent_id == "test_agent"
    assert stored.belong_to == access.default_workspace


@pytest.mark.asyncio
async def test_cancel_request_only_retracts_intents_of_credential_process(operation_setup):
    """请求无法借其他进程的 alias 撤回其意图，同步吊销也不串用凭据。"""
    harness, access = operation_setup
    first = await _issue(harness, access, "first")
    second = await _issue(harness, access, "second")
    own = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="本进程")), credential=first
    )
    other = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="其他进程")), credential=second
    )
    harness.credentials.revoke(second)
    cancelled = await harness.entry.execute(
        CancelIntentsRequest((own.pending_alias, other.pending_alias, own.pending_alias)),
        credential=first,
    )
    assert cancelled == [own.pending_alias]
    assert (
        harness.registry.get(own.pending_alias, access.default_workspace).status
        == PendingAtomStatus.CANCELLED
    )
    assert (
        harness.registry.get(other.pending_alias, access.default_workspace).status
        == PendingAtomStatus.PENDING
    )


@pytest.mark.asyncio
async def test_reference_request_preserves_order_and_missing_items(operation_setup):
    """读取请求交付逐项结果，不因不存在的引用丢失原始顺序。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    results = await harness.entry.execute(
        ResolveReferencesRequest(("missing", "base", "base")), credential=credential
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
            ResolveCallTargetRequest("call_target"), WorkspaceOperation.AGENT_CALL, id="call"
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,operation", _request_cases())
async def test_request_with_only_its_own_operation_succeeds(operation_request, operation):
    """每种请求只需对应的一项操作，不隐含 READ、WRITE 或其他 actor 许可。"""
    access = make_access_composition([make_actor_access_record(allowed_operations={operation})])
    harness = _OperationHarness(operation_authorizer=access.authorizer)
    _seed_base(harness)
    credential = await _issue(harness, access)
    try:
        result = await harness.entry.execute(operation_request, credential=credential)
        if operation in {WorkspaceOperation.MEMORY_WRITE, WorkspaceOperation.MEMORY_UPDATE}:
            stored = harness.registry.get(result.pending_alias, access.default_workspace)
            assert stored.status == PendingAtomStatus.PENDING
            assert stored.process_id == "process"
            assert stored.from_actor == ActorIdentity(user_id="test_user", agent_id="test_agent")
            assert stored.belong_to == access.default_workspace
            if operation == WorkspaceOperation.MEMORY_WRITE:
                assert stored.focus == WriteFocus(content="请求正文")
                assert harness.runtime.stats()["cold_reads"] == 0
            else:
                assert stored.focus == UpdateFocus(
                    base_alias="base",
                    base_uuid=str(harness.memories["base"].id),
                    instruction="更新指令",
                )
                assert harness.runtime.stats()["cold_reads"] == 1
            assert harness.registry.size == 1
            assert harness.citations == []
        else:
            assert harness.registry.size == 0
            assert harness.runtime.stats()["cold_reads"] == 1
            if operation == WorkspaceOperation.MEMORY_READ:
                assert [(item.requested_alias, item.kind) for item in result] == [("base", "atom")]
                assert result[0].atom.payload.content == "正式基础内容"
                assert [item["memory_id"] for item in harness.citations] == [
                    harness.memories["base"].id
                ]
            elif operation == WorkspaceOperation.MEMORY_RUN:
                assert (result.requested_alias, result.kind) == ("base", "atom")
                assert result.atom.payload.content == "正式基础内容"
                assert [item["memory_id"] for item in harness.citations] == [
                    harness.memories["base"].id
                ]
            elif operation == WorkspaceOperation.MEMORY_SEARCH:
                assert [(atom.index.alias, atom.payload.content) for atom in result] == [
                    ("base", "正式基础内容")
                ]
                assert harness.citations == []
            else:
                assert (result.agent_id, result.persona, result.model_name) == (
                    "call_target",
                    "CALL 目标人设",
                    "call-model",
                )
                assert harness.citations == []
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,operation", _request_cases())
async def test_request_without_required_operation_is_denied_before_side_effects(
    operation_request, operation
):
    """逐次授权不可被入口绕过；拒绝不得冷读、登记或撤回现有意图。"""
    access = make_access_composition(
        [make_actor_access_record(allowed_operations=set(WorkspaceOperation) - {operation})]
    )
    harness = _OperationHarness(operation_authorizer=access.authorizer)
    _seed_base(harness)
    credential = await _issue(harness, access)
    prior = harness.registry.register_write(
        WriteFocus(content="保留意图"),
        belong_to=access.default_workspace,
        from_actor=ActorIdentity(user_id="test_user", agent_id="test_agent"),
        process_id="process",
    )
    try:
        with pytest.raises(OperationDeniedError) as error:
            await harness.entry.execute(operation_request, credential=credential)
        assert error.value.details["operation"] == operation.value
        assert error.value.details["reason"] == "operation_not_allowed"
        assert harness.registry.size == 1
        assert (
            harness.registry.get(prior.pending_alias, access.default_workspace).status
            == PendingAtomStatus.PENDING
        )
        assert harness.runtime.stats()["cold_reads"] == 0
        assert harness.citations == []
        assert harness.owner_calls == []
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_request,operation", _request_cases())
async def test_system_context_is_denied_for_every_actor_request(operation_request, operation):
    """root 准入不能借执行凭据获得 actor 操作，拒绝时不触达 owner 或意图登记。"""
    access = make_access_composition([make_actor_access_record(agent_id="system")])
    harness = _OperationHarness(operation_authorizer=access.authorizer)
    _seed_base(harness)
    credential = await _issue(harness, access, agent_id="system")
    try:
        with pytest.raises(OperationDeniedError) as error:
            await harness.entry.execute(operation_request, credential=credential)

        assert error.value.details == {
            "operation": operation.value,
            "reason": "operation_not_allowed",
        }
        assert harness.owner_calls == []
        assert harness.registry.size == 0
        assert harness.runtime.stats()["cold_reads"] == 0
        assert harness.citations == []
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
        pytest.param(
            SubmitWriteIntentRequest(WriteFocus(content="独立写入")),
            WorkspaceOperation.MEMORY_WRITE,
            SubmitUpdateIntentRequest("base", "独立更新"),
            WorkspaceOperation.MEMORY_UPDATE,
            id="write-not-update",
        ),
        pytest.param(
            SubmitUpdateIntentRequest("base", "独立更新"),
            WorkspaceOperation.MEMORY_UPDATE,
            SubmitWriteIntentRequest(WriteFocus(content="独立写入")),
            WorkspaceOperation.MEMORY_WRITE,
            id="update-not-write",
        ),
    ],
)
async def test_paired_operations_remain_independent_after_an_authorized_request(
    allowed_request, allowed_operation, denied_request, denied_operation
):
    """获准请求先填充缓存或登记意图，不能使另一操作借复用状态绕过权限检查。"""
    access = make_access_composition(
        [make_actor_access_record(allowed_operations={allowed_operation})]
    )
    harness = _OperationHarness(operation_authorizer=access.authorizer)
    _seed_base(harness)
    credential = await _issue(harness, access)
    try:
        await harness.entry.execute(allowed_request, credential=credential)
        before_calls = list(harness.owner_calls)
        before_citations = list(harness.citations)
        before_size = harness.registry.size
        with pytest.raises(OperationDeniedError) as error:
            await harness.entry.execute(denied_request, credential=credential)

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
async def test_internal_cancel_without_operations_only_retracts_bound_process_pending_intents():
    """内部撤回只需有效凭据，不能撤回其他进程或已经交给物化流水线的记录。"""
    access = make_access_composition([make_actor_access_record(allowed_operations=frozenset())])
    harness = _OperationHarness(operation_authorizer=access.authorizer)
    credential = await _issue(harness, access)
    actor = ActorIdentity(user_id="test_user", agent_id="test_agent")

    def register(content, process_id):
        return harness.registry.register_write(
            WriteFocus(content=content),
            belong_to=access.default_workspace,
            from_actor=actor,
            process_id=process_id,
        )

    materializing = register("已认领", "process")
    harness.registry.claim_process("process")
    own = register("本进程 PENDING", "process")
    other = register("其他进程 PENDING", "other-process")
    try:
        cancelled = await harness.entry.execute(
            CancelIntentsRequest(
                (
                    own.pending_alias,
                    other.pending_alias,
                    materializing.pending_alias,
                    "missing",
                    own.pending_alias,
                )
            ),
            credential=credential,
        )

        assert cancelled == [own.pending_alias]
        assert (
            harness.registry.get(own.pending_alias, access.default_workspace).status
            == PendingAtomStatus.CANCELLED
        )
        assert (
            harness.registry.get(other.pending_alias, access.default_workspace).status
            == PendingAtomStatus.PENDING
        )
        assert (
            harness.registry.get(materializing.pending_alias, access.default_workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
        assert harness.registry.size == 3
        assert harness.owner_calls == []
        assert harness.citations == []
    finally:
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation_request,_operation",
    [*_request_cases(), pytest.param(CancelIntentsRequest(("pending",)), None, id="cancel")],
)
@pytest.mark.parametrize("credential_state", ["unknown", "revoked"])
async def test_unknown_or_revoked_credential_rejects_every_request_without_side_effects(
    operation_setup, operation_request, _operation, credential_state
):
    """请求种类不得影响吊销检查，失效凭据不触发读取缓存或登记。"""
    harness, access = operation_setup
    credentials = {
        "unknown": ExecutionCredential(),
        "revoked": await _issue(harness, access),
    }
    harness.credentials.revoke(credentials["revoked"])
    with pytest.raises(ExecutionCredentialRevokedError, match="unknown or revoked"):
        await harness.entry.execute(operation_request, credential=credentials[credential_state])
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []


@pytest.mark.asyncio
async def test_unknown_request_type_is_programming_error(operation_setup):
    """入口不把尚未注册的请求降级为成功或领域拒绝。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    with pytest.raises(TypeError, match="Unsupported operation request: OperationRequest"):
        await harness.entry.execute(OperationRequest[None](), credential=credential)
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0


@pytest.mark.asyncio
async def test_equal_but_unissued_credential_cannot_redeem_or_revoke_issued_object(operation_setup):
    """等值与相同哈希不能替代对象身份，也不能吊销另一个对象的绑定。"""
    harness, access = operation_setup
    issued = await _issue(harness, access)

    class EqualCredential(ExecutionCredential):
        def __hash__(self):
            return hash(issued)

        def __eq__(self, other):
            return other is issued

    impostor = EqualCredential()
    with pytest.raises(ExecutionCredentialRevokedError, match="unknown or revoked"):
        await harness.entry.execute(
            SubmitWriteIntentRequest(WriteFocus(content="不能使用等值凭据")), credential=impostor
        )
    harness.credentials.revoke(impostor)
    pending = await harness.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="原凭据仍有效")), credential=issued
    )
    assert harness.registry.size == 1
    assert (
        harness.registry.get(pending.pending_alias, access.default_workspace).focus.content
        == "原凭据仍有效"
    )


@pytest.mark.asyncio
async def test_inflight_read_finishes_after_credential_revocation(operation_setup):
    """只读操作已通过入口时自然完成，吊销不取消调用方或改变读取结果。"""
    harness, access = operation_setup
    credential = await _issue(harness, access)
    started = asyncio.Event()
    release = asyncio.Event()
    original_read = harness.backing.retrieve_by_aliases

    async def delayed_read(aliases, *, scope):
        started.set()
        await release.wait()
        return await original_read(aliases, scope=scope)

    harness.backing.retrieve_by_aliases = delayed_read
    task = asyncio.create_task(
        harness.entry.execute(ResolveReferencesRequest(("base",)), credential=credential)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        harness.credentials.revoke(credential)
        assert task.cancelling() == 0
        release.set()
        (result,) = await asyncio.wait_for(task, timeout=2)
        assert result.kind == "atom"
        assert result.atom.payload.content == "正式基础内容"
        with pytest.raises(ExecutionCredentialRevokedError, match="revoked"):
            await harness.entry.execute(ResolveReferencesRequest(("base",)), credential=credential)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _WriteAwaitingBeforeRegistration(ActorMemoryService):
    """真实能力服务，只在 WRITE 登记前插入一次等待，模拟登记前出现 await 的能力方法。"""

    def __init__(self, *args, started: asyncio.Event, release: asyncio.Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._started = started
        self._release = release

    async def submit_write_intent(self, focus, **kwargs):
        self._started.set()
        await self._release.wait()
        return await super().submit_write_intent(focus, **kwargs)


@pytest.mark.asyncio
async def test_write_registered_after_revocation_is_withdrawn_without_cancelling_caller(
    operation_setup,
):
    """WRITE 与 UPDATE 同样在返回入口后复查凭据：等待期间进程关闭，迟到登记被撤回。"""
    harness, access = operation_setup
    started = asyncio.Event()
    release = asyncio.Event()
    entry = WorkspaceOperationEntry(
        _WriteAwaitingBeforeRegistration(
            harness.bus,
            operation_authorizer=access.authorizer,
            memory_reader=harness.runtime.aliases,
            started=started,
            release=release,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            profile_reader=harness.runtime.profiles,
        ),
        credential_registry=harness.credentials,
        intent_registry=harness.registry,
    )
    credential = await _issue(harness, access)
    task = asyncio.create_task(
        entry.execute(
            SubmitWriteIntentRequest(WriteFocus(content="迟到的写入")), credential=credential
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        harness.credentials.revoke(credential)
        assert task.cancelling() == 0
        release.set()
        with pytest.raises(ExecutionCredentialRevokedError, match="revoked"):
            await asyncio.wait_for(task, timeout=2)

        # 登记确实发生过，但已不再是可被 completed 认领的 PENDING。
        assert harness.registry.size == 1
        assert harness.registry.claim_process("process") == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
