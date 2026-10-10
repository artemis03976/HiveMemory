"""真实操作入口、引用读取视图与全局 RPC 协作验证引用记录，仅替换库侧持久化。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    WriteFocus,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import (
    ExecutionCredential,
    ExecutionCredentialRevokedError,
    ResolveReferencesRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.credentials import ExecutionCredentialRegistry
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_runtime,
)


class _MemoryBacking:
    """替代库侧存储，保留完整原子与引用 RPC 的可观察结果。"""

    def __init__(self) -> None:
        self.atoms: dict[UUID, MemoryAtom] = {}
        self.citations: list[tuple[UUID, str, IdentityScope]] = []

    async def read(self, memory_id: UUID, *, identity_scope: IdentityScope) -> MemoryAtom | None:
        atom = self.atoms.get(memory_id)
        return atom.model_copy(deep=True) if atom else None

    async def retrieve_by_aliases(
        self, aliases: list[str], *, identity_scope: IdentityScope
    ) -> list[MemoryAtom]:
        return [
            atom.model_copy(deep=True)
            for atom in self.atoms.values()
            if atom.index.alias in aliases
        ]

    async def retrieve(self, request: RetrievalRequest) -> list[MemoryAtom]:
        return [atom.model_copy(deep=True) for atom in self.atoms.values()][: request.top_k]

    async def record_citation(
        self, *, memory_id: UUID, source: str, identity_scope: IdentityScope
    ) -> None:
        # 捕获跨边界 RPC 的公开契约；scope 不写入正式原子。
        self.citations.append((memory_id, source, identity_scope))
        self.atoms[memory_id].meta.lifecycle.access_count += 1


@dataclass
class _ReferenceChain:
    """每个用例独立的真实 workspace 协作边界。"""

    bus: GlobalSystemBus
    backing: _MemoryBacking
    runtime: WorkspaceRuntime
    memory: MemoryApplicationService
    entry: WorkspaceOperationEntry
    credentials: ExecutionCredentialRegistry
    composition: AccessTestComposition
    access: WorkspaceAccessContext
    credential: ExecutionCredential


@pytest_asyncio.fixture
async def chain():
    """引用走真实读取端口、缓存、意图结算与总线，不替换被测协作。"""
    bus = GlobalSystemBus()
    backing = _MemoryBacking()
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_READ, backing.read)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES, backing.retrieve_by_aliases)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, backing.retrieve)
    bus.register(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION, backing.record_citation)
    runtime = make_workspace_runtime(global_bus=bus)
    runtime.subscribe(bus)
    composition = make_access_composition([make_actor_access_record()])
    access = await composition.authenticate()
    credentials = ExecutionCredentialRegistry()
    credential = credentials.issue(
        access=access, target_workspace=composition.default_workspace, process_id="references"
    )
    memory = MemoryApplicationService(
        bus, operation_authorizer=composition.authorizer, memory_reader=runtime.aliases
    )
    entry = WorkspaceOperationEntry(
        memory,
        agent=AgentApplicationService(
            bus, operation_authorizer=composition.authorizer, profile_reader=runtime.profiles
        ),
        credential_registry=credentials,
        intent_registry=runtime.intents,
    )
    try:
        yield _ReferenceChain(
            bus, backing, runtime, memory, entry, credentials, composition, access, credential
        )
    finally:
        credentials.revoke(credential)
        runtime.close()
        composition.gateway.close()
        composition.gateway.revoke_all_contexts()


def _add_atom(chain, alias="canonical", *, visibility="PUBLIC", user_id="test_user"):
    """把正式原子加入持久化替身，由真实读取视图完成归属与 policy 检查。"""
    atom = MemoryAtom(
        meta=make_memory_metadata(
            user_id=user_id, source_agent_id="other_agent", visibility=visibility
        ),
        index=IndexLayer(
            alias=alias, title="引用原子", summary="引用原子摘要", memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="正式引用内容"),
    )
    chain.backing.atoms[atom.id] = atom
    return atom


async def _resolve(chain, *aliases):
    """经凭据绑定的公开操作请求读取，不向请求提交者暴露身份。"""
    return await chain.entry.execute(
        ResolveReferencesRequest(tuple(aliases)), credential=chain.credential
    )


async def _write(chain):
    return await chain.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="待结算内容")), credential=chain.credential
    )


async def _settle(chain, pending, *, atom=None, discarded=False):
    """用真实认领与事件订阅驱动终态，不直接改登记的内部状态。"""
    chain.runtime.intents.claim_process("references")
    await chain.bus.publish(
        GlobalEvents.PENDING_ATOM_SETTLED,
        settlement=PendingAtomSettlement(
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
            resolution=(
                PendingAtomResolution.DISCARDED if discarded else PendingAtomResolution.CREATED
            ),
            canonical_uuid=str(atom.id) if atom else None,
            canonical_alias=atom.index.alias if atom else None,
        ),
    )


@pytest.mark.asyncio
async def test_reference_read_records_each_delivered_atom_once_with_authorized_scope(chain):
    """重复项仍逐项交付，引用按正式 memory id 去重，身份只能来自入口凭据。"""
    first = _add_atom(chain, "first")
    second = _add_atom(chain, "second")

    results = await _resolve(chain, "first", "missing", " first ", "second", "first")

    assert [(item.requested_alias, item.kind) for item in results] == [
        ("first", "atom"),
        ("missing", "not_found"),
        (" first ", "atom"),
        ("second", "atom"),
        ("first", "atom"),
    ]
    assert [item.atom.payload.content for item in results if item.atom] == ["正式引用内容"] * 4
    assert [(memory_id, source) for memory_id, source, _ in chain.backing.citations] == [
        (first.id, "workspace.reference_read"),
        (second.id, "workspace.reference_read"),
    ]
    assert [scope.actor_identity.agent_id for _, _, scope in chain.backing.citations] == [
        "test_agent",
        "test_agent",
    ]
    assert [scope.workspace_identity for _, _, scope in chain.backing.citations] == [
        chain.composition.default_workspace,
        chain.composition.default_workspace,
    ]
    assert first.meta.lifecycle.access_count == 1
    assert second.meta.lifecycle.access_count == 1


@pytest.mark.asyncio
async def test_reference_read_records_cache_hits_in_each_request(chain):
    """L1 命中可脱离冷读路由交付，每次引用读取仍独立记录一次。"""
    atom = _add_atom(chain)
    await _resolve(chain, "canonical")
    chain.bus.unregister(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES)

    (result,) = await _resolve(chain, "canonical")

    assert result.kind == "atom"
    assert result.atom.payload.content == "正式引用内容"
    assert atom.meta.lifecycle.access_count == 2
    assert [(memory_id, source) for memory_id, source, _ in chain.backing.citations] == [
        (atom.id, "workspace.reference_read"),
        (atom.id, "workspace.reference_read"),
    ]


@pytest.mark.asyncio
async def test_settled_redirect_and_canonical_alias_share_one_citation(chain):
    """意图 redirect 与正式 alias 指向同一原子时，批次内只引用一次。"""
    atom = _add_atom(chain)
    pending = await _write(chain)
    await _settle(chain, pending, atom=atom)

    results = await _resolve(chain, pending.pending_alias, "canonical", pending.pending_alias)

    assert [result.kind for result in results] == ["redirect", "atom", "redirect"]
    assert [result.atom.id for result in results] == [atom.id, atom.id, atom.id]
    assert atom.meta.lifecycle.access_count == 1
    assert [(memory_id, source) for memory_id, source, _ in chain.backing.citations] == [
        (atom.id, "workspace.reference_read")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "materializing", "failed", "discarded", "cancelled"])
async def test_reference_read_without_delivered_canonical_atom_does_not_record_citation(
    chain, state
):
    """各意图终态只交付自身状态，未交付正式内容不触发生命周期引用。"""
    pending = await _write(chain)
    if state == "materializing":
        chain.runtime.intents.claim_process("references")
    elif state == "failed":
        chain.runtime.intents.claim_process("references")
        await chain.bus.publish(
            GlobalEvents.PENDING_ATOM_FAILED,
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
        )
    elif state == "discarded":
        await _settle(chain, pending, discarded=True)
    elif state == "cancelled":
        chain.runtime.intents.cancel_process("references")

    (result,) = await _resolve(chain, pending.pending_alias)

    expected = {"materializing": "pending", "cancelled": "not_found"}.get(state, state)
    assert result.kind == expected
    assert result.atom is None
    assert chain.backing.citations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["missing", "private", "other_workspace"])
async def test_unreadable_redirect_hides_target_and_does_not_record_citation(chain, target):
    """缺失、policy 拒绝与归属越界的结算目标均不交付，也不记引用。"""
    atom = _add_atom(
        chain,
        visibility="PRIVATE" if target == "private" else "PUBLIC",
        user_id="other_user" if target == "other_workspace" else "test_user",
    )
    if target == "missing":
        chain.backing.atoms.pop(atom.id)
    pending = await _write(chain)
    await _settle(chain, pending, atom=atom)

    (result,) = await _resolve(chain, pending.pending_alias)

    assert result.kind == "redirect"
    assert result.atom is None
    assert result.canonical_alias is None
    assert result.canonical_uuid is None
    assert chain.backing.citations == []


@pytest.mark.asyncio
async def test_update_base_resolution_does_not_record_citation(chain):
    """UPDATE 的基础读取仅用于登记资格，不是引用内容的交付。"""
    atom = _add_atom(chain)

    pending = await chain.entry.execute(
        SubmitUpdateIntentRequest("canonical", "更新指令"), credential=chain.credential
    )

    assert pending.focus.base_uuid == str(atom.id)
    assert (
        chain.runtime.intents.get(
            pending.pending_alias, chain.composition.default_workspace
        ).focus.instruction
        == "更新指令"
    )
    assert atom.meta.lifecycle.access_count == 0
    assert chain.backing.citations == []


@pytest.mark.asyncio
async def test_other_read_and_search_methods_do_not_record_citation(chain):
    """引用记录只属于 resolve_references，不扩散到管理外的普通点读与检索。"""
    atom = _add_atom(chain)
    kwargs = {
        "target_workspace": chain.composition.default_workspace,
        "access": chain.access,
    }

    point = await chain.memory.read(atom.id, **kwargs)
    aliases = await chain.memory.retrieve_by_aliases(["canonical"], **kwargs)
    search = await chain.memory.retrieve(semantic_query="正式内容", **kwargs)

    assert point.payload.content == "正式引用内容"
    assert [item.id for item in aliases] == [atom.id]
    assert [item.id for item in search] == [atom.id]
    assert atom.meta.lifecycle.access_count == 0
    assert chain.backing.citations == []


@pytest.mark.asyncio
async def test_citation_failure_logs_and_preserves_all_read_results(chain, caplog):
    """一次引用失败不改变交付或阻断后续原子的引用记录。"""
    first = _add_atom(chain, "first")
    second = _add_atom(chain, "second")

    async def record_citation(*, memory_id: UUID, source: str, identity_scope: IdentityScope):
        if memory_id == first.id:
            raise RuntimeError("生命周期存储暂时不可达")
        await chain.backing.record_citation(
            memory_id=memory_id, source=source, identity_scope=identity_scope
        )

    chain.bus.unregister(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION)
    chain.bus.register(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION, record_citation)

    results = await _resolve(chain, "first", "first", "second")

    assert [(result.kind, result.atom.payload.content) for result in results] == [
        ("atom", "正式引用内容"),
        ("atom", "正式引用内容"),
        ("atom", "正式引用内容"),
    ]
    assert first.meta.lifecycle.access_count == 0
    assert second.meta.lifecycle.access_count == 1
    assert [(memory_id, source) for memory_id, source, _ in chain.backing.citations] == [
        (second.id, "workspace.reference_read")
    ]
    assert "Failed to record memory citation" in caplog.text
    assert str(first.id) in caplog.text
    assert (
        sum("Failed to record memory citation" in record.message for record in caplog.records) == 1
    )


@pytest.mark.asyncio
async def test_cancellation_during_citation_propagates_to_reference_caller(chain):
    """引用调用挂起时取消调用方，应传播 CancelledError，不能返回伪成功。"""
    atom = _add_atom(chain)
    started = asyncio.Event()
    release = asyncio.Event()

    async def record_citation(*, memory_id: UUID, source: str, identity_scope: IdentityScope):
        started.set()
        await release.wait()
        await chain.backing.record_citation(
            memory_id=memory_id, source=source, identity_scope=identity_scope
        )

    chain.bus.unregister(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION)
    chain.bus.register(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION, record_citation)
    task = asyncio.create_task(_resolve(chain, "canonical"))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert atom.meta.lifecycle.access_count == 0
        assert chain.backing.citations == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["denied", "revoked", "unknown"])
async def test_rejected_reference_request_cannot_record_citation_even_after_cache_warmup(
    chain, rejection
):
    """热缓存不能绕过凭据或 resource.read，拒绝时无额外引用。"""
    atom = _add_atom(chain)
    await _resolve(chain, "canonical")
    credential = chain.credential
    entry = chain.entry
    composition = None
    error = ExecutionCredentialRevokedError
    if rejection == "revoked":
        chain.credentials.revoke(credential)
    elif rejection == "unknown":
        credential = ExecutionCredential()
    else:
        composition = make_access_composition(
            [
                make_actor_access_record(
                    allowed_operations=set(WorkspaceOperation) - {WorkspaceOperation.RESOURCE_READ}
                )
            ]
        )
        access = await composition.authenticate()
        credential = chain.credentials.issue(
            access=access, target_workspace=composition.default_workspace, process_id="denied"
        )
        entry = WorkspaceOperationEntry(
            MemoryApplicationService(
                chain.bus,
                operation_authorizer=composition.authorizer,
                memory_reader=chain.runtime.aliases,
            ),
            agent=AgentApplicationService(
                chain.bus,
                operation_authorizer=composition.authorizer,
                profile_reader=chain.runtime.profiles,
            ),
            credential_registry=chain.credentials,
            intent_registry=chain.runtime.intents,
        )
        error = OperationDeniedError
    try:
        with pytest.raises(error):
            await entry.execute(ResolveReferencesRequest(("canonical",)), credential=credential)
        assert atom.meta.lifecycle.access_count == 1
        assert len(chain.backing.citations) == 1
    finally:
        chain.credentials.revoke(credential)
        if composition:
            composition.gateway.close()
            composition.gateway.revoke_all_contexts()
