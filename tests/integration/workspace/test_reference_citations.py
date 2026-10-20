"""真实操作入口、引用读取视图与全局 RPC 协作验证引用记录，仅替换库侧持久化。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.contracts.events import GlobalEvents
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
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import (
    ResolveReferencesRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.read_view import WorkspaceReadView
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_read_view,
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
    runtime: WorkspaceReadView
    entry: WorkspaceOperationEntry
    composition: AccessTestComposition
    access: WorkspaceAccessContext


@pytest_asyncio.fixture
async def chain():
    """引用走真实读取端口、缓存、意图结算与总线，不替换被测协作。"""
    bus = GlobalSystemBus()
    install_intent_admission(bus)
    backing = _MemoryBacking()
    bus.register(PatchouliRoutes.MEMORY_READ, backing.read)
    bus.register(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, backing.retrieve_by_aliases)
    bus.register(PatchouliRoutes.MEMORY_RETRIEVE, backing.retrieve)
    bus.register(PatchouliRoutes.RECORD_MEMORY_CITATION, backing.record_citation)
    runtime = make_workspace_read_view(global_bus=bus)
    runtime.subscribe(bus)
    composition = make_access_composition([make_actor_access_record()])
    access = await composition.authenticate_thread(process_id="references", register_process=True)
    memory = ActorMemoryService(
        bus,
        operation_authorizer=composition.authorizer,
        memory_reader=runtime.memory,
        threads=composition.threads,
    )
    entry = WorkspaceOperationEntry(
        memory,
        agent=ActorAgentService(
            operation_authorizer=composition.authorizer,
            profile_reader=runtime.profiles,
        ),
        threads=composition.threads,
    )
    try:
        yield _ReferenceChain(bus, backing, runtime, entry, composition, access)
    finally:
        runtime.close()
        composition.gateway.close()
        composition.gateway.revoke_all_contexts()


def _add_atom(chain, alias="canonical", *, visibility="PUBLIC"):
    """把正式原子加入持久化替身，由真实读取视图完成归属与 policy 检查。"""
    atom = MemoryAtom(
        meta=make_memory_metadata(
            user_id="test_user", source_agent_id="other_agent", visibility=visibility
        ),
        index=IndexLayer(
            alias=alias, title="引用原子", summary="引用原子摘要", memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="正式引用内容"),
    )
    chain.backing.atoms[atom.id] = atom
    return atom


async def _resolve(chain, *aliases):
    """经线程 context 绑定的公开操作请求读取，不向请求提交者暴露身份。"""
    return await chain.entry.execute(ResolveReferencesRequest(tuple(aliases)), access=chain.access)


async def _write(chain):
    return await chain.entry.execute(
        SubmitWriteIntentRequest(WriteFocus(content="待结算内容")), access=chain.access
    )


async def _settle(chain, pending, *, atom):
    """用真实事件订阅驱动结算，不直接改登记的内部状态。"""
    await chain.bus.publish(
        GlobalEvents.PENDING_ATOM_SETTLED,
        settlement=PendingAtomSettlement(
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_uuid=str(atom.id),
            canonical_alias=atom.index.alias,
        ),
    )


@pytest.mark.asyncio
async def test_reference_read_records_each_delivered_atom_once_with_authorized_scope(chain):
    """重复项仍逐项交付，引用按正式 memory id 去重，身份只能来自入口线程 context。"""
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
    chain.bus.unregister(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES)

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
async def test_pending_reference_read_does_not_record_citation(chain):
    """未交付正式内容的意图不触发生命周期引用；各终态的投影由别名解析测试守护。"""
    pending = await _write(chain)

    (result,) = await _resolve(chain, pending.pending_alias)

    assert result.kind == "pending"
    assert result.atom is None
    assert chain.backing.citations == []


@pytest.mark.asyncio
async def test_unreadable_redirect_hides_target_and_does_not_record_citation(chain):
    """结算目标对当前 Actor 不可读时不交付、脱敏坐标，也不记引用。

    冷读的缺失、归属越界与 policy 拒绝三个判断由 MemoryReader 单测逐一覆盖，
    这里取 policy 拒绝代表证明引用链经过该判断。
    """
    atom = _add_atom(chain, visibility="PRIVATE")
    pending = await _write(chain)
    await _settle(chain, pending, atom=atom)

    (result,) = await _resolve(chain, pending.pending_alias)

    assert result.kind == "redirect"
    assert result.atom is None
    assert result.canonical_alias is None
    assert result.canonical_uuid is None
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

    chain.bus.unregister(PatchouliRoutes.RECORD_MEMORY_CITATION)
    chain.bus.register(PatchouliRoutes.RECORD_MEMORY_CITATION, record_citation)

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

    chain.bus.unregister(PatchouliRoutes.RECORD_MEMORY_CITATION)
    chain.bus.register(PatchouliRoutes.RECORD_MEMORY_CITATION, record_citation)
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
