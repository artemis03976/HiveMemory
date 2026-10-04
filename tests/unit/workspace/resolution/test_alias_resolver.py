"""AliasResolver 的单元测试（A2 §2.2 workspace memory read 能力）。

被测对象：L1 命中不回源且逐次授权、L2 冷读受 Workspace 代次守护（代次变化
拒绝旧值回填与成功返回）、归属纵深防御、不写负缓存、alias 批读的顺序与
去重、语义检索的协作预热，以及关闭语义。backing 以内存替身实现（被测
单元边界之外的 Patchouli），按 backing 契约只返回当前 Actor 可读的原子。

resolver 与 backing 位于授权点以下（A1 访问边界返工第 4.1 节）：只流动
授权点组装的可信 ``IdentityScope``，不接收访问 context。
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest

from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.memory_access import memory_is_readable
from hivememory.core.models import (
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceIdentity,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.workspace.cache.atom import AtomCache
from hivememory.workspace.cache.epoch import WorkspaceEpochs
from hivememory.workspace.resolution.alias import AliasResolver
from hivememory.workspace.resolution.guard import ColdReadGuard
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope

A1 = make_identity_scope(user_id="u1", agent_id="a1")
A2 = make_identity_scope(user_id="u1", agent_id="a2")
MAIN = A1.workspace_identity


def _atom(
    alias: str,
    *,
    content: str = "content",
    workspace_id: str = "main_workspace",
    private_to: str | None = None,
) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(
            source_agent_id=private_to or "a1",
            user_id="u1",
            workspace_id=workspace_id,
            visibility="PRIVATE" if private_to else "PUBLIC",
        ),
        index=IndexLayer(
            title="Resolver memory",
            summary="Memory used to verify resolver behavior.",
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


class _FakeBacking:
    """内存版 Patchouli backing：按 Workspace 存储，只返回当前 Actor 可读的原子。

    ``on_fetch`` 在每次冷读返回前执行，用于模拟读取期间发生的并发变更。
    """

    def __init__(self) -> None:
        self.atoms: dict[tuple[WorkspaceIdentity, UUID], MemoryAtom] = {}
        self.search_results: list[MemoryAtom] = []
        self.calls = 0
        self.on_fetch: Callable[[], None] | None = None

    def store(self, atom: MemoryAtom) -> MemoryAtom:
        self.atoms[(atom.workspace_identity, atom.id)] = atom
        return atom

    def remove(self, atom: MemoryAtom) -> None:
        self.atoms.pop((atom.workspace_identity, atom.id), None)

    def _visible(self, atom: MemoryAtom, scope: IdentityScope) -> bool:
        return memory_is_readable(
            atom,
            workspace_identity=scope.workspace_identity,
            actor_identity=scope.actor_identity,
        )

    def _fetched(self) -> None:
        self.calls += 1
        if self.on_fetch is not None:
            self.on_fetch()

    async def read(self, memory_id, *, scope):
        atom = self.atoms.get((scope.workspace_identity, memory_id))
        result = atom.model_copy(deep=True) if atom and self._visible(atom, scope) else None
        self._fetched()
        return result

    async def retrieve_by_aliases(self, aliases, *, scope):
        found = [
            atom.model_copy(deep=True)
            for atom in self.atoms.values()
            if atom.index.alias in aliases and self._visible(atom, scope)
        ]
        self._fetched()
        return found

    async def retrieve(self, request):
        results = [atom.model_copy(deep=True) for atom in self.search_results]
        self._fetched()
        return results

    async def get_agent_profile(self, agent_alias, *, scope):  # pragma: no cover
        raise AssertionError("alias resolver 不读取 Profile")


def _resolver(backing: _FakeBacking) -> tuple[AliasResolver, WorkspaceEpochs, ColdReadGuard]:
    epochs = WorkspaceEpochs()
    guard = ColdReadGuard(epochs, max_stale_retries=2)
    resolver = AliasResolver(cache=AtomCache(capacity=16), guard=guard, backing=backing)
    return resolver, epochs, guard


@pytest.mark.asyncio
async def test_valid_hit_is_served_without_reaching_backing():
    """冷读回填后再次读取直接命中：即使存储侧已无该原子，也不回源查询。"""
    backing = _FakeBacking()
    atom = backing.store(_atom("fact_hit", content="cached"))
    resolver, _, _ = _resolver(backing)
    await resolver.read(atom.id, scope=A1)
    backing.remove(atom)

    again = await resolver.read(atom.id, scope=A1)

    assert (again.payload.content, backing.calls) == ("cached", 1)


@pytest.mark.asyncio
async def test_shared_entry_is_authorized_per_actor_without_cold_read():
    """共享条目按原子 policy 对每个 Actor 分别授权；拒绝结果不触发冷读、不污染他人。"""
    backing = _FakeBacking()
    private = backing.store(_atom("fact_private", private_to="a2"))
    resolver, _, _ = _resolver(backing)
    await resolver.read(private.id, scope=A2)

    denied = await resolver.read(private.id, scope=A1)
    allowed = await resolver.read(private.id, scope=A2)

    assert denied is None
    assert allowed.id == private.id
    assert backing.calls == 1


@pytest.mark.asyncio
async def test_stale_cold_read_is_retried_and_only_current_value_is_cached():
    """冷读期间 Workspace 代次变化：旧值既不返回也不回填，重读取得当前值。

    捕获 epoch 守护缺失、并发变更前的旧原子晚到后覆盖缓存的缺陷。
    """
    backing = _FakeBacking()
    atom = backing.store(_atom("fact_race", content="v1"))
    resolver, epochs, _ = _resolver(backing)

    def concurrent_update() -> None:
        # 仅第一次冷读期间发生并发提交：内容推进到 v2 且失效推进代次。
        backing.on_fetch = None
        updated = atom.model_copy(deep=True)
        updated.payload.content = "v2"
        backing.store(updated)
        epochs.advance(MAIN)

    backing.on_fetch = concurrent_update
    result = await resolver.read(atom.id, scope=A1)
    backing.remove(atom)
    cached = await resolver.read(atom.id, scope=A1)

    assert (result.payload.content, cached.payload.content) == ("v2", "v2")


@pytest.mark.asyncio
async def test_persistent_concurrent_changes_fail_explicitly_without_caching():
    """代次在每次冷读期间都变化：重试耗尽后显式失败，且不留下缓存条目。"""
    backing = _FakeBacking()
    atom = backing.store(_atom("fact_busy"))
    resolver, epochs, _ = _resolver(backing)
    backing.on_fetch = lambda: epochs.advance(MAIN)

    with pytest.raises(ResourceUnavailableError) as exc_info:
        await resolver.read(atom.id, scope=A1)

    assert exc_info.value.details["reason"] == "stale_read_retry_exhausted"
    backing.on_fetch = None
    await resolver.read(atom.id, scope=A1)
    assert backing.calls == 4  # 3 次被拒的冷读 + 1 次未命中缓存的冷读


@pytest.mark.asyncio
async def test_backing_atom_from_another_workspace_is_neither_delivered_nor_cached():
    """纵深防御：backing 返回越界原子时不交付、不回填。"""
    backing = _FakeBacking()
    foreign = _atom("fact_foreign", workspace_id="other_workspace")

    async def leaky_retrieve_by_aliases(aliases, *, scope):
        backing.calls += 1
        return [foreign]

    backing.retrieve_by_aliases = leaky_retrieve_by_aliases  # type: ignore[method-assign]
    resolver, _, _ = _resolver(backing)

    first = await resolver.resolve_aliases(["fact_foreign"], scope=A1)
    second = await resolver.resolve_aliases(["fact_foreign"], scope=A1)

    assert (first, second, backing.calls) == ([], [], 2)


@pytest.mark.asyncio
async def test_missing_result_is_not_negatively_cached():
    """缺失/不可见结果不写负缓存：资源随后出现时下一次读取可取得。"""
    backing = _FakeBacking()
    resolver, _, _ = _resolver(backing)
    atom = _atom("fact_late")

    assert await resolver.read(atom.id, scope=A1) is None
    backing.store(atom)

    assert (await resolver.read(atom.id, scope=A1)).id == atom.id


@pytest.mark.asyncio
async def test_alias_batch_keeps_request_order_and_cold_reads_only_misses():
    """alias 批读按请求顺序去重交付，缓存命中与冷读结果合并，缺失项不出现。"""
    backing = _FakeBacking()
    first = backing.store(_atom("fact_a"))
    second = backing.store(_atom("fact_b"))
    third = backing.store(_atom("fact_c"))
    resolver, _, _ = _resolver(backing)
    await resolver.resolve_aliases(["fact_c"], scope=A1)
    requested: list[list[str]] = []
    original = backing.retrieve_by_aliases

    async def recording_retrieve(aliases, *, scope):
        requested.append(list(aliases))
        return await original(aliases, scope=scope)

    backing.retrieve_by_aliases = recording_retrieve  # type: ignore[method-assign]

    result = await resolver.resolve_aliases(
        ["fact_c", "fact_a", "  fact_a ", "", "fact_b", "fact_missing"],
        scope=A1,
    )

    assert [atom.id for atom in result] == [third.id, first.id, second.id]
    assert requested == [["fact_a", "fact_b", "fact_missing"]]


@pytest.mark.asyncio
async def test_search_warms_cache_for_subsequent_reads():
    """语义检索结果协作预热 AtomCache，随后按 UUID 读取直接命中。"""
    backing = _FakeBacking()
    atom = _atom("fact_found")
    backing.search_results = [atom]
    resolver, _, _ = _resolver(backing)
    request = RetrievalRequest(semantic_query="found", identity_scope=A1)

    found = await resolver.search(request, scope=A1)
    cached = await resolver.read(atom.id, scope=A1)

    assert ([item.id for item in found], cached.id, backing.calls) == ([atom.id], atom.id, 1)


@pytest.mark.asyncio
async def test_search_during_workspace_change_returns_results_without_warming():
    """检索期间代次变化：结果照常返回（检索不依赖缓存），但不预热共享缓存。"""
    backing = _FakeBacking()
    atom = _atom("fact_found")
    backing.search_results = [atom]
    resolver, epochs, _ = _resolver(backing)
    backing.on_fetch = lambda: epochs.advance(MAIN)
    request = RetrievalRequest(semantic_query="found", identity_scope=A1)

    found = await resolver.search(request, scope=A1)
    backing.on_fetch = None
    after = await resolver.read(atom.id, scope=A1)

    assert [item.id for item in found] == [atom.id]
    assert after is None  # 未预热：backing 点读存储中没有该原子


@pytest.mark.asyncio
async def test_closed_resolver_rejects_new_reads_explicitly():
    """关闭后新读显式失败，不以缓存残留或空结果伪装成功。"""
    backing = _FakeBacking()
    atom = backing.store(_atom("fact_closed"))
    resolver, _, guard = _resolver(backing)
    await resolver.read(atom.id, scope=A1)

    guard.close()

    with pytest.raises(ResourceUnavailableError) as exc_info:
        await resolver.read(atom.id, scope=A1)
    assert exc_info.value.details["reason"] == "workspace_runtime_closed"


@pytest.mark.asyncio
async def test_read_in_flight_at_close_returns_result_without_backfill():
    """关闭时在途的冷读把结果交还原调用方，但不再回填缓存。"""
    backing = _FakeBacking()
    atom = backing.store(_atom("fact_inflight"))
    epochs = WorkspaceEpochs()
    guard = ColdReadGuard(epochs)
    cache = AtomCache(capacity=16)
    resolver = AliasResolver(cache=cache, guard=guard, backing=backing)
    backing.on_fetch = guard.close

    result = await resolver.read(atom.id, scope=A1)

    assert (result.id, cache.size) == (atom.id, 0)
