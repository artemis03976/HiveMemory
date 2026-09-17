"""WorkspaceMemoryService 的单元测试。

被测对象：workspace.services.memory.WorkspaceMemoryService。保护的是
正式读取边界行为：not found / not visible / unavailable 的稳定错误映射、
snapshot 投影正确性、检索结果防御性复验，以及只接受 admission 签发的
access context。低层 store 与检索使魔是边界外协作者，用手写内存 fake。
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from hivememory.core.errors import (
    OperationDeniedError,
    ResourceNotFoundError,
    ResourceNotVisibleError,
    ResourceUnavailableError,
    ScopeRequiredError,
)
from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    MetaData,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import StorageOfflineError
from hivememory.engines.retrieval.policy import memory_is_readable
from hivememory.workspace import (
    LocalTrustedAdmissionService,
    WorkspaceMemoryService,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1")


def _atom(
    *,
    memory_id=None,
    alias="fact_alpha",
    content="alpha content",
    visibility=MemoryVisibility.PUBLIC,
    agent_id="a1",
    team_id=None,
    workspace=MAIN,
):
    if visibility == MemoryVisibility.PRIVATE:
        policy = MemoryAccessPolicy(visibility=visibility, target_agent_id=agent_id)
    elif visibility == MemoryVisibility.TEAM:
        policy = MemoryAccessPolicy(visibility=visibility, target_team_id=team_id)
    else:
        policy = MemoryAccessPolicy.public()
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=MetaData(
            workspace_identity=workspace,
            source_agent_id=agent_id,
            source_team_id=team_id,
            access_policy=policy,
            version=3,
        ),
        index=IndexLayer(
            title="alpha title",
            summary="alpha summary",
            tags=["tag-a"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


class _FakeMidTermPort:
    """内存版 MidTermStoragePort fake：与真实 adapter 相同的 policy 合同。"""

    def __init__(self, atoms):
        self._atoms = list(atoms)
        self.offline = False

    async def get(self, identity_scope, memory_id, *, enforce_actor_visibility=True):
        if self.offline:
            raise StorageOfflineError()
        return self._match(
            identity_scope,
            lambda atom: atom.id == memory_id,
            enforce=enforce_actor_visibility,
        )

    async def get_by_alias(self, identity_scope, alias, *, enforce_actor_visibility=True):
        if self.offline:
            raise StorageOfflineError()
        return self._match(
            identity_scope,
            lambda atom: atom.get_alias() == alias,
            enforce=enforce_actor_visibility,
        )

    def _match(self, identity_scope, predicate, *, enforce):
        for atom in self._atoms:
            if atom.meta.workspace_identity != identity_scope.workspace_identity:
                continue
            if not predicate(atom):
                continue
            if not memory_is_readable(
                atom,
                workspace_identity=identity_scope.workspace_identity,
                actor_identity=identity_scope.actor_identity,
                enforce_actor_visibility=enforce,
            ):
                return None
            return atom
        return None

    # 未实现的 port 方法不应被服务触达；触达即测试失败。
    async def upsert(self, memory):  # pragma: no cover - 防御
        raise AssertionError("read path must not mutate storage")


class _FakeRetrieval:
    """内存版检索使魔 fake：返回预置 atoms，可模拟离线。"""

    def __init__(self, atoms):
        self._atoms = list(atoms)
        self.offline = False

    async def retrieve(self, request):
        if self.offline:
            raise StorageOfflineError()
        from hivememory.core.protocol.models import RetrievalResponse

        response = RetrievalResponse()
        response.memories = list(self._atoms)
        return response


def _service(atoms=(), search_atoms=None):
    store = MidTermStoreStub(atoms)
    return (
        WorkspaceMemoryService(
            mid_term=store,
            retrieval=_FakeRetrieval(search_atoms if search_atoms is not None else list(atoms)),
        ),
        store,
    )


class MidTermStoreStub:
    """手写的内存版门面 stub：与 MidTermMemoryStore 相同的透传合同（真实
    门面的透传行为由 tests/unit/patchouli/memory_library/test_stores.py 保护）。"""

    def __init__(self, atoms):
        self._port = _FakeMidTermPort(atoms)

    @property
    def port(self):
        return self._port

    async def get(self, scope, memory_id, *, enforce_actor_visibility=True):
        return await self._port.get(
            scope, memory_id, enforce_actor_visibility=enforce_actor_visibility
        )

    async def get_by_alias(self, scope, alias, *, enforce_actor_visibility=True):
        return await self._port.get_by_alias(
            scope, alias, enforce_actor_visibility=enforce_actor_visibility
        )


async def _context(operation=WorkspaceOperation.RESOURCE_READ, *, agent_id="a1"):
    service = LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )
    return await service.admit(
        _principal(),
        _actor(agent_id),
        MAIN,
        operation,
    )


def _principal():
    from hivememory.workspace import CallerPrincipal

    return CallerPrincipal("local-process:test")


def _actor(agent_id="a1"):
    return make_identity_scope(user_id="u1", agent_id=agent_id).actor_identity


@pytest.mark.asyncio
async def test_read_memory_by_uuid_projects_snapshot_with_source_revision():
    """可见 atom 的 uuid 读取返回投影快照，携带 canonical 与 revision 身份。"""
    atom = _atom()
    service, _ = _service([atom])
    context = await _context()

    snapshot = await service.read_memory(context, str(atom.id))

    assert snapshot.memory_id == str(atom.id)
    assert snapshot.alias == "fact_alpha"
    assert snapshot.content == "alpha content"
    assert snapshot.visibility == "PUBLIC"
    assert snapshot.source_revision == 3


@pytest.mark.asyncio
async def test_read_memory_missing_uuid_raises_not_found():
    """不存在的 uuid 返回稳定的 not found 错误。"""
    service, _ = _service([])
    context = await _context()

    with pytest.raises(ResourceNotFoundError):
        await service.read_memory(context, str(uuid4()))


@pytest.mark.asyncio
async def test_read_memory_invisible_alias_reports_not_found_like_storage_semantics():
    """alias 读取在存储预过滤层合并不可见与缺失：统一 not found（既有语义）。"""
    atom = _atom(alias="fact_secret", visibility=MemoryVisibility.PRIVATE, agent_id="a1")
    service, _ = _service([atom])
    other_context = await _context(agent_id="a2")

    with pytest.raises(ResourceNotFoundError):
        await service.read_memory_by_alias(other_context, "fact_secret")
    with pytest.raises(ResourceNotFoundError):
        await service.read_memory_by_alias(other_context, "fact_never_exists")


@pytest.mark.asyncio
async def test_read_memory_invisible_uuid_raises_not_visible():
    """UUID 点读可区分"存在但不可见"：返回稳定 not visible。"""
    atom = _atom(alias="fact_secret", visibility=MemoryVisibility.PRIVATE, agent_id="a1")
    service, _ = _service([atom])
    other_context = await _context(agent_id="a2")

    with pytest.raises(ResourceNotVisibleError):
        await service.read_memory(other_context, str(atom.id))


@pytest.mark.asyncio
async def test_read_memory_storage_offline_raises_unavailable():
    """存储离线映射为 unavailable，不伪装成 not found。"""
    atom = _atom()
    service, store = _service([atom])
    store.port.offline = True
    context = await _context()

    with pytest.raises(ResourceUnavailableError):
        await service.read_memory(context, str(atom.id))


@pytest.mark.asyncio
async def test_search_drops_atoms_failing_defensive_policy_check():
    """检索结果的防御性复验会丢弃不属于本 Workspace 的越权返回值。"""
    own = _atom(alias="fact_own")
    foreign_workspace = make_workspace_identity(owner_user_id="u9", workspace_id="other_ws")
    smuggled = _atom(alias="fact_smuggled", workspace=foreign_workspace)
    service, _ = _service([], search_atoms=[own, smuggled])
    context = await _context(operation=WorkspaceOperation.RESOURCE_SEARCH)

    snapshots = await service.search_memory(context, "alpha", top_k=5)

    assert [s.alias for s in snapshots] == ["fact_own"]


@pytest.mark.asyncio
async def test_read_memory_rejects_bare_scope_and_wrong_operation():
    """裸 IdentityScope 拒绝；grant 与操作不匹配拒绝。"""
    service, _ = _service([])
    context = await _context(operation=WorkspaceOperation.RESOURCE_READ)

    with pytest.raises(ScopeRequiredError):
        await service.read_memory(make_identity_scope(user_id="u1"), str(uuid4()))
    # RESOURCE_READ 的 grant 不能用于检索操作
    with pytest.raises(OperationDeniedError):
        await service.search_memory(context, "query")
