"""MemoryManagementService 身份边界的单元测试。

被测对象：application 层公开用例的身份边界。本层是授权点以下的资源 owner：
- ``identity_scope`` 类型错误经 ``require_identity_scope`` 拒绝，不触达资源后端；
- 资源归属校验保留：atom 的 Workspace 归属与 scope 不一致按
  ``WorkspaceMismatchError`` 拒绝；
- 管理 GET 固定 owner-management 语义（不做 Agent 可见性过滤），
  ``read_memory`` 强制 Actor 可见性；
- ``retrieve`` 只接收请求，scope 完全来自 ``RetrievalRequest.identity_scope``；
- ``retrieve_by_aliases`` 拆出归属与发起者，交给局部路由执行资源授权。
local bus 为记录型假总线（边界外协作者）。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.core.errors import (
    ScopeRequiredError,
    WorkspaceMismatchError,
)
from hivememory.core.models import IdentityScope, IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope, make_workspace_identity


class RecordingBus:
    """记录局部路由调用与参数的假总线（边界外协作者）。"""

    def __init__(self, response=None):
        self.calls: list[tuple[str, dict]] = []
        self._response = response

    async def request(self, route, *args, **kwargs):
        self.calls.append((route, {"args": args, "kwargs": kwargs}))
        return self._response


def _run(coro):
    return asyncio.run(coro)


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


def _scope(workspace_id: str = "main_workspace") -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="a1", workspace_id=workspace_id)


# ---- require_identity_scope 拒绝错误类型 ----


def test_identity_scope_wrong_type_rejected_as_scope_required():
    """identity_scope 类型错误（如传入 WorkspaceIdentity）按 ScopeRequiredError 拒绝。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        _run(
            service.read_memory(
                str(uuid4()),
                identity_scope=make_workspace_identity(owner_user_id="u1"),
            )
        )
    assert bus.calls == []


# ---- 资源归属校验保留 ----


def test_create_memory_rejects_atom_from_foreign_workspace():
    """atom 归属 Workspace ≠ scope.workspace：WorkspaceMismatchError 且不触达后端。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)
    foreign_atom = _make_memory_atom(user_id="u2")

    with pytest.raises(WorkspaceMismatchError, match="workspace.mismatch"):
        _run(service.create_memory(_scope(), foreign_atom))
    assert bus.calls == []


def test_create_memory_forwards_scope_and_atom_to_memory_create():
    """归属一致的 create_memory 以 (belong_to, atom) 请求 MEMORY_CREATE。"""
    response = object()
    bus = RecordingBus(response=response)
    service = MemoryManagementService(bus=bus)
    atom = _make_memory_atom()
    scope = _scope()

    result = _run(service.create_memory(scope, atom))

    assert result is response
    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_CREATE
    assert call["args"] == (scope.workspace_identity, atom)


# ---- 管理读取与 Actor 可见读取的语义区分 ----


def test_management_get_forwards_trusted_scope_without_actor_visibility_filter():
    """管理 GET 以调用方 scope 请求且固定 owner-management 语义（enforce=False）。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)
    scope = _scope()

    _run(service.get_memory(str(uuid4()), identity_scope=scope, refresh_vitality=False))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["belong_to"] == scope.workspace_identity
    assert call["kwargs"]["from_actor"] == scope.actor_identity
    # owner-management 语义（D4）：管理读取不做 Agent 可见性过滤
    assert call["kwargs"]["enforce_actor_visibility"] is False


def test_read_memory_enforces_actor_visibility():
    """Actor 可见点读强制可见性（enforce=True）：可见性由本层按 policy 强制。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)
    scope = _scope()

    _run(service.read_memory(str(uuid4()), identity_scope=scope, refresh_vitality=False))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["belong_to"] == scope.workspace_identity
    assert call["kwargs"]["from_actor"] == scope.actor_identity
    assert call["kwargs"]["enforce_actor_visibility"] is True


# ---- 检索 backing：scope 完全来自请求 ----


def test_retrieve_splits_public_scope_into_internal_query():
    """公开请求在边界转换；本地检索只接收归属与可见性主体。"""
    response = [object()]
    bus = RecordingBus(response=response)
    service = MemoryManagementService(bus=bus)
    request = RetrievalRequest(semantic_query="query", identity_scope=_scope())

    result = _run(service.retrieve(request))

    assert result is response
    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE
    query = call["args"][0]
    assert query.semantic_query == "query"
    assert query.belong_to == request.identity_scope.workspace_identity
    assert query.from_actor == request.identity_scope.actor_identity
    # 请求未给出 top_k：原样缺省转发，条数由检索引擎配置决定
    assert call["kwargs"] == {"top_k": None}


def test_retrieve_by_aliases_forwards_aliases_and_scope():
    """alias 批量读取按归属定位资源，并独立传入可见性主体。"""
    bus = RecordingBus(response=[])
    service = MemoryManagementService(bus=bus)
    scope = _scope()

    result = _run(service.retrieve_by_aliases(["fact_a"], scope))

    assert result == []
    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES
    assert call["args"] == (["fact_a"], scope.workspace_identity)
    assert call["kwargs"] == {"from_actor": scope.actor_identity}
