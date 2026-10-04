"""MemoryManagementService 身份边界的单元测试。

被测对象：application 层公开用例的身份边界（A1 访问边界返工第 4.6 节）：
- 本层是授权点以下的资源 owner：公开方法不再接收 ``access`` 参数，构造
  函数不再接收 ``access_guard``；
- ``identity_scope`` 缺失或类型错误经 ``require_identity_scope`` 以
  ``ScopeRequiredError`` 拒绝，不触达资源后端；
- 资源归属校验保留：atom 的 Workspace 归属与 scope 不一致按
  ``WorkspaceMismatchError`` 拒绝；
- 管理 GET 固定 owner-management 语义（不做 Agent 可见性过滤），
  ``read_memory`` 强制 Actor 可见性；
- ``retrieve`` 只接收请求，scope 完全来自 ``RetrievalRequest.identity_scope``；
- ``retrieve_by_aliases`` 以 (aliases, scope) 请求局部路由。
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


# ---- 授权点参数不再出现在本层签名 ----


def test_constructor_rejects_access_guard():
    """构造函数不再接收 access_guard：guard 注入已随边界返工删除。"""
    with pytest.raises(TypeError, match="access_guard"):
        MemoryManagementService(bus=RecordingBus(), access_guard=object())


@pytest.mark.parametrize(
    "invoke",
    [
        lambda svc, access: svc.get_memory("11111111-1111-1111-1111-111111111111", access=access),
        lambda svc, access: svc.read_memory("11111111-1111-1111-1111-111111111111", access=access),
        lambda svc, access: svc.list_memories(access=access),
        lambda svc, access: svc.update_memory(
            "11111111-1111-1111-1111-111111111111", access=access
        ),
        lambda svc, access: svc.delete_memory(
            "11111111-1111-1111-1111-111111111111", access=access
        ),
        lambda svc, access: svc.record_feedback(
            "11111111-1111-1111-1111-111111111111", access=access, positive=True, source="test"
        ),
        lambda svc, access: svc.retrieve_by_aliases(["alias_a"], access=access),
        lambda svc, access: svc.retrieve(
            RetrievalRequest(
                semantic_query="query",
                identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
            ),
            access=access,
        ),
    ],
    ids=[
        "get_memory",
        "read_memory",
        "list_memories",
        "update_memory",
        "delete_memory",
        "record_feedback",
        "retrieve_by_aliases",
        "retrieve",
    ],
)
def test_public_methods_no_longer_accept_access_parameter(invoke):
    """公开方法不再接收 access 参数：访问 context 不进入 application 层签名。"""
    service = MemoryManagementService(bus=RecordingBus())

    with pytest.raises(TypeError, match="access"):
        invoke(service, object())


# ---- identity_scope 缺失 / 错类型经 require_identity_scope 拒绝 ----


@pytest.mark.parametrize(
    "invoke",
    [
        lambda svc: svc.get_memory("11111111-1111-1111-1111-111111111111"),
        lambda svc: svc.read_memory("11111111-1111-1111-1111-111111111111"),
        lambda svc: svc.list_memories(),
        lambda svc: svc.update_memory("11111111-1111-1111-1111-111111111111"),
        lambda svc: svc.delete_memory("11111111-1111-1111-1111-111111111111"),
        lambda svc: svc.record_feedback(
            "11111111-1111-1111-1111-111111111111", positive=True, source="test"
        ),
        lambda svc: svc.retrieve_by_aliases(["alias_a"]),
        lambda svc: svc.create_memory(atom=_make_memory_atom()),
    ],
    ids=[
        "get_memory",
        "read_memory",
        "list_memories",
        "update_memory",
        "delete_memory",
        "record_feedback",
        "retrieve_by_aliases",
        "create_memory",
    ],
)
def test_missing_identity_scope_rejected_as_scope_required(invoke):
    """identity_scope 缺失按 ScopeRequiredError 拒绝，且不触达资源后端。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)

    with pytest.raises(ScopeRequiredError, match="workspace.scope_required"):
        _run(invoke(service))
    assert bus.calls == []


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
    """归属一致的 create_memory 以 (scope, atom) 请求 MEMORY_CREATE。"""
    response = object()
    bus = RecordingBus(response=response)
    service = MemoryManagementService(bus=bus)
    atom = _make_memory_atom()
    scope = _scope()

    result = _run(service.create_memory(scope, atom))

    assert result is response
    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_CREATE
    assert call["args"] == (scope, atom)


# ---- 管理读取与 Actor 可见读取的语义区分 ----


def test_management_get_forwards_trusted_scope_without_actor_visibility_filter():
    """管理 GET 以调用方 scope 请求且固定 owner-management 语义（enforce=False）。"""
    bus = RecordingBus()
    service = MemoryManagementService(bus=bus)
    scope = _scope()

    _run(service.get_memory(str(uuid4()), identity_scope=scope, refresh_vitality=False))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["identity_scope"] == scope
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
    assert call["kwargs"]["identity_scope"] == scope
    assert call["kwargs"]["enforce_actor_visibility"] is True


# ---- 检索 backing：scope 完全来自请求 ----


def test_retrieve_forwards_request_verbatim_with_scope_frozen_inside():
    """retrieve 只接收请求并原样转发：scope 由授权点冻结在 RetrievalRequest 内。"""
    response = [object()]
    bus = RecordingBus(response=response)
    service = MemoryManagementService(bus=bus)
    request = RetrievalRequest(semantic_query="query", identity_scope=_scope())

    result = _run(service.retrieve(request))

    assert result is response
    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE
    assert call["args"] == (request,)
    assert call["kwargs"] == {}


def test_retrieve_by_aliases_forwards_aliases_and_scope():
    """alias 批量读取以 (aliases, scope) 请求 MEMORY_RETRIEVE_BY_ALIASES。"""
    bus = RecordingBus(response=[])
    service = MemoryManagementService(bus=bus)
    scope = _scope()

    result = _run(service.retrieve_by_aliases(["fact_a"], scope))

    assert result == []
    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES
    assert call["args"] == (["fact_a"], scope)
