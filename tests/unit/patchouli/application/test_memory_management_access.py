"""MemoryManagementService access 消费的单元测试。

被测对象：application 层各用例的 operation 绑定与迁移期兼容行为
（父计划 5.7.1 契约修订，WRX-1 冻结）：
- 管理 CRUD/GET/LIST 绑定 ``management.memory``；
- Actor-visible 点读 ``read_memory`` 绑定 ``resource.read`` 且强制可见性；
- ``retrieve`` 绑定 ``resource.search``，``retrieve_by_aliases`` 绑定
  ``resource.read``；
- 错误 grant、scope 不一致被拒绝；无 access 的旧调用按受信适配放行。
local bus 为记录型假总线（边界外协作者）。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.core.errors import OperationDeniedError, ScopeRequiredError, WorkspaceMismatchError
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus  # noqa: F401  (文档引用)
from hivememory.workspace import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    WorkspaceOperation,
)
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
OTHER = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")


class RecordingBus:
    """记录局部路由调用与 scope 参数的假总线（边界外协作者）。"""

    def __init__(self, response=None):
        self.calls: list[tuple[str, dict]] = []
        self._response = response

    async def request(self, route, *args, **kwargs):
        self.calls.append((route, {"args": args, "kwargs": kwargs}))
        return self._response


def _run(coro):
    return asyncio.run(coro)


async def _context(operation, workspace=MAIN):
    admission = LocalTrustedAdmissionService(
        {"local-process:test": list(WorkspaceOperation)},
        issued_by="test",
    )
    actor = make_identity_scope(
        user_id="u1", agent_id="a1", workspace_id=workspace.workspace_id
    ).actor_identity
    return await admission.admit(CallerPrincipal("local-process:test"), actor, workspace, operation)


def test_management_get_rejects_agent_read_grant():
    """resource.read grant 调用管理 GET：在 application 入口失败（不泄漏管理语义）。"""
    service = MemoryManagementService(bus=RecordingBus())
    context = _run(_context(WorkspaceOperation.RESOURCE_READ))

    with pytest.raises(OperationDeniedError):
        _run(service.get_memory(str(uuid4()), access=context))


def test_management_get_accepts_management_grant_and_passes_owner_semantics():
    """management.memory grant 走管理 GET：owner-management 语义由服务固定。"""
    bus = RecordingBus(response=None)
    service = MemoryManagementService(bus=bus)
    context = _run(_context(WorkspaceOperation.MANAGEMENT_MEMORY))

    _run(service.get_memory(str(uuid4()), access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["enforce_actor_visibility"] is False
    assert call["kwargs"]["identity_scope"].workspace_identity == MAIN


def test_read_memory_enforces_actor_visibility_and_requires_access():
    """Actor-visible 点读强制可见性（enforce=True），且不接受裸 scope。"""
    bus = RecordingBus(response=None)
    service = MemoryManagementService(bus=bus)
    context = _run(_context(WorkspaceOperation.RESOURCE_READ))

    _run(service.read_memory(str(uuid4()), access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["enforce_actor_visibility"] is True
    assert call["kwargs"]["identity_scope"] == context.identity_scope

    # 新用例无迁移路径：裸 scope 必须拒绝
    with pytest.raises(ScopeRequiredError):
        _run(service.read_memory(str(uuid4()), identity_scope=make_identity_scope()))


def test_retrieve_binds_resource_search_and_rejects_scope_mismatch():
    """检索绑定 resource.search；请求 scope 偏离 access 上下文即拒绝。"""
    bus = RecordingBus(response=None)
    service = MemoryManagementService(bus=bus)
    context = _run(_context(WorkspaceOperation.RESOURCE_SEARCH))

    request_other = RetrievalRequest(
        semantic_query="query",
        identity_scope=make_identity_scope(
            user_id="u1", agent_id="a1", workspace_id=OTHER.workspace_id
        ),
    )
    with pytest.raises(WorkspaceMismatchError):
        _run(service.retrieve(request_other, access=context))

    request_main = RetrievalRequest(
        semantic_query="query",
        identity_scope=make_identity_scope(
            user_id="u1", agent_id="a1", workspace_id=MAIN.workspace_id
        ),
    )
    _run(service.retrieve(request_main, access=context))
    assert bus.calls[0][0] == PatchouliLocalRoutes.MEMORY_RETRIEVE


def test_retrieve_by_aliases_binds_resource_read():
    """正式 alias 读取绑定 resource.read（与管理 GET 的操作不同）。"""
    bus = RecordingBus(response=None)
    service = MemoryManagementService(bus=bus)
    context = _run(_context(WorkspaceOperation.RESOURCE_READ))

    _run(service.retrieve_by_aliases(["fact_a"], access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES
    assert call["args"] == (["fact_a"], context.identity_scope)


def test_legacy_bare_scope_path_still_works_as_trusted_adapter():
    """迁移期兼容：无 access 的旧管理调用（管理 HTTP 链路）保持既有行为。"""
    bus = RecordingBus(response=None)
    service = MemoryManagementService(bus=bus)
    legacy_scope = make_identity_scope(user_id="u1", agent_id="a1")

    atom = _run(
        service.get_memory(str(uuid4()), identity_scope=legacy_scope, refresh_vitality=False)
    )
    assert atom is None
    assert bus.calls[0][1]["kwargs"]["identity_scope"] == legacy_scope
