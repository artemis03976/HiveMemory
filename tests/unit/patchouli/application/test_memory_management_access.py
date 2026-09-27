"""MemoryManagementService access 消费的单元测试。

被测对象：application 层各用例的 access 消费：
- 管理 CRUD/GET/LIST 绑定 ``management.memory``，Agent 级 operation 调用
  同一管理入口在 application 入口失败（A1 计划第 4.1 节绑定基线）；
- Actor-visible 读取（``read_memory`` / ``retrieve`` / ``retrieve_by_aliases``）
  自 A2 起是 L2 backing：operation 检查在 workspace 能力层执行，此处只校验
  context 有效性、不重复行为检查（A2 §8 D-3）；点读强制可见性且不接受裸
  scope；
- scope 不一致被拒绝；无 access 的兼容清单方法按受信适配放行。
local bus 为记录型假总线（边界外协作者）。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from hivememory.core.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.errors import (
    OperationDeniedError,
    ScopeRequiredError,
    WorkspaceMismatchError,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

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
    """按指定 operation 构造最小许可的认证上下文。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                workspace_id=workspace.workspace_id,
                agent_id="a1",
                allowed_operations=frozenset({operation}),
            )
        ],
        default_workspace=workspace,
    )
    return await composition.authenticate(agent_id="a1", user_id="u1"), composition.guard


def _service(bus, guard):
    return MemoryManagementService(bus=bus, access_guard=guard)


def test_management_get_rejects_agent_read_operation():
    """resource.read context 调用管理 GET：application 入口拒绝（不泄漏管理语义）。"""
    bus = RecordingBus()
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    service = _service(bus, guard)

    with pytest.raises(OperationDeniedError):
        _run(service.get_memory(str(uuid4()), access=context))
    # 行为授权失败时不触达资源后端
    assert bus.calls == []


def test_management_get_accepts_management_operation_with_owner_semantics():
    """management.memory context 走管理 GET：owner-management 语义由服务固定。"""
    bus = RecordingBus(response=None)
    context, guard = _run(_context(WorkspaceOperation.MANAGEMENT_MEMORY))
    service = _service(bus, guard)

    _run(service.get_memory(str(uuid4()), access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["enforce_actor_visibility"] is False
    assert call["kwargs"]["identity_scope"] == context.identity_scope


def test_read_memory_requires_access_and_enforces_visibility():
    """Actor-visible 点读强制可见性（enforce=True）；裸 scope 一律拒绝。"""
    bus = RecordingBus(response=None)
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    service = _service(bus, guard)

    _run(service.read_memory(str(uuid4()), access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["enforce_actor_visibility"] is True
    assert call["kwargs"]["identity_scope"] == context.identity_scope

    # 兼容清单之外的新用例：无迁移路径
    with pytest.raises(ScopeRequiredError):
        _run(
            service.read_memory(
                str(uuid4()), identity_scope=make_identity_scope(user_id="u1", agent_id="a1")
            )
        )


def test_backing_read_does_not_repeat_operation_check():
    """点读 backing 不重复行为检查：operation 授权已迁至能力层（A2 §8 D-3）。

    捕获 Patchouli 侧 operation 检查未退役、与能力层形成双重检查，导致只持
    ``profile.read`` 等其他许可的能力层调用在 backing 被误拒的缺陷。资源
    可见性仍强制执行。
    """
    bus = RecordingBus(response=None)
    context, guard = _run(_context(WorkspaceOperation.MANAGEMENT_MEMORY))
    service = _service(bus, guard)

    _run(service.read_memory(str(uuid4()), access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_GET
    assert call["kwargs"]["enforce_actor_visibility"] is True


def test_backing_read_rejects_context_not_issued_by_guard():
    """backing 仍校验 context 有效性：未经网关签发的同值 context 拒绝且不触达后端。"""
    bus = RecordingBus(response=None)
    _, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    service = _service(bus, guard)
    forged = WorkspaceAccessContext(identity_scope=make_identity_scope(user_id="u1", agent_id="a1"))

    with pytest.raises(ScopeRequiredError):
        _run(service.read_memory(str(uuid4()), access=forged))
    assert bus.calls == []


def test_retrieve_rejects_request_scope_mismatching_access():
    """检索请求 scope 偏离 access 上下文即拒绝，不能据请求体重新选择 Workspace。"""
    bus = RecordingBus(response=None)
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_SEARCH))
    service = _service(bus, guard)

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


def test_retrieve_by_aliases_uses_trusted_access_scope():
    """alias 批量读取以 access 的可信 scope 请求局部路由。"""
    bus = RecordingBus(response=None)
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    service = _service(bus, guard)

    _run(service.retrieve_by_aliases(["fact_a"], access=context))

    route, call = bus.calls[0]
    assert route == PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES
    assert call["args"] == (["fact_a"], context.identity_scope)


def test_legacy_bare_scope_path_still_works_as_trusted_adapter():
    """迁移期兼容：无 access 的旧管理调用（管理 HTTP 链路）保持既有行为。"""
    bus = RecordingBus(response=None)
    context, guard = _run(_context(WorkspaceOperation.MANAGEMENT_MEMORY))
    service = _service(bus, guard)
    legacy_scope = make_identity_scope(user_id="u1", agent_id="a1")

    atom = _run(
        service.get_memory(str(uuid4()), identity_scope=legacy_scope, refresh_vitality=False)
    )
    assert atom is None
    assert bus.calls[0][1]["kwargs"]["identity_scope"] == legacy_scope
