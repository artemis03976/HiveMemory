"""TopicManagementService 访问检查与读语义的单元测试。

被测对象：Patchouli Topic 公共入口（A1 计划第 1.1/4.1 节补齐的访问差额）：
- ``list_active_topics``/``get_topic_data`` 绑定 ``resource.read``；
- ``settle_topic``/``evict_topic`` 绑定 ``management.topic``，未获准的
  读取操作不能触发结算/驱逐副作用；
- 无 access 的旧调用（Topic 管理 HTTP 链路）为兼容清单内的受信适配；
- ``get_topic_data`` 隐藏越域话题，不泄漏可见性。
local bus 为记录型假总线（边界外协作者）。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import TopicData
from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.workspace import WorkspaceOperation
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


class RecordingBus:
    """记录局部路由调用并按路由返回预置结果的假总线（边界外协作者）。"""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._responses = responses or {}

    async def request(self, route, *args, **kwargs):
        self.calls.append((route, args, kwargs))
        return self._responses.get(route)


def _run(coro):
    return asyncio.run(coro)


async def _context(operation):
    """按指定 operation 构造最小许可的认证上下文与配套守卫。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="a1",
                allowed_operations=frozenset({operation}),
            )
        ],
        default_workspace=MAIN,
    )
    context = await composition.authenticate(agent_id="a1", user_id="u1")
    return context, composition.guard


def test_list_topics_with_resource_read_returns_snapshots():
    """resource.read 列出 Topic 快照：list → tuple 转换保持既有契约。"""
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    bus = RecordingBus({PatchouliLocalRoutes.TOPIC_LIST_ACTIVE: ["snapshot"]})
    scope = context.identity_scope

    result = _run(
        TopicManagementService(bus=bus, access_guard=guard).list_active_topics(
            identity_scope=scope, access=context
        )
    )

    assert result == ("snapshot",)
    route, _, kwargs = bus.calls[0]
    assert route == PatchouliLocalRoutes.TOPIC_LIST_ACTIVE
    assert kwargs["identity_scope"] == scope


def test_get_topic_data_reads_with_resource_read_and_hides_foreign_workspace():
    """resource.read 读取话题数据；越域话题与缺失统一返回 None。"""
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    foreign = TopicData(
        topic_id="t1",
        workspace_identity=make_workspace_identity(
            owner_user_id="u2", workspace_id="main_workspace"
        ),
        topic_title="Other",
        last_update=1.0,
    )
    bus = RecordingBus({PatchouliLocalRoutes.TOPIC_GET: foreign})
    service = TopicManagementService(bus=bus, access_guard=guard)

    result = _run(
        service.get_topic_data(identity_scope=context.identity_scope, access=context, topic_id="t1")
    )
    assert result is None

    own = TopicData(
        topic_id="t1",
        workspace_identity=MAIN,
        topic_title="Own",
        last_update=1.0,
    )
    bus_own = RecordingBus({PatchouliLocalRoutes.TOPIC_GET: own})
    result_own = _run(
        TopicManagementService(bus=bus_own, access_guard=guard).get_topic_data(
            identity_scope=context.identity_scope, access=context, topic_id="t1"
        )
    )
    assert result_own is own


def test_settle_and_evict_require_management_topic_operation():
    """resource.read 不能结算/驱逐：生命周期变更绑定 management.topic。"""
    context, guard = _run(_context(WorkspaceOperation.RESOURCE_READ))
    bus = RecordingBus()
    service = TopicManagementService(bus=bus, access_guard=guard)

    with pytest.raises(OperationDeniedError):
        _run(service.settle_topic(identity_scope=context.identity_scope, access=context))
    with pytest.raises(OperationDeniedError):
        _run(
            service.evict_topic(
                identity_scope=context.identity_scope, access=context, topic_id="t1"
            )
        )
    # 行为授权失败时不产生结算/驱逐副作用
    assert bus.calls == []


def test_settle_and_evict_with_management_topic_operation_reach_local_routes():
    """management.topic 通过后按原业务链调用 local bus（路由契约保持）。"""
    context, guard = _run(_context(WorkspaceOperation.MANAGEMENT_TOPIC))
    bus = RecordingBus(
        {
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE: "settle-result",
            PatchouliLocalRoutes.TOPIC_EVICT: "evict-result",
        }
    )
    service = TopicManagementService(bus=bus, access_guard=guard)
    scope = context.identity_scope

    settle = _run(service.settle_topic(identity_scope=scope, access=context, topic_id="t_settle"))
    evict = _run(service.evict_topic(identity_scope=scope, access=context, topic_id="t_evict"))
    assert settle == "settle-result"
    assert evict == "evict-result"
    assert bus.calls[0][:2] == (PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE, (scope, "t_settle"))
    assert bus.calls[1][:2] == (PatchouliLocalRoutes.TOPIC_EVICT, (scope, "t_evict"))


def test_legacy_bare_scope_path_still_works_as_trusted_adapter():
    """迁移期兼容：无 access 的 Topic 管理调用保持既有行为。"""
    bus = RecordingBus({PatchouliLocalRoutes.TOPIC_LIST_ACTIVE: ["snapshot"]})
    legacy_scope = make_identity_scope(user_id="u1", agent_id="a1")
    service = TopicManagementService(
        bus=bus,
        access_guard=make_access_composition([make_actor_access_record(owner_user_id="u1")]).guard,
    )

    result = _run(service.list_active_topics(identity_scope=legacy_scope))
    assert result == ("snapshot",)
    assert bus.calls[0][2]["identity_scope"] == legacy_scope
