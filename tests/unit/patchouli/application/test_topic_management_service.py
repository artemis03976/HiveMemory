"""TopicManagementService 身份边界与读语义的单元测试。

被测对象：Patchouli Topic 公共入口（A1 访问边界返工第 4.6 节）：
- 本层是授权点以下的资源 owner：公开方法不接收 ``access`` 参数，构造函数
  不接收 ``access_guard``；``resource.read`` / ``management.topic`` 的行为
  授权在 workspace 能力层与任务进程阶段检查完成；
- ``identity_scope`` 缺失由必填签名拒绝，不触达资源后端；
- ``get_topic_data`` 隐藏越域话题，不泄漏可见性；
- settle/evict 在公共边界拆出归属，再传给内部路由。
local bus 为记录型假总线（边界外协作者）。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.models import IdentityScope, TopicData
from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.workspace import make_identity_scope, make_workspace_identity

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


def _scope(workspace_id: str = MAIN.workspace_id) -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="a1", workspace_id=workspace_id)


def test_list_active_topics_forwards_scope_to_local_route():
    """列表以 scope 的归属请求 TOPIC_LIST_ACTIVE，并保持 list → tuple 转换。"""
    bus = RecordingBus({PatchouliLocalRoutes.TOPIC_LIST_ACTIVE: ["snapshot"]})
    scope = _scope()

    result = _run(TopicManagementService(bus=bus).list_active_topics(identity_scope=scope))

    assert result == ("snapshot",)
    route, _, kwargs = bus.calls[0]
    assert route == PatchouliLocalRoutes.TOPIC_LIST_ACTIVE
    assert kwargs["belong_to"] == scope.workspace_identity


def test_list_active_topics_forwards_include_empty_flag():
    """include_empty=True 原样传入 TOPIC_LIST_ACTIVE（池快照含空话题）。"""
    bus = RecordingBus({PatchouliLocalRoutes.TOPIC_LIST_ACTIVE: []})

    result = _run(
        TopicManagementService(bus=bus).list_active_topics(
            identity_scope=_scope(), include_empty=True
        )
    )

    assert result == ()
    _, _, kwargs = bus.calls[0]
    assert kwargs["include_empty"] is True


def test_get_topic_data_reads_with_scope_and_hides_foreign_workspace():
    """越域话题与缺失统一返回 None；自己的话题照常返回，不泄漏可见性。"""
    foreign = TopicData(
        topic_id="t1",
        workspace_identity=make_workspace_identity(
            owner_user_id="u2", workspace_id="main_workspace"
        ),
        topic_title="Other",
        last_update=1.0,
    )
    bus_foreign = RecordingBus({PatchouliLocalRoutes.TOPIC_GET: foreign})
    service = TopicManagementService(bus=bus_foreign)
    scope = _scope()

    assert _run(service.get_topic_data(identity_scope=scope, topic_id="t1")) is None

    bus_missing = RecordingBus()
    assert (
        _run(
            TopicManagementService(bus=bus_missing).get_topic_data(
                identity_scope=scope, topic_id="t1"
            )
        )
        is None
    )
    assert bus_missing.calls[0][0] == PatchouliLocalRoutes.TOPIC_GET

    own = TopicData(
        topic_id="t1",
        workspace_identity=MAIN,
        topic_title="Own",
        last_update=1.0,
    )
    bus_own = RecordingBus({PatchouliLocalRoutes.TOPIC_GET: own})
    result_own = _run(
        TopicManagementService(bus=bus_own).get_topic_data(identity_scope=scope, topic_id="t1")
    )
    assert result_own is own


def test_settle_and_evict_forward_scope_to_local_routes():
    """settle/evict 以 (belong_to, topic_id) 请求既有路由，返回业务结果。"""
    bus = RecordingBus(
        {
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE: "settle-result",
            PatchouliLocalRoutes.TOPIC_EVICT: "evict-result",
        }
    )
    service = TopicManagementService(bus=bus)
    scope = _scope()

    settle = _run(service.settle_topic(identity_scope=scope, topic_id="t_settle"))
    evict = _run(service.evict_topic(identity_scope=scope, topic_id="t_evict"))
    assert settle == "settle-result"
    assert evict == "evict-result"
    assert bus.calls[0][:2] == (
        PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
        (scope.workspace_identity, "t_settle"),
    )
    assert bus.calls[1][:2] == (
        PatchouliLocalRoutes.TOPIC_EVICT,
        (scope.workspace_identity, "t_evict"),
    )


def test_constructor_rejects_access_guard_and_methods_reject_access_parameter():
    """授权点参数不再出现在本层签名：构造与公开方法均不接受 access。"""
    with pytest.raises(TypeError, match="access_guard"):
        TopicManagementService(bus=RecordingBus(), access_guard=object())

    service = TopicManagementService(bus=RecordingBus())
    scope = _scope()
    with pytest.raises(TypeError, match="access"):
        _run(service.list_active_topics(identity_scope=scope, access=object()))
    with pytest.raises(TypeError, match="access"):
        _run(service.settle_topic(identity_scope=scope, access=object()))


@pytest.mark.parametrize(
    "invoke",
    [
        lambda svc: svc.list_active_topics(),
        lambda svc: svc.get_topic_data(topic_id="t1"),
        lambda svc: svc.settle_topic(),
        lambda svc: svc.evict_topic(topic_id="t1"),
    ],
    ids=["list_active_topics", "get_topic_data", "settle_topic", "evict_topic"],
)
def test_missing_identity_scope_rejected_as_required_argument(invoke):
    """identity_scope 缺失由必填签名拒绝，且不触达资源后端。"""
    bus = RecordingBus()
    service = TopicManagementService(bus=bus)

    with pytest.raises(TypeError, match="identity_scope"):
        _run(invoke(service))
    assert bus.calls == []
