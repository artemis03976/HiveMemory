"""TopicManagementService 读语义的单元测试。

被测对象：Patchouli Topic 公共入口。本层是授权点以下的资源 owner，阶段读取与
root 管理的授权在 workspace 能力层与任务进程阶段完成。列表、读取、结算与驱逐经
真实本地总线与 Perception 的链路见 ``tests/integration/patchouli/test_topic_access_chain.py``；
本文件只守护本层的纵深防御：下游返回越域话题时 ``get_topic_data`` 仍然隐藏，不泄漏可见性。
local bus 为记录型假总线（边界外协作者），用于构造真实存储不会返回的越域结果。
"""

from __future__ import annotations

import asyncio

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


def test_get_topic_data_hides_foreign_workspace_result_and_keeps_own():
    """下游返回越域话题时按不存在返回 None；本 Workspace 的话题照常返回。"""
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
