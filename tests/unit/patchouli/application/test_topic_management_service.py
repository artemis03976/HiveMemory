"""
TopicManagementService 单元测试

- 薄转发契约（参数化）：list_active_topics 的 route/参数透传与 list→tuple 转换
- 契约断言：get_topic_data 读取不触发 touch 副作用
- 真实逻辑：get_topic_data 的 owner 校验与越权隐藏

注：get_topic_data → TOPIC_GET(touch=False) 的真实链路副作用（touch=False 无状态变更）
已由 tests/integration/patchouli/test_topic_access_chain.py 覆盖。
"""

from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import TopicData
from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from tests.helpers.workspace import make_identity_scope


class TestTopicManagementService:
    @pytest.fixture
    def bus(self):
        bus = Mock()
        bus.request = AsyncMock()
        return bus

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "include_empty",
        [False, True],
        ids=["exclude-empty", "include-empty"],
    )
    async def test_list_active_topics_passes_route_and_params(self, bus, include_empty):
        identity_scope = make_identity_scope(user_id="u1")
        bus.request.return_value = ["snapshot"]
        service = TopicManagementService(bus=bus)

        result = await service.list_active_topics(
            identity_scope=identity_scope,
            include_empty=include_empty,
        )

        # 契约：route 固定，include_empty 仅在 True 时注入，结果 list → tuple
        call = bus.request.await_args
        assert call.args == (PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,)
        assert call.kwargs == {"identity_scope": identity_scope, **(
            {"include_empty": True} if include_empty else {}
        )}
        assert result == ("snapshot",)

    @pytest.mark.asyncio
    async def test_get_topic_data_requests_without_touch(self, bus):
        identity_scope = make_identity_scope(user_id="u1")
        bus.request.return_value = TopicData(
            topic_id="t1",
            workspace_identity=identity_scope.workspace_identity,
            topic_title="Gateway",
            last_update=1.0,
            last_accessed_at=2.0,
        )
        service = TopicManagementService(bus=bus)

        await service.get_topic_data(identity_scope=identity_scope, topic_id="t1")

        # 契约：只读读取必须 touch=False，避免更新 last_accessed_at
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_GET,
            "t1",
            identity_scope=identity_scope,
            touch=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("topic_data", [None, "other-owner"])
    async def test_get_topic_data_hides_missing_and_unauthorized_topics(
        self,
        bus,
        topic_data,
    ):
        if topic_data == "other-owner":
            topic_data = TopicData(
                topic_id="t1",
                workspace_identity=make_identity_scope(user_id="u2").workspace_identity,
                topic_title="Other",
                last_update=1.0,
                last_accessed_at=2.0,
            )
        bus.request.return_value = topic_data
        service = TopicManagementService(bus=bus)

        result = await service.get_topic_data(
            identity_scope=make_identity_scope(user_id="u1"),
            topic_id="t1",
        )

        assert result is None
