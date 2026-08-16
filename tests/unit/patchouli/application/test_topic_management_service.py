from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import Identity, TopicData
from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class TestTopicManagementService:
    @pytest.fixture
    def bus(self):
        bus = Mock()
        bus.request = AsyncMock()
        return bus

    @pytest.mark.asyncio
    async def test_list_active_topics_uses_local_route(self, bus):
        identity = Identity(user_id="u1")
        bus.request.return_value = ["snapshot"]
        service = TopicManagementService(bus=bus)

        result = await service.list_active_topics(identity=identity)

        assert result == ("snapshot",)
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            identity=identity,
        )

    @pytest.mark.asyncio
    async def test_list_active_topics_can_include_empty_topics(self, bus):
        identity = Identity(user_id="u1")
        bus.request.return_value = ["snapshot"]
        service = TopicManagementService(bus=bus)

        result = await service.list_active_topics(
            identity=identity,
            include_empty=True,
        )

        assert result == ("snapshot",)
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            identity=identity,
            include_empty=True,
        )

    @pytest.mark.asyncio
    async def test_get_topic_data_checks_owner_without_touching_topic(self, bus):
        identity = Identity(user_id="u1")
        topic_data = TopicData(
            topic_id="t1",
            user_id="u1",
            topic_title="Gateway",
            last_update=1.0,
            last_accessed_at=2.0,
        )
        bus.request.return_value = topic_data
        service = TopicManagementService(bus=bus)

        result = await service.get_topic_data(identity=identity, topic_id="t1")

        assert result is topic_data
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_GET,
            "t1",
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
                user_id="u2",
                topic_title="Other",
                last_update=1.0,
                last_accessed_at=2.0,
            )
        bus.request.return_value = topic_data
        service = TopicManagementService(bus=bus)

        result = await service.get_topic_data(
            identity=Identity(user_id="u1"),
            topic_id="t1",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_settle_topic_uses_local_route(self, bus):
        service = TopicManagementService(bus=bus)

        await service.settle_topic(topic_id="t1")

        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            "t1",
        )
