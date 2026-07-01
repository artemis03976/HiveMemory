from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import Identity
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

        assert result == ["snapshot"]
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

        assert result == ["snapshot"]
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            identity=identity,
            include_empty=True,
        )

    @pytest.mark.asyncio
    async def test_evict_topic_uses_local_route(self, bus):
        bus.request.return_value = {"success": True}
        service = TopicManagementService(bus=bus)

        result = await service.evict_topic(topic_id="t1")

        assert result == {"success": True}
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.TOPIC_EVICT, "t1")

    @pytest.mark.asyncio
    async def test_prepare_topic_uses_local_route(self, bus):
        identity = Identity(user_id="u1")
        bus.request.return_value = "real_topic"
        service = TopicManagementService(bus=bus)

        result = await service.prepare_topic(
            "NEW_TOPIC",
            "title",
            "summary",
            identity,
        )

        assert result == "real_topic"
        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_PREPARE,
            "NEW_TOPIC",
            "title",
            "summary",
            identity,
        )

    @pytest.mark.asyncio
    async def test_settle_topic_uses_local_route(self, bus):
        service = TopicManagementService(bus=bus)

        await service.settle_topic(topic_id="t1")

        bus.request.assert_awaited_once_with(
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            "t1",
        )
