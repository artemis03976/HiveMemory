from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import Identity, TopicData
from hivememory.patchouli.application import TopicManagementService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.runtime.bus import PatchouliBus


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
    async def test_get_topic_data_does_not_change_topic_access_state(self):
        store = ShortTermMemoryStore()
        buffer = store.create_buffer("u1", topic_title="Gateway")
        initial_accessed_at = buffer.last_accessed_at
        bus = PatchouliBus()

        async def get_topic(topic_id: str, *, touch: bool = True):
            return store.get_topic_data(topic_id, touch=touch)

        bus.register(PatchouliLocalRoutes.TOPIC_GET, get_topic)
        service = TopicManagementService(bus=bus)

        result = await service.get_topic_data(
            identity=Identity(user_id="u1"),
            topic_id=buffer.topic_id,
        )

        assert result is not None
        assert buffer.last_accessed_at == initial_accessed_at
        assert store.get_last_active_topic() is None

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
