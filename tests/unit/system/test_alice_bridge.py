"""AliceBridge 单元测试"""

import pytest
from unittest.mock import AsyncMock

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.bridge import AliceBridge
from hivememory.alice.runtime.bus import AliceBus
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class TestAliceBridgeRoutes:
    def setup_method(self):
        self.local_bus = AliceBus()
        self.global_bus = GlobalSystemBus()
        self.bridge = AliceBridge(
            local_bus=self.local_bus,
            global_bus=self.global_bus,
        )

    def test_mount_public_routes_registers_on_global(self):
        self.bridge.mount_public_routes()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT in routes
        assert AliceRoutes.RUN_AGENT_STREAM in routes
        assert AliceRoutes.REGISTER_PRERETRIEVAL_ALIASES in routes
        assert AliceRoutes.GET_INTERACTION_STATE in routes

    @pytest.mark.asyncio
    async def test_request_through_bridge_forwards_to_local(self):
        local_handler = AsyncMock(return_value="agent_result")
        self.local_bus.register(AliceRoutes.RUN_AGENT, local_handler)
        self.bridge.mount_public_routes()

        result = await self.global_bus.request(
            AliceRoutes.RUN_AGENT,
            messages=[],
            identity="id",
            agent_id="a1",
            topic_id="t1",
        )

        assert result == "agent_result"
        local_handler.assert_awaited_once_with(
            messages=[],
            identity="id",
            agent_id="a1",
            topic_id="t1",
        )

    @pytest.mark.asyncio
    async def test_stream_route_through_bridge_returns_async_generator(self):
        async def stream_handler(**kwargs):
            async def _stream():
                yield {"event": "token", "data": {"content": "hi"}}
                yield {"event": "done", "data": {"final_text": "hi"}}

            return _stream()

        self.local_bus.register(AliceRoutes.RUN_AGENT_STREAM, stream_handler)
        self.bridge.mount_public_routes()

        stream = await self.global_bus.request(
            AliceRoutes.RUN_AGENT_STREAM,
            messages=[],
            identity="id",
            agent_id="a1",
            topic_id="t1",
        )

        events = []
        async for event in stream:
            events.append(event)

        assert [event["event"] for event in events] == ["token", "done"]

    @pytest.mark.asyncio
    async def test_unmount_removes_routes(self):
        self.bridge.mount_public_routes()
        self.bridge.unmount()

        with pytest.raises(KeyError):
            await self.global_bus.request(AliceRoutes.RUN_AGENT)
