"""
PatchouliBridge 单元测试

覆盖:
- mount_public_routes 在 GlobalSystemBus 上注册路由
- 通过 GlobalSystemBus request 能转发到 PatchouliBus 的本地 handler
- unmount 后路由不再可用
- 事件桥接: PatchouliBus publish → GlobalSystemBus subscriber 收到
- unmount 后事件桥接断开
"""

import pytest
from unittest.mock import AsyncMock

from hivememory.patchouli.contracts.domain_events import PatchouliEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime.bridge import PatchouliBridge
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class TestPatchouliBridgeRoutes:

    def setup_method(self):
        self.local_bus = PatchouliBus()
        self.global_bus = GlobalSystemBus()
        self.bridge = PatchouliBridge(
            local_bus=self.local_bus,
            global_bus=self.global_bus,
        )

    def test_mount_public_routes_registers_on_global(self):
        self.bridge.mount_public_routes()

        routes = self.global_bus.list_routes()
        assert PatchouliRoutes.PASSIVE_HANDLE_HOT in routes
        assert PatchouliRoutes.SUBMIT_INTERACTION in routes
        assert PatchouliRoutes.MEMORY_RETRIEVE in routes
        assert PatchouliRoutes.MEMORY_GET_BY_ALIAS in routes

    @pytest.mark.asyncio
    async def test_request_through_bridge_forwards_to_local(self):
        local_handler = AsyncMock(return_value="hot_result")
        self.local_bus.register("passive.analyze_and_retrieve", local_handler)
        self.bridge.mount_public_routes()

        result = await self.global_bus.request(
            PatchouliRoutes.PASSIVE_HANDLE_HOT, "event_data", user_id="u1"
        )

        assert result == "hot_result"
        local_handler.assert_awaited_once_with("event_data", user_id="u1")

    @pytest.mark.asyncio
    async def test_unmount_removes_routes(self):
        self.bridge.mount_public_routes()
        self.bridge.unmount()

        with pytest.raises(KeyError):
            await self.global_bus.request(PatchouliRoutes.PASSIVE_HANDLE_HOT)

    @pytest.mark.asyncio
    async def test_submit_interaction_forwards(self):
        local_handler = AsyncMock(return_value={"status": "ok"})
        self.local_bus.register("kernel.submit_interaction", local_handler)
        self.bridge.mount_public_routes()

        result = await self.global_bus.request(
            PatchouliRoutes.SUBMIT_INTERACTION, payload="test"
        )

        assert result == {"status": "ok"}
        local_handler.assert_awaited_once_with(payload="test")

    @pytest.mark.asyncio
    async def test_memory_retrieve_forwards(self):
        local_handler = AsyncMock(return_value={"memories": []})
        self.local_bus.register("memory.retrieve", local_handler)
        self.bridge.mount_public_routes()

        result = await self.global_bus.request(
            PatchouliRoutes.MEMORY_RETRIEVE,
            request="req",
            mode="active",
        )

        assert result == {"memories": []}
        local_handler.assert_awaited_once_with(request="req", mode="active")

    @pytest.mark.asyncio
    async def test_memory_get_by_alias_forwards(self):
        local_handler = AsyncMock(return_value={"alias": "fact_1"})
        self.local_bus.register("memory.get_memory_by_alias", local_handler)
        self.bridge.mount_public_routes()

        result = await self.global_bus.request(
            PatchouliRoutes.MEMORY_GET_BY_ALIAS,
            alias="fact_1",
            user_id="u1",
        )

        assert result == {"alias": "fact_1"}
        local_handler.assert_awaited_once_with(alias="fact_1", user_id="u1")


class TestPatchouliBridgeEvents:

    def setup_method(self):
        self.local_bus = PatchouliBus()
        self.global_bus = GlobalSystemBus()
        self.bridge = PatchouliBridge(
            local_bus=self.local_bus,
            global_bus=self.global_bus,
        )

    @pytest.mark.asyncio
    async def test_event_bridge_republishes_on_global(self):
        global_subscriber = AsyncMock()
        self.global_bus.subscribe(PatchouliEvents.MEMORY_GENERATED, global_subscriber)
        self.bridge.mount_event_bridges()

        await self.local_bus.publish(PatchouliEvents.MEMORY_GENERATED, alias="mem_1")

        global_subscriber.assert_awaited_once_with(alias="mem_1")

    @pytest.mark.asyncio
    async def test_unmount_removes_event_subscriptions(self):
        global_subscriber = AsyncMock()
        self.global_bus.subscribe(PatchouliEvents.TOPIC_EVICTED, global_subscriber)
        self.bridge.mount_event_bridges()
        self.bridge.unmount()

        await self.local_bus.publish(PatchouliEvents.TOPIC_EVICTED, topic_id="t1")

        global_subscriber.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mount_convenience_method(self):
        local_handler = AsyncMock(return_value="ok")
        self.local_bus.register("passive.analyze_and_retrieve", local_handler)
        global_subscriber = AsyncMock()
        self.global_bus.subscribe(PatchouliEvents.MEMORY_GENERATED, global_subscriber)

        self.bridge.mount()

        result = await self.global_bus.request(PatchouliRoutes.PASSIVE_HANDLE_HOT)
        assert result == "ok"

        await self.local_bus.publish(PatchouliEvents.MEMORY_GENERATED, data="x")
        global_subscriber.assert_awaited_once_with(data="x")
