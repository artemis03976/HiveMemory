"""
PatchouliBus 单元测试

PatchouliBus 继承自 AsyncSystemBus，测试覆盖:
- register/unregister: 路由注册与注销
- request: RPC 调用
- subscribe/unsubscribe: 事件订阅
- publish: 事件发布
- list_routes / list_events: 查询接口
"""

import pytest
from unittest.mock import AsyncMock, Mock

from hivememory.patchouli.runtime.bus import PatchouliBus


class TestPatchouliBusRegistration:
    """路由注册与注销测试"""

    def test_register_adds_handler(self):
        bus = PatchouliBus()
        handler = AsyncMock()

        bus.register("test.route", handler)

        assert "test.route" in bus.list_routes()

    def test_register_overwrites_existing_handler(self):
        bus = PatchouliBus()
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        bus.register("test.route", handler1)
        bus.register("test.route", handler2)

        assert bus._handlers["test.route"] is handler2

    def test_unregister_removes_handler(self):
        bus = PatchouliBus()
        handler = AsyncMock()
        bus.register("test.route", handler)

        bus.unregister("test.route")

        assert "test.route" not in bus.list_routes()

    def test_unregister_nonexistent_is_noop(self):
        bus = PatchouliBus()

        bus.unregister("missing.route")

        assert bus.list_routes() == []


class TestPatchouliBusRequest:
    """RPC request 测试"""

    @pytest.mark.asyncio
    async def test_request_calls_registered_handler(self):
        bus = PatchouliBus()
        handler = AsyncMock(return_value="result")

        bus.register("test.route", handler)
        result = await bus.request("test.route", "arg1", key="value")

        assert result == "result"
        handler.assert_awaited_once_with("arg1", key="value")

    @pytest.mark.asyncio
    async def test_request_raises_key_error_for_missing_route(self):
        bus = PatchouliBus()

        with pytest.raises(KeyError, match="test.missing"):
            await bus.request("test.missing")

    @pytest.mark.asyncio
    async def test_request_returns_non_awaitable_result(self):
        bus = PatchouliBus()

        def sync_handler(x):
            return x * 2

        bus.register("sync.route", sync_handler)
        result = await bus.request("sync.route", 5)

        assert result == 10


class TestPatchouliBusPubSub:
    """发布/订阅测试"""

    def test_subscribe_adds_callback(self):
        bus = PatchouliBus()
        callback = AsyncMock()

        bus.subscribe("test.event", callback)

        assert "test.event" in bus.list_events()

    def test_subscribe_multiple_callbacks_for_same_event(self):
        bus = PatchouliBus()
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        bus.subscribe("test.event", callback1)
        bus.subscribe("test.event", callback2)

        assert len(bus._subscribers["test.event"]) == 2

    def test_unsubscribe_removes_callback(self):
        bus = PatchouliBus()
        callback = AsyncMock()
        bus.subscribe("test.event", callback)

        bus.unsubscribe("test.event", callback)

        assert "test.event" not in bus.list_events()

    def test_unsubscribe_removes_event_when_last_callback_removed(self):
        bus = PatchouliBus()
        callback = AsyncMock()
        bus.subscribe("test.event", callback)

        bus.unsubscribe("test.event", callback)

        assert "test.event" not in bus._subscribers

    @pytest.mark.asyncio
    async def test_publish_calls_all_subscribers(self):
        bus = PatchouliBus()
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        bus.subscribe("test.event", callback1)
        bus.subscribe("test.event", callback2)

        await bus.publish("test.event", "data")

        callback1.assert_awaited_once_with("data")
        callback2.assert_awaited_once_with("data")

    @pytest.mark.asyncio
    async def test_publish_noop_when_no_subscribers(self):
        bus = PatchouliBus()

        # 不应抛出异常
        await bus.publish("no.subscribers", "data")

    @pytest.mark.asyncio
    async def test_publish_isolates_subscriber_failure(self):
        bus = PatchouliBus()
        callback1 = AsyncMock(side_effect=RuntimeError("subscriber error"))
        callback2 = AsyncMock()
        bus.subscribe("test.event", callback1)
        bus.subscribe("test.event", callback2)

        # callback2 仍应被调用，即使 callback1 失败
        await bus.publish("test.event", "data")

        callback2.assert_awaited_once_with("data")


class TestPatchouliBusList:
    """查询接口测试"""

    def test_list_routes_returns_sorted_routes(self):
        bus = PatchouliBus()
        bus.register("route.b", AsyncMock())
        bus.register("route.a", AsyncMock())

        routes = bus.list_routes()

        assert routes == ["route.a", "route.b"]

    def test_list_events_returns_sorted_events(self):
        bus = PatchouliBus()
        bus.subscribe("event.b", AsyncMock())
        bus.subscribe("event.a", AsyncMock())

        events = bus.list_events()

        assert events == ["event.a", "event.b"]

    def test_list_routes_empty_when_no_registration(self):
        bus = PatchouliBus()

        assert bus.list_routes() == []

    def test_list_events_empty_when_no_subscription(self):
        bus = PatchouliBus()

        assert bus.list_events() == []


class TestPatchouliBusRepr:
    """__repr__ 测试"""

    def test_repr_shows_route_and_event_counts(self):
        bus = PatchouliBus()
        bus.register("route.1", AsyncMock())
        bus.subscribe("event.1", AsyncMock())

        repr_str = repr(bus)

        assert "routes=1" in repr_str
        assert "events=1" in repr_str
