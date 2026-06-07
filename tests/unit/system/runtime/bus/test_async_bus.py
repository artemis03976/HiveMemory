"""
AsyncSystemBus 单元测试

覆盖:
- RPC: register / request / unregister
- Pub/Sub: subscribe / publish / unsubscribe
- 异常隔离: 单个 subscriber 异常不影响其他
- 内省: list_routes / list_events / __repr__
"""

import pytest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from hivememory.system.runtime.bus.async_bus import AsyncSystemBus


class TestAsyncSystemBusRPC:

    def setup_method(self):
        self.bus = AsyncSystemBus()

    @pytest.mark.asyncio
    async def test_register_and_request(self):
        handler = AsyncMock(return_value="result")
        self.bus.register("svc.method", handler)

        result = await self.bus.request("svc.method")

        handler.assert_awaited_once()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_request_unregistered_route_raises_keyerror(self):
        with pytest.raises(KeyError, match="not registered"):
            await self.bus.request("nonexistent.route")

    @pytest.mark.asyncio
    async def test_register_overwrites_existing(self):
        old_handler = AsyncMock(return_value="old")
        new_handler = AsyncMock(return_value="new")

        self.bus.register("svc.method", old_handler)
        self.bus.register("svc.method", new_handler)

        result = await self.bus.request("svc.method")

        old_handler.assert_not_awaited()
        new_handler.assert_awaited_once()
        assert result == "new"

    @pytest.mark.asyncio
    async def test_request_passes_args_kwargs(self):
        handler = AsyncMock(return_value="ok")
        self.bus.register("svc.method", handler)

        await self.bus.request("svc.method", "arg1", "arg2", key="val")

        handler.assert_awaited_once_with("arg1", "arg2", key="val")

    @pytest.mark.asyncio
    async def test_request_accepts_sync_handler(self):
        handler = Mock(return_value=["snapshot"])
        self.bus.register("svc.sync_method", handler)

        result = await self.bus.request("svc.sync_method", key="val")

        handler.assert_called_once_with(key="val")
        assert result == ["snapshot"]

    @pytest.mark.asyncio
    async def test_unregister_removes_handler(self):
        handler = AsyncMock(return_value="x")
        self.bus.register("svc.method", handler)
        self.bus.unregister("svc.method")

        with pytest.raises(KeyError):
            await self.bus.request("svc.method")

    def test_unregister_nonexistent_is_noop(self):
        self.bus.unregister("nonexistent.route")


class TestAsyncSystemBusPubSub:

    def setup_method(self):
        self.bus = AsyncSystemBus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        callback = AsyncMock()
        self.bus.subscribe("evt.test", callback)

        await self.bus.publish("evt.test", "data", key="val")

        callback.assert_awaited_once_with("data", key="val")

    @pytest.mark.asyncio
    async def test_publish_multiple_subscribers(self):
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        self.bus.subscribe("evt.test", cb1)
        self.bus.subscribe("evt.test", cb2)

        await self.bus.publish("evt.test", "data")

        cb1.assert_awaited_once_with("data")
        cb2.assert_awaited_once_with("data")

    @pytest.mark.asyncio
    async def test_publish_no_subscribers_is_noop(self):
        await self.bus.publish("evt.nobody_listens", "data")

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_callback(self):
        callback = AsyncMock()
        self.bus.subscribe("evt.test", callback)
        self.bus.unsubscribe("evt.test", callback)

        await self.bus.publish("evt.test", "data")

        callback.assert_not_awaited()

    def test_unsubscribe_nonexistent_is_noop(self):
        self.bus.unsubscribe("evt.nonexistent", AsyncMock())

    @pytest.mark.asyncio
    async def test_publish_subscriber_exception_isolated(self):
        bad_cb = AsyncMock(side_effect=RuntimeError("boom"))
        good_cb = AsyncMock()
        self.bus.subscribe("evt.test", bad_cb)
        self.bus.subscribe("evt.test", good_cb)

        await self.bus.publish("evt.test", "data")

        bad_cb.assert_awaited_once()
        good_cb.assert_awaited_once_with("data")


class TestAsyncSystemBusIntrospection:

    def setup_method(self):
        self.bus = AsyncSystemBus()

    def test_list_routes_sorted(self):
        self.bus.register("b.method", AsyncMock())
        self.bus.register("a.method", AsyncMock())

        assert self.bus.list_routes() == ["a.method", "b.method"]

    def test_list_routes_empty(self):
        assert self.bus.list_routes() == []

    def test_list_events_sorted(self):
        self.bus.subscribe("evt.b", AsyncMock())
        self.bus.subscribe("evt.a", AsyncMock())

        assert self.bus.list_events() == ["evt.a", "evt.b"]

    def test_list_events_empty(self):
        assert self.bus.list_events() == []

    def test_repr(self):
        self.bus.register("r1", AsyncMock())
        self.bus.subscribe("e1", AsyncMock())

        r = repr(self.bus)

        assert "AsyncSystemBus" in r
        assert "routes=1" in r
        assert "events=1" in r
