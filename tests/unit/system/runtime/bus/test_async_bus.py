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

from hivememory.system.runtime.bus.async_bus import AsyncSystemBus


class TestAsyncSystemBusRPC:

    def setup_method(self):
        self.bus = AsyncSystemBus()

    @pytest.mark.asyncio
    async def test_register_and_request(self):
        received = []

        async def handler(*args, **kwargs):
            received.append((args, kwargs))
            return "result"

        self.bus.register("svc.method", handler)

        result = await self.bus.request("svc.method")

        assert result == "result"
        assert received == [((), {})]

    @pytest.mark.asyncio
    async def test_request_unregistered_route_raises_keyerror(self):
        with pytest.raises(KeyError, match="not registered"):
            await self.bus.request("nonexistent.route")

    @pytest.mark.asyncio
    async def test_register_overwrites_existing(self):
        calls = []

        async def old_handler():
            calls.append("old")

        async def new_handler():
            calls.append("new")

        self.bus.register("svc.method", old_handler)
        self.bus.register("svc.method", new_handler)

        await self.bus.request("svc.method")

        assert calls == ["new"]

    @pytest.mark.asyncio
    async def test_request_passes_args_kwargs(self):
        received = []

        async def handler(*args, **kwargs):
            received.append((args, kwargs))

        self.bus.register("svc.method", handler)

        await self.bus.request("svc.method", "arg1", "arg2", key="val")

        assert received == [(("arg1", "arg2"), {"key": "val"})]

    @pytest.mark.asyncio
    async def test_request_accepts_sync_handler(self):
        received = []

        def handler(*args, **kwargs):
            received.append((args, kwargs))
            return ["snapshot"]

        self.bus.register("svc.sync_method", handler)

        result = await self.bus.request("svc.sync_method", key="val")

        assert result == ["snapshot"]
        assert received == [((), {"key": "val"})]

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
        received = []

        async def callback(*args, **kwargs):
            received.append((args, kwargs))

        self.bus.subscribe("evt.test", callback)

        await self.bus.publish("evt.test", "data", key="val")

        assert received == [(("data",), {"key": "val"})]

    @pytest.mark.asyncio
    async def test_publish_multiple_subscribers(self):
        received1 = []
        received2 = []

        async def cb1(*args, **kwargs):
            received1.append(args)

        async def cb2(*args, **kwargs):
            received2.append(args)

        self.bus.subscribe("evt.test", cb1)
        self.bus.subscribe("evt.test", cb2)

        await self.bus.publish("evt.test", "data")

        assert received1 == [("data",)]
        assert received2 == [("data",)]

    @pytest.mark.asyncio
    async def test_publish_no_subscribers_is_noop(self):
        await self.bus.publish("evt.nobody_listens", "data")

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_callback(self):
        received = []

        async def callback(*args, **kwargs):
            received.append(args)

        self.bus.subscribe("evt.test", callback)
        self.bus.unsubscribe("evt.test", callback)

        await self.bus.publish("evt.test", "data")

        assert received == []

    def test_unsubscribe_nonexistent_is_noop(self):
        self.bus.unsubscribe("evt.nonexistent", AsyncMock())

    @pytest.mark.asyncio
    async def test_publish_subscriber_exception_isolated(self):
        received = []

        async def bad_cb(*args, **kwargs):
            raise RuntimeError("boom")

        async def good_cb(*args, **kwargs):
            received.append(args)

        self.bus.subscribe("evt.test", bad_cb)
        self.bus.subscribe("evt.test", good_cb)

        await self.bus.publish("evt.test", "data")

        assert received == [("data",)]


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
