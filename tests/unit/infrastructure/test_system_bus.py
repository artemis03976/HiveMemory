"""
SystemBus 单元测试

测试覆盖:
- RPC 模式: register / request / async_request
- Pub/Sub 模式: subscribe / unsubscribe / emit
- 异常隔离: 单个订阅者异常不影响其他
- 内省: list_routes / list_events / __repr__
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from hivememory.infrastructure.system_bus import SystemBus


class TestSystemBusRPC:
    """SystemBus RPC 模式单元测试"""

    def setup_method(self):
        self.bus = SystemBus()

    def test_register_and_request(self):
        """注册后可正常调用"""
        handler = Mock(return_value="result")
        self.bus.register("svc.method", handler)

        result = self.bus.request("svc.method")

        handler.assert_called_once()
        assert result == "result"

    def test_request_unregistered_route(self):
        """未注册路由抛 KeyError"""
        with pytest.raises(KeyError, match="未注册"):
            self.bus.request("nonexistent.route")

    def test_register_overwrites_existing(self):
        """重复注册覆盖旧 handler"""
        old_handler = Mock(return_value="old")
        new_handler = Mock(return_value="new")

        self.bus.register("svc.method", old_handler)
        self.bus.register("svc.method", new_handler)

        result = self.bus.request("svc.method")

        old_handler.assert_not_called()
        new_handler.assert_called_once()
        assert result == "new"

    def test_request_with_args_kwargs(self):
        """参数正确传递给 handler"""
        handler = Mock(return_value="ok")
        self.bus.register("svc.method", handler)

        self.bus.request("svc.method", "arg1", "arg2", key="val")

        handler.assert_called_once_with("arg1", "arg2", key="val")


class TestSystemBusAsyncRPC:
    """SystemBus 异步 RPC 单元测试"""

    def setup_method(self):
        self.bus = SystemBus()

    @pytest.mark.asyncio
    async def test_async_request_sync_handler(self):
        """同步 handler 通过 async_request 正常调用"""
        handler = Mock(return_value=42)
        self.bus.register("svc.sync", handler)

        result = await self.bus.async_request("svc.sync", "a")

        handler.assert_called_once_with("a")
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_request_async_handler(self):
        """异步 handler 通过 async_request 正常 await"""
        async def async_handler(x):
            return x * 2

        self.bus.register("svc.async", async_handler)

        result = await self.bus.async_request("svc.async", 5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_async_request_unregistered(self):
        """未注册路由抛 KeyError"""
        with pytest.raises(KeyError, match="未注册"):
            await self.bus.async_request("nonexistent")


class TestSystemBusPubSub:
    """SystemBus Pub/Sub 模式单元测试"""

    def setup_method(self):
        self.bus = SystemBus()

    def test_subscribe_and_emit(self):
        """订阅后收到事件"""
        callback = Mock()
        self.bus.subscribe("evt.test", callback)

        self.bus.emit("evt.test", "data", key="val")

        callback.assert_called_once_with("data", key="val")

    def test_emit_multiple_subscribers(self):
        """多个订阅者都收到事件"""
        cb1 = Mock()
        cb2 = Mock()
        self.bus.subscribe("evt.test", cb1)
        self.bus.subscribe("evt.test", cb2)

        self.bus.emit("evt.test", "data")

        cb1.assert_called_once_with("data")
        cb2.assert_called_once_with("data")

    def test_emit_no_subscribers(self):
        """无订阅者时 emit 不报错"""
        self.bus.emit("evt.nobody_listens", "data")

    def test_unsubscribe(self):
        """取消订阅后不再收到事件"""
        callback = Mock()
        self.bus.subscribe("evt.test", callback)
        self.bus.unsubscribe("evt.test", callback)

        self.bus.emit("evt.test", "data")

        callback.assert_not_called()

    def test_unsubscribe_nonexistent_event(self):
        """取消订阅不存在的事件不报错"""
        self.bus.unsubscribe("evt.nonexistent", Mock())

    def test_emit_subscriber_exception_isolated(self):
        """单个订阅者异常不影响其他订阅者"""
        bad_cb = Mock(side_effect=RuntimeError("boom"))
        good_cb = Mock()
        self.bus.subscribe("evt.test", bad_cb)
        self.bus.subscribe("evt.test", good_cb)

        self.bus.emit("evt.test", "data")

        bad_cb.assert_called_once()
        good_cb.assert_called_once_with("data")


class TestSystemBusIntrospection:
    """SystemBus 内省方法单元测试"""

    def setup_method(self):
        self.bus = SystemBus()

    def test_list_routes(self):
        """返回排序的路由列表"""
        self.bus.register("b.method", Mock())
        self.bus.register("a.method", Mock())

        routes = self.bus.list_routes()

        assert routes == ["a.method", "b.method"]

    def test_list_routes_empty(self):
        """无路由时返回空列表"""
        assert self.bus.list_routes() == []

    def test_list_events(self):
        """返回排序的事件列表"""
        self.bus.subscribe("evt.b", Mock())
        self.bus.subscribe("evt.a", Mock())

        events = self.bus.list_events()

        assert events == ["evt.a", "evt.b"]

    def test_list_events_empty(self):
        """无事件时返回空列表"""
        assert self.bus.list_events() == []

    def test_repr(self):
        """__repr__ 格式正确"""
        self.bus.register("r1", Mock())
        self.bus.subscribe("e1", Mock())

        r = repr(self.bus)

        assert "routes=1" in r
        assert "events=1" in r
