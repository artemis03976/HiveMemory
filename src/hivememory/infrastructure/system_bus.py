"""
HiveMemory 系统总线 (SystemBus)

定位：进程内的统一通信基础设施，类似电脑主板。
职责：
    - RPC 模式 (Request-Response): 模拟 HTTP/API 调用，一个路由对应一个 handler
    - Pub/Sub 模式 (Event Broadcast): 模拟消息队列，一个事件可有多个订阅者

设计原则：
    - 同步优先：当前阶段所有 RPC 调用保持同步，不引入 async 改造
    - async_request 预留给未来 handler 异步化后使用
    - emit 对 sync handler 直接调用，对 async handler 用 create_task
    - 路由命名规范: {service}.{method}，如 "perception.get_active_topics_menu"

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class SystemBus:
    """
    HiveMemory 系统总线 — 进程内通信的主板

    两种通信模式：
        - RPC: register() + request() / async_request()
          一个路由对应一个 handler，调用方获取返回值。
        - Pub/Sub: subscribe() + emit()
          一个事件可有多个订阅者，fire-and-forget。

    使用示例:
        >>> bus = SystemBus()
        >>> bus.register("perception.get_active_topics_menu", perception.get_active_topics_menu)
        >>> menu = bus.request("perception.get_active_topics_menu")
        >>>
        >>> bus.subscribe("perception.flushed", librarian._on_perception_flush)
        >>> bus.emit("perception.flushed", messages=msgs, reason=reason)
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._subscribers: Dict[str, List[Callable]] = {}

    # ========== RPC 模式 (Request-Response) ==========

    def register(self, route: str, handler: Callable) -> None:
        """
        注册 RPC 路由

        一个路由只能有一个 handler，重复注册会覆盖并发出警告。

        Args:
            route: 路由名称，格式 "{service}.{method}"
            handler: 处理函数
        """
        if route in self._handlers:
            logger.warning(f"SystemBus: 路由 '{route}' 已存在，将被覆盖")
        self._handlers[route] = handler
        logger.debug(f"SystemBus: 注册路由 '{route}'")

    def request(self, route: str, *args, **kwargs) -> Any:
        """
        同步 RPC 调用

        Args:
            route: 路由名称
            *args, **kwargs: 传递给 handler 的参数

        Returns:
            handler 的返回值

        Raises:
            KeyError: 路由未注册
        """
        handler = self._handlers.get(route)
        if handler is None:
            raise KeyError(f"SystemBus: 路由 '{route}' 未注册")
        return handler(*args, **kwargs)

    async def async_request(self, route: str, *args, **kwargs) -> Any:
        """
        异步 RPC 调用

        自动区分 sync/async handler：
        - async handler: 直接 await
        - sync handler: 在事件循环中直接调用（不阻塞其他协程时可用）

        Args:
            route: 路由名称
            *args, **kwargs: 传递给 handler 的参数

        Returns:
            handler 的返回值

        Raises:
            KeyError: 路由未注册
        """
        handler = self._handlers.get(route)
        if handler is None:
            raise KeyError(f"SystemBus: 路由 '{route}' 未注册")
        if asyncio.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        return handler(*args, **kwargs)

    # ========== Pub/Sub 模式 (Event Broadcast) ==========

    def subscribe(self, event: str, callback: Callable) -> None:
        """
        订阅事件

        一个事件可有多个订阅者。

        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)
        logger.debug(f"SystemBus: 订阅事件 '{event}'")

    def unsubscribe(self, event: str, callback: Callable) -> None:
        """
        取消订阅

        Args:
            event: 事件名称
            callback: 要移除的回调函数
        """
        if event in self._subscribers:
            self._subscribers[event] = [
                cb for cb in self._subscribers[event] if cb is not callback
            ]

    def emit(self, event: str, *args, **kwargs) -> None:
        """
        发布事件 (fire-and-forget)

        调用所有订阅者。sync handler 直接调用，async handler 通过
        create_task 调度。单个订阅者的异常不影响其他订阅者。

        Args:
            event: 事件名称
            *args, **kwargs: 传递给订阅者的参数
        """
        subscribers = self._subscribers.get(event, [])
        for cb in subscribers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cb(*args, **kwargs))
                    except RuntimeError:
                        # 没有运行中的事件循环，同步执行
                        asyncio.run(cb(*args, **kwargs))
                else:
                    cb(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"SystemBus: 事件 '{event}' 的订阅者执行失败: {e}",
                    exc_info=True,
                )

    # ========== 内省 ==========

    def list_routes(self) -> List[str]:
        """列出所有已注册的 RPC 路由"""
        return sorted(self._handlers.keys())

    def list_events(self) -> List[str]:
        """列出所有有订阅者的事件"""
        return sorted(self._subscribers.keys())

    def __repr__(self) -> str:
        return (
            f"SystemBus(routes={len(self._handlers)}, "
            f"events={len(self._subscribers)})"
        )


__all__ = [
    "SystemBus",
]
