"""
AsyncSystemBus — 纯异步系统总线基类

第四次架构演进的统一通信骨架。

设计原则:
    - 纯 asyncio，只接受 async handler，不做 asyncio.run() 回退
    - RPC: register() + request() (async only)
    - Pub/Sub: subscribe() + publish() (async, 异常隔离)
    - request() 对未注册路由抛 KeyError
    - request() 在调用 handler 前按其签名与类型标注只校验、不转换参数，
      不符时抛 RouteArgumentError（见 ``arguments`` 模块）
    - publish() 对无订阅者的事件静默 no-op
"""

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from hivememory.components.bus.arguments import RouteArgumentChecker

logger = logging.getLogger(__name__)


class AsyncSystemBus:
    """纯异步系统总线基类 — 所有新总线实现的共同祖先。"""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._checkers: dict[str, RouteArgumentChecker | None] = {}
        self._subscribers: dict[str, list[Callable[..., Awaitable[None]]]] = {}

    # ========== RPC（请求-响应）==========

    def register(self, route: str, handler: Callable[..., Awaitable[Any]]) -> None:
        if route in self._handlers:
            logger.warning(f"AsyncSystemBus: route '{route}' overwritten")
        self._handlers[route] = handler
        self._checkers[route] = RouteArgumentChecker.for_handler(route, handler)

    def unregister(self, route: str) -> None:
        self._handlers.pop(route, None)
        self._checkers.pop(route, None)

    async def request(self, route: str, *args: Any, **kwargs: Any) -> Any:
        handler = self._handlers.get(route)
        if handler is None:
            raise KeyError(f"AsyncSystemBus: route '{route}' not registered")
        checker = self._checkers.get(route)
        if checker is not None:
            checker.check(args, kwargs)
        result = handler(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    # ========== Pub/Sub（事件广播）==========

    def subscribe(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        if event in self._subscribers:
            self._subscribers[event] = [cb for cb in self._subscribers[event] if cb != callback]
            if not self._subscribers[event]:
                self._subscribers.pop(event, None)

    async def publish(self, event: str, *args: Any, **kwargs: Any) -> None:
        subscribers = self._subscribers.get(event, [])
        for cb in subscribers:
            try:
                await cb(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"AsyncSystemBus: subscriber for event '{event}' failed: {e}",
                    exc_info=True,
                )

    def list_routes(self) -> list[str]:
        return sorted(self._handlers.keys())

    def list_unresolved_routes(self) -> list[str]:
        """handler 类型标注无法在运行时解析、只按签名校验参数的路由。"""
        return sorted(
            route
            for route, checker in self._checkers.items()
            if checker is not None and not checker.annotations_resolved
        )

    def list_events(self) -> list[str]:
        return sorted(self._subscribers.keys())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(routes={len(self._handlers)}, "
            f"events={len(self._subscribers)})"
        )
