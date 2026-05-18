"""
AsyncSystemBus — 纯异步系统总线基类

第四次架构演进的统一通信骨架。

设计原则:
    - 纯 asyncio，只接受 async handler，不做 asyncio.run() 回退
    - RPC: register() + request() (async only)
    - Pub/Sub: subscribe() + publish() (async, 异常隔离)
    - request() 对未注册路由抛 KeyError
    - publish() 对无订阅者的事件静默 no-op
"""

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class AsyncSystemBus:
    """纯异步系统总线基类 — 所有新总线实现的共同祖先。"""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._subscribers: dict[str, list[Callable[..., Awaitable[None]]]] = {}

    # ========== RPC (Request-Response) ==========

    def register(self, route: str, handler: Callable[..., Awaitable[Any]]) -> None:
        if route in self._handlers:
            logger.warning(f"AsyncSystemBus: route '{route}' overwritten")
        self._handlers[route] = handler

    def unregister(self, route: str) -> None:
        self._handlers.pop(route, None)

    async def request(self, route: str, *args: Any, **kwargs: Any) -> Any:
        handler = self._handlers.get(route)
        if handler is None:
            raise KeyError(f"AsyncSystemBus: route '{route}' not registered")
        return await handler(*args, **kwargs)

    # ========== Pub/Sub (Event Broadcast) ==========

    def subscribe(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        if event in self._subscribers:
            self._subscribers[event] = [
                cb for cb in self._subscribers[event] if cb is not callback
            ]

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

    def list_events(self) -> list[str]:
        return sorted(self._subscribers.keys())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(routes={len(self._handlers)}, "
            f"events={len(self._subscribers)})"
        )
