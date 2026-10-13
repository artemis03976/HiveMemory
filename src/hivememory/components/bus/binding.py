"""在构造处校验签名、在存储时擦除类型的异构路由绑定。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from hivememory.core.contracts.route import Route


class RouteBinding(NamedTuple):
    """保存已校验的路由与 handler，供总线逐项注册。"""

    route: Route[..., Any]
    handler: Callable[..., Awaitable[Any]]

    @classmethod
    def of[**P, R](cls, route: Route[P, R], handler: Callable[P, Awaitable[R]]) -> RouteBinding:
        """先检查单条绑定的完整签名，再统一存入异构绑定表。"""
        return cls(route, handler)
