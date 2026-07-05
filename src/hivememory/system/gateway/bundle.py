"""Gateway装配产物 — 将 TheEye 与 CommandDispatcher 绑定为单一交付物。"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.system.gateway.commands import SystemCommandDispatcher
from hivememory.system.gateway.eye import TheEye


@dataclass(frozen=True)
class GatewayBundle:
    """
    System Gateway 装配结果。

    TheEye 与 SystemCommandDispatcher 总是一起被创建、一起被注入，
    用 Bundle 绑定两者，避免在调用链中分别传递两个相关对象。
    """

    eye: TheEye
    command_dispatcher: SystemCommandDispatcher | None


__all__ = ["GatewayBundle"]
