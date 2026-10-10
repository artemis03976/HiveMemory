"""Alice 跨系统总线桥接。

将 Alice 统一执行入口注册到系统级总线；运行时的资源操作经 workspace
操作提交函数进入能力层。
"""

from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.components.bus.global_bus import GlobalSystemBus

if TYPE_CHECKING:
    from hivememory.alice.application import AgentRunService


@dataclass(frozen=True)
class AlicePublicApi:
    """由 AliceBridge 挂载的 Alice 公开 API 面。"""

    agent: AgentRunService


class AliceBridge:
    """管理 Alice 统一执行入口在全局总线上的挂载与卸载。"""

    def __init__(
        self,
        *,
        public_api: AlicePublicApi,
        global_bus: GlobalSystemBus | None = None,
    ) -> None:
        self._public_api = public_api
        self._global_bus = global_bus
        self._public_routes_registered = False

    @property
    def public_routes_registered(self) -> bool:
        return self._public_routes_registered

    def mount(self) -> None:
        if self._global_bus is None:
            return

        if not self._public_routes_registered:
            self._register_public_routes()
            self._public_routes_registered = True

    def unmount(self) -> None:
        if self._global_bus is None:
            return

        if self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False

    # ========== 公开路由（Alice → 全局总线） ==========

    def _register_public_routes(self) -> None:
        if self._global_bus is None:
            return
        self._global_bus.register(
            AliceRoutes.RUN_AGENT,
            self._run_agent_route,
        )

    def _unregister_public_routes(self) -> None:
        if self._global_bus is None:
            return
        self._global_bus.unregister(AliceRoutes.RUN_AGENT)

    async def _run_agent_route(self, *args: Any, **kwargs: Any) -> Any:
        """统一执行路由 handler：非流式等待执行结果，流式返回事件生成器对象。"""
        call = self._public_api.agent.run_agent(*args, **kwargs)
        if isinstance(call, Coroutine):
            return await call
        return call


__all__ = ["AliceBridge", "AlicePublicApi"]
