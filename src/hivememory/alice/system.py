"""
AliceSystem - 多智能体编排与计算子系统

SubsystemProtocol 实现，持有 AliceRuntime 和 AliceService。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.service import AliceService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)


class AliceSystem(SubsystemProtocol):
    """
    Alice 子系统 - 多智能体编排与计算子系统宿主

    职责：
    - 持有 AliceRuntime
    - 提供 AliceService (run_agent / run_agent_stream)
    - 将公开路由注册到全局总线
    - 实现 SubsystemProtocol 生命周期
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: Optional[GlobalSystemBus] = None,
    ) -> None:
        self._config = config
        self._global_bus = global_bus

        self._runtime = AliceRuntime(
            config=config,
            global_bus=global_bus,
        )
        self._service = AliceService(runtime=self._runtime)

        self._public_routes_registered = False

        logger.info("AliceSystem 初始化完成")

    @property
    def name(self) -> str:
        return "alice"

    @property
    def service(self) -> AliceService:
        return self._service

    @property
    def runtime(self) -> AliceRuntime:
        return self._runtime

    async def start(self) -> None:
        self._runtime.mount_local_routes()
        if self._global_bus and not self._public_routes_registered:
            self._register_public_routes()
            self._public_routes_registered = True

    async def stop(self) -> None:
        if self._global_bus and self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False
        self._runtime.unmount_local_routes()

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self._runtime.health(),
        }

    def _register_public_routes(self) -> None:
        self._global_bus.register(
            AliceRoutes.RUN_AGENT,
            self._service.run_agent,
        )
        self._global_bus.register(
            AliceRoutes.RUN_AGENT_STREAM,
            self._run_agent_stream_route,
        )
        self._global_bus.register(
            AliceRoutes.REGISTER_PRERETRIEVAL_ALIASES,
            self._service.register_preretrieval_aliases,
        )
        self._global_bus.register(
            AliceRoutes.GET_INTERACTION_STATE,
            self._service.get_interaction_state,
        )

    def _unregister_public_routes(self) -> None:
        self._global_bus.unregister(AliceRoutes.RUN_AGENT)
        self._global_bus.unregister(AliceRoutes.RUN_AGENT_STREAM)
        self._global_bus.unregister(AliceRoutes.REGISTER_PRERETRIEVAL_ALIASES)
        self._global_bus.unregister(AliceRoutes.GET_INTERACTION_STATE)

    async def _run_agent_stream_route(self, *args: Any, **kwargs: Any) -> Any:
        """为 AsyncSystemBus 适配流式 handler，返回 async generator 对象。"""
        return self._service.run_agent_stream(*args, **kwargs)
