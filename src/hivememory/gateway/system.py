"""GatewaySystem：Gateway 标准子系统门面。"""

from __future__ import annotations

import logging
from typing import Any

from hivememory.gateway.contracts.public_routes import GatewayPublicRoutes
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

logger = logging.getLogger(__name__)


class GatewaySystem(SubsystemProtocol):
    """
    Gateway 子系统宿主，负责生命周期和公开路由挂载。
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: GlobalSystemBus | None = None,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._config = config
        self._global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        self._runtime = GatewayRuntime(
            config=self._config.gateway,
            global_bus=global_bus,
            runtime_events=self._runtime_events,
        )

        self._service = GatewayService(runtime=self._runtime)
        
        self._public_routes_registered = False

        logger.info("GatewaySystem 初始化完成")

    @property
    def name(self) -> str:
        return "gateway"

    @property
    def service(self) -> GatewayService:
        return self._service

    @property
    def runtime(self) -> GatewayRuntime:
        return self._runtime

    @property
    def public_routes_registered(self) -> bool:
        return self._public_routes_registered

    async def start(self) -> None:
        self._runtime.mount_local_routes(self._service)
        if self._global_bus is not None and not self._public_routes_registered:
            self._global_bus.register(
                GatewayPublicRoutes.PROCESS,
                self._service.process,
            )
            self._public_routes_registered = True

    async def stop(self) -> None:
        if self._global_bus is not None and self._public_routes_registered:
            self._global_bus.unregister(GatewayPublicRoutes.PROCESS)
            self._public_routes_registered = False
        self._runtime.unmount_local_routes()

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self._runtime.health(),
            "public_routes_registered": self._public_routes_registered,
        }


__all__ = ["GatewaySystem"]
