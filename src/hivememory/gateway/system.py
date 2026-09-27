"""GatewaySystem：Gateway 标准子系统门面。"""

from __future__ import annotations

import logging
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RuntimeEventSink,
)
from hivememory.config.gateway import SystemGatewayConfig
from hivememory.config.shared import LLMConfig
from hivememory.core.contracts.subsystem import SubsystemProtocol
from hivememory.gateway.contracts.public_routes import GatewayPublicRoutes
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.infrastructure.llm import get_gateway_llm_service

logger = logging.getLogger(__name__)


class GatewaySystem(SubsystemProtocol):
    """
    Gateway 子系统宿主，负责生命周期和公开路由挂载。
    """

    def __init__(
        self,
        config: SystemGatewayConfig,
        global_bus: GlobalSystemBus,
        runtime_events: RuntimeEventSink | None = None,
        *,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self._config = config
        self._global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        # Gateway LLM 配置由组合根解析后注入；未配置模型时不创建 LLM 服务。
        llm_service = (
            get_gateway_llm_service(llm_config)
            if llm_config is not None and llm_config.model is not None
            else None
        )

        self._runtime = GatewayRuntime(
            config=self._config,
            global_bus=global_bus,
            runtime_events=self._runtime_events,
            llm_service=llm_service,
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
        if not self._public_routes_registered:
            self._global_bus.register(
                GatewayPublicRoutes.PROCESS,
                self._service.process,
            )
            self._public_routes_registered = True

    async def stop(self) -> None:
        if self._public_routes_registered:
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
