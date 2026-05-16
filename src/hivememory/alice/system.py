"""
AliceSystem - 多智能体编排与计算子系统

SubsystemProtocol 实现，持有 AgentRuntimeHost 和 AliceService。
Phase C 最小骨架：先成为 Agent runtime 的正式宿主。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.bridge import AliceBridge
from hivememory.alice.runtime.host import AgentRuntimeHost
from hivememory.alice.service import AliceService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

if TYPE_CHECKING:
    from hivememory.patchouli.kernel import PatchouliKernel

logger = logging.getLogger(__name__)


class AliceSystem(SubsystemProtocol):
    """
    Alice 子系统 - 多智能体编排与计算子系统宿主

    Phase C 职责：
    - 持有 AgentRuntimeHost (KernelLoopExecutor, WorkerAgentService)
    - 提供 AliceService (run_agent / run_agent_stream)
    - 接入全局 bus / bridge
    - 实现 SubsystemProtocol 生命周期
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        kernel: "PatchouliKernel",
        global_bus: Optional[GlobalSystemBus] = None,
    ) -> None:
        self._config = config

        self._runtime_host = AgentRuntimeHost(
            kernel=kernel,
            config=config,
        )

        self._service = AliceService(runtime_host=self._runtime_host)

        self._local_bus = AliceBus()
        self._bridge = (
            AliceBridge(local_bus=self._local_bus, global_bus=global_bus)
            if global_bus is not None
            else None
        )

        self._local_routes_registered = False
        self._bridge_mounted = False

        logger.info("AliceSystem 初始化完成")

    @property
    def name(self) -> str:
        return "alice"

    @property
    def service(self) -> AliceService:
        return self._service

    @property
    def runtime_host(self) -> AgentRuntimeHost:
        return self._runtime_host

    async def start(self) -> None:
        if not self._local_routes_registered:
            self._register_local_routes()
            self._local_routes_registered = True
        if self._bridge and not self._bridge_mounted:
            self._bridge.mount()
            self._bridge_mounted = True

    async def stop(self) -> None:
        if self._bridge and self._bridge_mounted:
            self._bridge.unmount()
            self._bridge_mounted = False
        if self._local_routes_registered:
            self._unregister_local_routes()
            self._local_routes_registered = False

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self._runtime_host.health(),
        }

    def _register_local_routes(self) -> None:
        self._local_bus.register(
            AliceRoutes.RUN_AGENT,
            self._service.run_agent,
        )

    def _unregister_local_routes(self) -> None:
        self._local_bus.unregister(AliceRoutes.RUN_AGENT)
