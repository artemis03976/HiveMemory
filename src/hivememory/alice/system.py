"""
AliceSystem - 多智能体编排与计算子系统

SubsystemProtocol 实现，持有 AliceRuntime 和 AliceService。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hivememory.alice.runtime.bridge import AliceBridge, AlicePublicApi
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.service import AliceService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

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
        runtime_events: RuntimeEventSink | None = None,
        model_registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._config = config
        self._runtime_events = runtime_events or NullRuntimeEventSink()

        self._runtime = AliceRuntime(
            alice_config=config.alice,
            shared_config=config.shared,
            memory_compiler_config=config.memory_compiler,
            runtime_events=self._runtime_events,
            model_registry=model_registry,
        )

        self._service = AliceService(runtime=self._runtime)

        self._bridge = AliceBridge(
            local_bus=self._runtime.local_bus,
            runtime=self._runtime,
            public_api=AlicePublicApi(agent=self._service),
            global_bus=global_bus,
        )

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
        self._bridge.mount()

    async def stop(self) -> None:
        self._bridge.unmount()
        self._runtime.unmount_local_routes()

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self._runtime.health(),
        }
