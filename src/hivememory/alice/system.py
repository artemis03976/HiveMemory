"""
AliceSystem - 多智能体编排与计算子系统

SubsystemProtocol 实现，装配 AliceRuntime、AgentRunService 与 AliceBridge。
"""

from __future__ import annotations

import logging
from typing import Any

from hivememory.alice.application import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.runtime.bridge import AliceBridge, AlicePublicApi
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher

logger = logging.getLogger(__name__)


class AliceSystem(SubsystemProtocol):
    """
    Alice 子系统 - 多智能体编排与计算子系统宿主

    职责：
    - 装配 AliceRuntime 与 AgentRunService
    - 提供稳定的 run_agent / run_agent_stream 用例入口
    - 将公开路由注册到全局总线
    - 实现 SubsystemProtocol 生命周期
    """

    def __init__(
        self,
        config: HiveMemoryConfig,
        global_bus: GlobalSystemBus | None = None,
        event_publisher: RuntimeEventPublisher | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._config = config
        publisher = event_publisher or RuntimeEventPublisher(NullRuntimeEventSink())

        self._runtime = AliceRuntime(
            alice_config=config.alice,
            memory_compiler_config=config.memory_compiler,
            model_registry=model_registry,
        )

        frame_factory = FrameFactory()
        prompt_assembler = AgentPromptAssembler(config.alice.koakuma)
        call_context_provider = CallContextProvider(
            self._runtime.profile_resolver,
            self._runtime.alias_resolver,
        )
        call_coordinator = CallCoordinator(
            self._runtime.agent_runtime,
            call_context_provider,
            frame_factory=frame_factory,
            prompt_assembler=prompt_assembler,
        )
        self._service = AgentRunService(
            agent_runtime=self._runtime.agent_runtime,
            call_coordinator=call_coordinator,
            frame_factory=frame_factory,
            prompt_assembler=prompt_assembler,
            atom_cache=self._runtime.atom_cache,
            stream_adapter=AgentRunStreamAdapter(),
            agent_run_events=AgentRunEventEmitter(publisher.scoped(component="agent_run_service")),
        )

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
    def service(self) -> AgentRunService:
        return self._service

    @property
    def runtime(self) -> AliceRuntime:
        return self._runtime

    async def start(self) -> None:
        self._bridge.mount()

    async def stop(self) -> None:
        self._bridge.unmount()

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self._runtime.health(),
        }
