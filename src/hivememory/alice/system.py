"""
AliceSystem - 多智能体编排与计算子系统

SubsystemProtocol 实现，装配 AliceRuntime、AgentRunService、AliceCPU 与
AliceBridge。
"""

from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime.model_resolution import ModelResolver
from hivememory.alice.application import AgentRunService, AliceCPU
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.runtime.bridge import AliceBridge, AlicePublicApi
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.alice import AliceConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.contracts.subsystem import SubsystemProtocol
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.workspace.contracts import OperationEntry

logger = logging.getLogger(__name__)


class AliceSystem(SubsystemProtocol):
    """
    Alice 子系统 - 多智能体编排与计算子系统宿主

    职责：
    - 装配 AliceRuntime 与 AgentRunService
    - 提供统一的 run_agent 用例入口与 CPUPort 端口实现
    - 将公开路由注册到全局总线
    - 实现 SubsystemProtocol 生命周期
    """

    def __init__(
        self,
        config: AliceConfig,
        global_bus: GlobalSystemBus | None = None,
        event_publisher: RuntimeEventPublisher | None = None,
        model_registry: ModelResolver | None = None,
        *,
        operation_entry: OperationEntry,
        memory_compiler_config: MemoryCompilerConfig | None = None,
    ) -> None:
        self._config = config
        publisher = event_publisher or RuntimeEventPublisher(NullRuntimeEventSink())

        self._runtime = AliceRuntime(
            alice_config=config,
            memory_compiler_config=memory_compiler_config or MemoryCompilerConfig(),
            model_registry=model_registry,
        )

        frame_factory = FrameFactory()
        prompt_assembler = AgentPromptAssembler(config.koakuma)
        call_context_provider = CallContextProvider()
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
            stream_adapter=AgentRunStreamAdapter(),
            agent_run_events=AgentRunEventEmitter(publisher.scoped(component="agent_run_service")),
        )

        self._bridge = AliceBridge(
            public_api=AlicePublicApi(agent=self._service),
            global_bus=global_bus,
        )

        # 任务进程经 CPUPort 端口调用本子系统；端口实现由组合根注入进程，
        # workspace 侧不出现 Alice 的路由名或结果类型。
        self._cpu = AliceCPU(global_bus, operation_entry) if global_bus is not None else None

        logger.info("AliceSystem 初始化完成")

    @property
    def name(self) -> str:
        return "alice"

    @property
    def service(self) -> AgentRunService:
        return self._service

    @property
    def cpu_port(self) -> AliceCPU:
        """Alice 充当任务进程 CPU 的端口实现（要求装配了全局总线）。"""
        if self._cpu is None:
            raise RuntimeError("AliceSystem 未装配全局总线，无法提供 CPU 端口实现")
        return self._cpu

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
