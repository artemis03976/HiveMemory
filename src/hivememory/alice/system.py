"""
AliceSystem - 多智能体编排与计算子系统

SubsystemProtocol 实现，装配 AgentRuntime、AgentRunService 与 AliceCPU；
对外只交出 CPU 端口。
"""

from __future__ import annotations

import logging
from typing import Any

from hivememory.agent_runtime import LLMClient, ModelResolver, build_agent_runtime
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.events import AgentRunEventEmitter
from hivememory.alice.interface.cpu import AliceCPU
from hivememory.alice.interface.stream import AgentRunStreamAdapter
from hivememory.alice.orchestration.call import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.run.service import AgentRunService
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
    - 装配 AgentRuntime 与 AgentRunService
    - 交出 CPUPort 端口实现，作为 Alice 唯一的对外执行入口
    - 通过 CPU 端口的可用状态管理执行准入
    - 实现 SubsystemProtocol 生命周期
    """

    def __init__(
        self,
        config: AliceConfig,
        event_publisher: RuntimeEventPublisher | None = None,
        model_registry: ModelResolver | None = None,
        *,
        operation_entry: OperationEntry,
        memory_compiler_config: MemoryCompilerConfig | None = None,
    ) -> None:
        self._config = config
        publisher = event_publisher or RuntimeEventPublisher(NullRuntimeEventSink())

        self._runtime = build_agent_runtime(
            runtime_config=config.runtime,
            koakuma_config=config.koakuma,
            memory_compiler_config=memory_compiler_config or MemoryCompilerConfig(),
            model_registry=model_registry,
        )

        frame_factory = FrameFactory()
        call_context_provider = CallContextProvider()
        call_coordinator = CallCoordinator(
            self._runtime,
            call_context_provider,
            frame_factory=frame_factory,
            prompt_assembler=AgentPromptAssembler(config.koakuma),
        )
        service = AgentRunService(
            summarizer=RelaySummarizer(LLMClient()),
            agent_runtime=self._runtime,
            call_coordinator=call_coordinator,
            frame_factory=frame_factory,
            koakuma_config=config.koakuma,
            agent_run_events=AgentRunEventEmitter(publisher.scoped(component="agent_run_service")),
            compaction_config=config.compaction,
        )

        self._cpu = AliceCPU(service, operation_entry, stream_adapter=AgentRunStreamAdapter())

        logger.info("AliceSystem 初始化完成")

    @property
    def name(self) -> str:
        return "alice"

    @property
    def cpu_port(self) -> AliceCPU:
        """Alice 充当任务进程 CPU 的端口实现；启停由宿主生命周期控制。"""
        return self._cpu

    async def start(self) -> None:
        self._cpu.start()

    async def stop(self) -> None:
        self._cpu.stop()

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self._runtime.health(),
        }
