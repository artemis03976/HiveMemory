"""Alice 运行服务与真实流适配器的测试装配。"""

from collections.abc import AsyncGenerator
from typing import Any

from hivememory.agent_runtime.model.client import LLMClient
from hivememory.agent_runtime.runtime import AgentRuntime, build_agent_runtime
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.events import AgentRunEventEmitter
from hivememory.alice.interface.stream import AgentRunStream
from hivememory.alice.orchestration.call import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.run.service import AgentRunService
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.app import HiveMemoryConfig
from hivememory.workspace.contracts import (
    CPUInputManifest,
    CPUOutput,
    CPUResultSlot,
    OperationSubmitter,
)


def stream_run(
    service: AgentRunService,
    manifest: CPUInputManifest,
    generation_options: dict[str, Any] | None = None,
    *,
    submit_operation: OperationSubmitter,
    result_slot: CPUResultSlot | None = None,
) -> AsyncGenerator[CPUOutput, None]:
    """直接组合执行服务与接口层，保持事件和类型化终态原样交付。"""
    stream = AgentRunStream()
    return stream.events(
        service.run_agent(
            manifest,
            generation_options=generation_options,
            submit_operation=submit_operation,
            result_slot=result_slot,
            run_output=stream.output,
        )
    )


def build_run_service(
    *, runtime_events=None, config=None, model_registry=None
) -> tuple[AgentRuntime, AgentRunService]:
    """按默认配置装配真实内核、CALL 编排与运行服务；调用方按需替换执行层。"""
    config = config or HiveMemoryConfig()
    runtime = build_agent_runtime(
        runtime_config=config.alice.runtime,
        koakuma_config=config.alice.koakuma,
        memory_compiler_config=config.memory_compiler,
        model_registry=model_registry,
    )
    frame_factory = FrameFactory()
    coordinator = CallCoordinator(
        runtime,
        CallContextProvider(),
        frame_factory=frame_factory,
        koakuma_config=config.alice.koakuma,
    )
    service = AgentRunService(
        summarizer=RelaySummarizer(LLMClient()),
        agent_runtime=runtime,
        call_coordinator=coordinator,
        frame_factory=frame_factory,
        koakuma_config=config.alice.koakuma,
        compaction_config=config.alice.compaction,
        agent_run_events=AgentRunEventEmitter(
            RuntimeEventPublisher(runtime_events or NullRuntimeEventSink())
        ),
    )
    return runtime, service
