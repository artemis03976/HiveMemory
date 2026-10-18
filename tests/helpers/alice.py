"""Alice 运行服务与真实流适配器的测试装配。"""

from collections.abc import AsyncGenerator
from typing import Any

from hivememory.agent_runtime.model.generator import MTPGenerator
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
from hivememory.config.shared import LLMConfig
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.workspace.contracts import (
    CPUInputManifest,
    CPUOutput,
    CPUResultSlot,
    OperationSubmitter,
)
from tests.helpers.llm import ScriptedLLMService


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


class StaticModelResolver:
    """把任意模型名解析为同一个可执行测试模型的注册表替身，覆盖值照常合并。"""

    def __init__(self, model: str = "test-model") -> None:
        self._model = model

    def resolve(
        self,
        model_name: str,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        top_p_override: float | None = None,
    ) -> tuple[LLMConfig, str]:
        overrides = {
            "temperature": temperature_override,
            "max_tokens": max_tokens_override,
            "top_p": top_p_override,
        }
        config = LLMConfig(model=self._model).model_copy(
            update={key: value for key, value in overrides.items() if value is not None}
        )
        return config, model_name


def build_run_service(
    *,
    llm_service: BaseLLMService | None = None,
    runtime_events=None,
    config=None,
    model_registry=None,
) -> tuple[AgentRuntime, AgentRunService]:
    """按默认配置装配真实内核、CALL 编排与运行服务；模型边界由调用方注入替身。

    未给出注册表时使用 ``StaticModelResolver``，与生产一致地经注册表解析帧模型；
    未给出 LLM 服务时使用没有预置结果的替身，任何模型调用都显式失败。
    """
    config = config or HiveMemoryConfig()
    if model_registry is None:
        model_registry = StaticModelResolver()
    if llm_service is None:
        llm_service = ScriptedLLMService()
    runtime = build_agent_runtime(
        runtime_config=config.alice.runtime,
        koakuma_config=config.alice.koakuma,
        memory_compiler_config=config.memory_compiler,
        llm_service=llm_service,
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
        summarizer=RelaySummarizer(MTPGenerator(llm_service)),
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
