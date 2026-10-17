"""AgentRunService 的固定输入预算、失败原因投影与缺失执行终态防御。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hivememory.agent_runtime.context_budget import (
    ModelContextWindowExceededError,
    TurnContextBudgetExceededError,
)
from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.application.agent_run_service import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RecordingRuntimeEventSink,
)
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_execution_labels


def _build_memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(
            source_agent_id="agent-1",
            user_id="u1",
            confidence_score=0.9,
        ),
        index=IndexLayer(
            title="test memory",
            summary="summary text",
            tags=["tag"],
            memory_type=MemoryType.FACT,
            alias="mem_alias",
        ),
        payload=PayloadLayer(content="memory content"),
    )


def _build_input_manifest(memory: MemoryAtom):
    return make_input_manifest(
        labels=make_execution_labels(agent_id="omni_doll"),
        process_id="process-test",
        topic_id="topic_1",
        user_message="hello",
        memories=[memory],
        memory_context="ctx",
    )


def _build_service(*, runtime_events=None) -> tuple[AliceRuntime, AgentRunService]:
    config = HiveMemoryConfig()
    runtime = AliceRuntime(
        alice_config=config.alice,
        memory_compiler_config=config.memory_compiler,
    )
    frame_factory = FrameFactory()
    prompt_assembler = AgentPromptAssembler(config.alice.koakuma)
    coordinator = CallCoordinator(
        runtime.agent_runtime,
        CallContextProvider(),
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
    )
    service = AgentRunService(
        agent_runtime=runtime.agent_runtime,
        call_coordinator=coordinator,
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
        stream_adapter=AgentRunStreamAdapter(),
        agent_run_events=AgentRunEventEmitter(
            RuntimeEventPublisher(runtime_events or NullRuntimeEventSink())
        ),
    )
    return runtime, service


@pytest.mark.asyncio
async def test_run_agent_stream_without_executor_terminal_fails_cleanly():
    recorder = RecordingRuntimeEventSink()
    _runtime, service = _build_service(runtime_events=recorder)
    manifest = _build_input_manifest(_build_memory_atom())
    executor = MagicMock()
    executor.run = AsyncMock(
        return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    )
    executor.terminal_result = None
    executor.runtime_products = None
    with patch(
        "hivememory.alice.application.agent_run_service.RunExecutor",
        return_value=executor,
    ):
        with pytest.raises(RuntimeError, match="ended without done"):
            async for _ in service.run_agent(manifest, stream=True, submit_operation=MagicMock()):
                pass

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].message == "Agent stream ended without done event."


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_fixed_input_over_budget_fails_without_calling_model(stream):
    """固定用户输入已装不下时两种交付均明确失败，初始用户事件不算输出。"""
    runtime, service = _build_service()
    manifest = _build_input_manifest(_build_memory_atom()).model_copy(
        update={"user_message": "x" * 140000}
    )
    runtime.agent_runtime.run_frame = AsyncMock(side_effect=AssertionError("不应调用模型"))
    from hivememory.workspace.contracts import CPUExecutionResult, CPUResultSlot

    slot = CPUResultSlot()
    if stream:
        events = [
            event
            async for event in service.run_agent(
                manifest, stream=True, submit_operation=MagicMock(), result_slot=slot
            )
        ]
        result = CPUExecutionResult(**events[-1]["data"])
        assert [event["event"] for event in events] == ["done"]
    else:
        result = await service.run_agent(
            manifest, stream=False, submit_operation=MagicMock(), result_slot=slot
        )
    assert result.status == "failed"
    assert result.failure_reason == "input_too_large"
    assert result.has_output is False
    assert slot.result == result
    runtime.agent_runtime.run_frame.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TurnContextBudgetExceededError(100, 90), "turn_context_budget_exceeded"),
        (ModelContextWindowExceededError("private detail"), "model_context_window_exceeded"),
        (RuntimeError("private detail"), None),
    ],
)
async def test_frame_error_maps_only_stable_cpu_reason(error, reason):
    """只有已知的上下文失败产生公开原因码，内部异常正文不进入执行结果。"""
    runtime, service = _build_service()
    runtime.agent_runtime.run_frame = AsyncMock(
        return_value=FrameExecutionResult(status=FrameExecutionStatus.FAILED, error=error)
    )
    result = await service.run_agent(
        _build_input_manifest(_build_memory_atom()), stream=False, submit_operation=MagicMock()
    )
    assert result.status == "failed"
    assert result.failure_reason == reason
    assert "private detail" not in result.model_dump_json()
