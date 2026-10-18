"""AgentRunService 的固定输入预算、失败原因投影。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.model.budget import TurnContextBudgetExceededError
from hivememory.agent_runtime.model.client import LLMClient
from hivememory.agent_runtime.runtime import AgentRuntime, build_agent_runtime
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.events import AgentRunEventEmitter
from hivememory.alice.orchestration.call import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.run.output import NullRunOutput
from hivememory.alice.run.service import AgentRunService
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
)
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.errors import ModelContextWindowExceededError
from hivememory.core.models import (
    AgentProfile,
)
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.workspace import make_execution_labels


def _build_input_manifest():
    return make_input_manifest(
        labels=make_execution_labels(agent_id="omni_doll"),
        process_id="process-test",
        topic_id="topic_1",
        user_message="hello",
        turn_context="ctx",
    )


def _build_service(*, runtime_events=None, config=None) -> tuple[AgentRuntime, AgentRunService]:
    config = config or HiveMemoryConfig()
    runtime = build_agent_runtime(
        runtime_config=config.alice.runtime,
        koakuma_config=config.alice.koakuma,
        memory_compiler_config=config.memory_compiler,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,has_teaching",
    [("enabled", True), ("koakuma_disabled", False), ("teaching_disabled", False)],
)
async def test_root_uses_compiled_blocks_and_language_with_its_own_teaching(mode, has_teaching):
    """根帧直接采用清单文本与语言；工具开关只影响教学，不重渲染 Profile。"""
    config = HiveMemoryConfig()
    config.alice.koakuma.enabled = mode != "koakuma_disabled"
    config.alice.koakuma.mtp_prompt.enabled = mode != "teaching_disabled"
    runtime, service = _build_service(config=config)
    runtime.run_frame = AsyncMock(
        return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    )
    manifest = make_input_manifest(
        agent_profile=AgentProfile(persona="DO NOT RENDER", language="zh"),
        language="en",
        instructions=" \nCOMPILED NOTICE AND PERSONA\n ",
        turn_context="\nCOMPILED MATERIAL\n ",
        allowed_operations=frozenset(),
    )
    result = await service.run_agent(
        manifest,
        run_output=NullRunOutput(),
        submit_operation=MagicMock(),
    )
    frame = runtime.run_frame.await_args.args[0]
    system = frame.working_history[0]["content"]
    fixed = " \nCOMPILED NOTICE AND PERSONA\n \n\n\nCOMPILED MATERIAL\n "
    assert result.status == "completed"
    # 执行期的 MTP 回填沿用清单语言，而不是 Profile 的语言设置。
    assert frame.language == "en"
    assert frame.execution_policy.permitted_verbs == frozenset({"RUN"})
    assert frame.working_history[-1] == {"role": "user", "content": "hello"}
    assert "DO NOT RENDER" not in system
    assert system.endswith(fixed)
    teaching = system.removesuffix(fixed)
    assert ("You are an intelligent Agent running on HiveMemory" in teaching) == has_teaching
    assert ("- RUN:" in teaching) == has_teaching
    assert "- CALL:" not in teaching
    assert bool(teaching) == has_teaching


@pytest.mark.asyncio
async def test_fixed_input_over_budget_fails_without_calling_model():
    """固定用户输入已装不下时明确失败，初始用户事件不算输出。"""
    runtime, service = _build_service()
    manifest = _build_input_manifest().model_copy(update={"user_message": "x" * 140000})
    runtime.run_frame = AsyncMock(side_effect=AssertionError("不应调用模型"))
    from hivememory.workspace.contracts import CPUResultSlot

    slot = CPUResultSlot()
    result = await service.run_agent(
        manifest, run_output=NullRunOutput(), submit_operation=MagicMock(), result_slot=slot
    )
    assert result.status == "failed"
    assert result.failure_reason == "input_too_large"
    assert result.has_output is False
    assert slot.result == result
    runtime.run_frame.assert_not_awaited()


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
    runtime.run_frame = AsyncMock(
        return_value=FrameExecutionResult(status=FrameExecutionStatus.FAILED, error=error)
    )
    result = await service.run_agent(
        _build_input_manifest(),
        run_output=NullRunOutput(),
        submit_operation=MagicMock(),
    )
    assert result.status == "failed"
    assert result.failure_reason == reason
    assert "private detail" not in result.model_dump_json()
