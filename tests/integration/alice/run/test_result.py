"""
AgentRunService 结果组装集成测试 — 真实 RunExecutor/RunSession 协作

驱动 AgentRunService + service 内部真实构造的 RunExecutor + 真实 RunSession
协作；仅 stub LLM 执行端口与 FrameFactory 边界。取消转换为结果的行为由运行
服务的集成测试覆盖。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
    FrameProducts,
)
from hivememory.agent_runtime.model.generator import MTPGenerator
from hivememory.agent_runtime.model.resolution import ResolvedFrameModel
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.events import AgentRunEventEmitter
from hivememory.alice.orchestration.frames import FrameFactory, FrameSpec
from hivememory.alice.orchestration.session import RunSession
from hivememory.alice.run.service import AgentRunService
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.alice import CompactionConfig, KoakumaConfig
from hivememory.core.models import OMNI_DOLL_PROFILE, TurnEvent
from hivememory.workspace.contracts import CPUExecutionStatus
from tests.helpers.call import StubOperationSubmitter
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.outputs import NullRunOutput
from tests.helpers.workspace import make_runtime_scope


def _frame(
    frame_id: str = "frame-main",
    messages: list[dict[str, str]] | None = None,
) -> ExecutionFrame:
    return FrameFactory().create(
        FrameSpec(
            submit_operation=StubOperationSubmitter(),
            runtime_scope=make_runtime_scope(run_id="run-1", frame_id=frame_id),
            profile=OMNI_DOLL_PROFILE,
            messages=(messages if messages is not None else [{"role": "user", "content": "hello"}]),
            topic_id="topic-1",
            execution_policy=FrameExecutionPolicy(permitted_verbs=frozenset()),
            language="zh",
        )
    )


def _context(frame: ExecutionFrame):
    return make_input_manifest(
        labels=frame.runtime_scope.labels,
        process_id="interaction-test",
        topic_id=frame.topic_id,
        user_message="hello",
        agent_profile=frame.agent_profile,
    )


def _runtime_for_frame(
    frame: ExecutionFrame,
    agent_runtime=None,
    *,
    session: RunSession | None = None,
) -> tuple[AgentRunService, RunSession]:
    resolved_agent_runtime = agent_runtime or SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    resolved_agent_runtime.resolve_model = lambda *_args: ResolvedFrameModel(
        params=DEFAULT_LLM_PARAMS, input_token_limit=65536
    )
    factory = MagicMock(spec=FrameFactory)
    factory.scope.side_effect = FrameFactory.scope
    factory.create.return_value = frame
    service = AgentRunService(
        summarizer=RelaySummarizer(MTPGenerator(ScriptedLLMService())),
        agent_runtime=resolved_agent_runtime,
        call_coordinator=MagicMock(),
        frame_factory=factory,
        koakuma_config=KoakumaConfig(enabled=False),
        compaction_config=CompactionConfig(),
        agent_run_events=AgentRunEventEmitter(RuntimeEventPublisher(NullRuntimeEventSink())),
    )
    run_session = session or RunSession(agent_run_id="run-1", process_id="process-1")
    service._create_run_session = MagicMock(return_value=run_session)
    return service, run_session


@pytest.mark.asyncio
async def test_run_agent_assembles_result_from_completed_frame():
    frame = _frame()
    frame.progress.text_segments.extend(["hello", " world"])
    frame.progress.iteration = 3
    frame.progress.turn_events.append(
        TurnEvent(
            kind="assistant_message",
            sequence=frame.progress.sequence,
            role="assistant",
            content="hello world",
        )
    )
    frame.progress.sequence += 1
    agent_runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    service, session = _runtime_for_frame(frame, agent_runtime)

    result = await service.run_agent(
        _context(frame), run_output=NullRunOutput(), submit_operation=MagicMock()
    )

    assert result.final_text == "hello world"
    assert [event.kind for event in result.turn_events] == ["user_message", "assistant_message"]
    assert session.frames == {"frame-main": frame}


@pytest.mark.asyncio
async def test_run_agent_budget_exhaustion_maps_to_failed_run():
    frame = _frame()
    agent_runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    service, _ = _runtime_for_frame(frame, agent_runtime)

    result = await service.run_agent(
        _context(frame), run_output=NullRunOutput(), submit_operation=MagicMock()
    )

    assert result.status == CPUExecutionStatus.FAILED.value


@pytest.mark.asyncio
async def test_run_agent_preserves_factory_initialized_turn_events():
    messages = [
        {"role": "system", "content": "constraints"},
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current"},
    ]
    frame = _frame(messages=messages)
    service, _ = _runtime_for_frame(frame)

    result = await service.run_agent(
        _context(frame), run_output=NullRunOutput(), submit_operation=MagicMock()
    )

    assert [event.kind for event in result.turn_events] == ["user_message"]
    assert result.turn_events[0].content == "current"
    assert result.turn_events[0].sequence == 0
