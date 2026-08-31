"""
AgentRunService 结果组装集成测试 — 真实 RunExecutor/RunSession/StreamAdapter 协作

驱动 AgentRunService + service 内部真实构造的 RunExecutor + 真实 RunSession +
真实 AgentRunStreamAdapter 协作；仅 stub LLM 执行端口与 FrameFactory 边界。
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import FrameProducts, RuntimeProducts
from hivememory.alice.application.agent_run_service import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.core.models import OMNI_DOLL_PROFILE, TurnEvent
from hivememory.core.protocol.models import AgentRunContext, AgentRunStatus, RetrievalResponse
from hivememory.system.runtime.events import NullRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from tests.helpers.workspace import make_runtime_scope


def _frame(
    frame_id: str = "frame-main",
    messages: list[dict[str, str]] | None = None,
) -> ExecutionFrame:
    return FrameFactory().create(
        FrameSpec(
            runtime_scope=make_runtime_scope(run_id="run-1", frame_id=frame_id),
            profile=OMNI_DOLL_PROFILE,
            messages=(messages if messages is not None else [{"role": "user", "content": "hello"}]),
            topic_id="topic-1",
            execution_policy=FrameExecutionPolicy.from_profile(OMNI_DOLL_PROFILE),
        )
    )


def _context(frame: ExecutionFrame) -> AgentRunContext:
    return AgentRunContext(
        identity_scope=frame.identity_scope,
        interaction_id="interaction-test",
        topic_id=frame.topic_id,
        user_message="hello",
        topic_context=None,
        retrieval_result=RetrievalResponse(memories=[]),
        memory_context="",
        agent_profile=frame.agent_profile,
        storage_available=True,
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
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    factory = MagicMock(spec=FrameFactory)
    factory.scope.side_effect = FrameFactory.scope
    factory.create.return_value = frame
    service = AgentRunService(
        agent_runtime=resolved_agent_runtime,
        call_coordinator=MagicMock(),
        frame_factory=factory,
        prompt_assembler=MagicMock(
            build_main_agent_messages=MagicMock(return_value=frame.working_history)
        ),
        atom_cache=MagicMock(),
        stream_adapter=AgentRunStreamAdapter(),
        agent_run_events=AgentRunEventEmitter(RuntimeEventPublisher(NullRuntimeEventSink())),
    )
    run_session = session or RunSession(agent_run_id="run-1", generation_id="generation-1")
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
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    service, session = _runtime_for_frame(frame, agent_runtime)

    result = await service.run_agent(_context(frame))

    assert result.final_text == "hello world"
    assert result.mtp_iterations == 2
    assert result.total_iterations == 3
    assert [event.kind for event in result.turn_events] == ["user_message", "assistant_message"]
    agent_runtime.finalize_run.assert_called_once()
    assert agent_runtime.finalize_run.call_args.args[0] == "run-1"
    assert agent_runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.COMPLETED
    assert session.frames == {"frame-main": frame}


@pytest.mark.asyncio
async def test_run_agent_cancellation_unwinds_and_propagates():
    frame = _frame()
    started = asyncio.Event()

    async def run_frame(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    agent_runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=run_frame,
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    service, _ = _runtime_for_frame(frame, agent_runtime)

    task = asyncio.create_task(service.run_agent(_context(frame)))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    agent_runtime.finalize_run.assert_called_once()
    assert agent_runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_agent_budget_exhaustion_maps_to_failed_run():
    frame = _frame()
    agent_runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    service, _ = _runtime_for_frame(frame, agent_runtime)

    result = await service.run_agent(_context(frame))

    assert result.status == AgentRunStatus.FAILED.value
    assert result.materialize_tasks == []
    assert agent_runtime.finalize_run.call_args.args[1].status == (
        FrameExecutionStatus.BUDGET_EXHAUSTED
    )


@pytest.mark.asyncio
async def test_run_agent_stream_done_preserves_failed_terminal_status():
    frame = _frame()
    agent_runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    service, _ = _runtime_for_frame(frame, agent_runtime)

    events = [event async for event in service.run_agent_stream(_context(frame))]

    done = next(event for event in events if event["event"] == "done")
    assert done["data"]["status"] == AgentRunStatus.FAILED.value
    assert done["data"]["agent_run_id"] == "run-1"
    assert done["data"]["frame_id"] == "frame-main"


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

    result = await service.run_agent(_context(frame))

    assert [event.kind for event in result.turn_events] == ["user_message"]
    assert result.turn_events[0].content == "current"
    assert result.turn_events[0].sequence == 0
