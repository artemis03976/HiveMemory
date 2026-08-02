import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.products import FrameProducts, RuntimeProducts
from hivememory.alice.runtime.agent.frame_factory import FrameFactory
from hivememory.alice.runtime.agent.run_session import RunSession
from hivememory.alice.runtime.orchestrator import AgentOrchestrator
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope, TurnEvent
from hivememory.core.protocol.models import AgentRunStatus


def _frame(frame_id: str = "frame-main") -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic-1",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
    )


def _orchestrator(frame: ExecutionFrame, runtime=None) -> AgentOrchestrator:
    runtime = runtime or SimpleNamespace(
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
    profile_resolver = SimpleNamespace(resolve=AsyncMock(return_value=OMNI_DOLL_PROFILE))
    alias_resolver = SimpleNamespace(resolve=AsyncMock())
    return AgentOrchestrator(
        agent_runtime=runtime,
        agent_profile_resolver=profile_resolver,
        alias_resolver=alias_resolver,
        frame_factory=factory,
        prompt_assembler=MagicMock(),
    )


def _session(*, cancel_event: asyncio.Event | None = None) -> RunSession:
    return RunSession(
        agent_run_id="run-1",
        generation_id="generation-1",
        cancel_event=cancel_event if cancel_event is not None else asyncio.Event(),
    )


@pytest.mark.asyncio
async def test_run_agent_assembles_result_from_completed_frame():
    frame = _frame()
    frame.progress.text_segments.extend(["hello", " world"])
    frame.progress.iteration = 3
    frame.progress.turn_events.append(
        TurnEvent(
            kind="assistant_message",
            sequence=0,
            role="assistant",
            content="hello world",
        )
    )
    runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)
    session = _session()

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=frame.identity,
        topic_id="topic-1",
        session=session,
    )

    assert result.final_text == "hello world"
    assert result.mtp_iterations == 2
    assert result.total_iterations == 3
    assert [event.kind for event in result.turn_events] == ["user_message", "assistant_message"]
    runtime.finalize_run.assert_called_once()
    assert runtime.finalize_run.call_args.args[0] == "run-1"
    assert runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.COMPLETED
    assert session.frames == {"frame-main": frame}


@pytest.mark.asyncio
async def test_run_agent_cancelled_does_not_materialize_runtime_products():
    frame = _frame()
    cancel_event = asyncio.Event()
    cancel_event.set()
    runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)
    session = _session(cancel_event=cancel_event)

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=frame.identity,
        topic_id="topic-1",
        session=session,
    )

    assert result.status == AgentRunStatus.CANCELLED.value
    assert result.materialize_tasks == []
    assert runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_agent_budget_exhaustion_maps_to_failed_run():
    frame = _frame()
    runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)
    session = _session()

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=frame.identity,
        topic_id="topic-1",
        session=session,
    )

    assert result.status == AgentRunStatus.FAILED.value
    assert result.materialize_tasks == []
    assert runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_run_agent_stream_done_preserves_failed_terminal_status():
    frame = _frame()
    runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)
    session = _session()

    events = [
        event
        async for event in orchestrator.run_agent_stream(
            messages=[{"role": "user", "content": "stream"}],
            identity=frame.identity,
            topic_id="topic-1",
            session=session,
        )
    ]

    done = next(event for event in events if event["event"] == "done")
    assert done["data"]["status"] == AgentRunStatus.FAILED.value
    assert done["data"]["agent_run_id"] == "run-1"
    assert done["data"]["frame_id"] == "frame-main"


@pytest.mark.asyncio
async def test_run_agent_records_latest_user_message_once():
    frame = _frame()
    runtime = SimpleNamespace(
        max_iterations=5,
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)
    session = _session()

    result = await orchestrator.run_agent(
        messages=[
            {"role": "system", "content": "constraints"},
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ],
        identity=frame.identity,
        topic_id="topic-1",
        session=session,
    )

    assert [event.kind for event in result.turn_events] == ["user_message"]
    assert result.turn_events[0].content == "current"
    assert result.turn_events[0].sequence == 0
