import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.alice.runtime.orchestrator import AgentOrchestrator
from hivememory.core.models import Identity, OMNI_DOLL_PROFILE, RuntimeScope, TurnEvent
from hivememory.core.mtp.models import MTPCallRequest
from hivememory.core.protocol.models import AgentRunStatus


def _frame(*, depth: int = 0, frame_id: str = "frame-main") -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id=frame_id, depth=depth),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic-1",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
    )


def _orchestrator(frame: ExecutionFrame, runtime=None) -> AgentOrchestrator:
    runtime = runtime or SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        collect_tasks_by_run=MagicMock(return_value=[]),
        aliases_by_frame=MagicMock(return_value=[]),
    )
    scheduler = SimpleNamespace(
        create_main_frame=MagicMock(return_value=frame),
        suspend_frame=MagicMock(),
        resume_frame=MagicMock(),
        fork_sub_frame=AsyncMock(),
    )
    profile_resolver = SimpleNamespace(resolve=AsyncMock(return_value=OMNI_DOLL_PROFILE))
    alias_resolver = SimpleNamespace(resolve=AsyncMock())
    orchestrator = AgentOrchestrator(
        agent_runtime=runtime,
        frame_scheduler=scheduler,
        agent_profile_resolver=profile_resolver,
        alias_resolver=alias_resolver,
    )
    orchestrator._test_runtime = runtime
    orchestrator._test_scheduler = scheduler
    orchestrator._test_profile_resolver = profile_resolver
    orchestrator._test_alias_resolver = alias_resolver
    return orchestrator


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
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)),
        collect_tasks_by_run=MagicMock(return_value=[]),
        cancel_tasks_by_run=MagicMock(return_value=[]),
        aliases_by_frame=MagicMock(return_value=[]),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=frame.identity,
        topic_id="topic-1",
    )

    assert result.final_text == "hello world"
    assert result.mtp_iterations == 2
    assert result.total_iterations == 3
    assert result.turn_events[0].kind == "user_message"
    assert result.turn_events[0].sequence == 0
    assert result.turn_events[0].content == "hello"
    assert result.turn_events[1].kind == "assistant_message"
    assert result.turn_events[1].sequence == 1
    assert result.turn_events == frame.progress.turn_events
    assert result.materialize_tasks == []
    runtime.collect_tasks_by_run.assert_called_once_with("run-1")
    runtime.cancel_tasks_by_run.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_cancelled_cancels_pending_atoms_without_materialize_tasks():
    frame = _frame()
    cancel_event = asyncio.Event()
    cancel_event.set()
    runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)),
        collect_tasks_by_run=MagicMock(return_value=["should-not-use"]),
        cancel_tasks_by_run=MagicMock(return_value=["draft_cancelled"]),
        aliases_by_frame=MagicMock(return_value=[]),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=frame.identity,
        topic_id="topic-1",
        cancel_event=cancel_event,
    )

    assert result.status == AgentRunStatus.CANCELLED.value
    assert result.materialize_tasks == []
    runtime.cancel_tasks_by_run.assert_called_once_with("run-1")
    runtime.collect_tasks_by_run.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_records_current_user_message_before_execution():
    frame = _frame()
    runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)),
        collect_tasks_by_run=MagicMock(return_value=[]),
        cancel_tasks_by_run=MagicMock(return_value=[]),
        aliases_by_frame=MagicMock(return_value=[]),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)

    result = await orchestrator.run_agent(
        messages=[
            {"role": "system", "content": "constraints"},
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ],
        identity=frame.identity,
        topic_id="topic-1",
    )

    assert [event.kind for event in result.turn_events] == ["user_message"]
    assert result.turn_events[0].sequence == 0
    assert result.turn_events[0].content == "current"
    assert frame.progress.sequence == 1


@pytest.mark.asyncio
async def test_run_agent_stream_records_current_user_message():
    frame = _frame()

    async def run_frame_stream(**_kwargs):
        if False:
            yield None

    runtime = SimpleNamespace(
        run_frame_stream=run_frame_stream,
        collect_tasks_by_run=MagicMock(return_value=[]),
        cancel_tasks_by_run=MagicMock(return_value=[]),
        aliases_by_frame=MagicMock(return_value=[]),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)

    events = []
    async for event in orchestrator.run_agent_stream(
        messages=[{"role": "user", "content": "stream current"}],
        identity=frame.identity,
        topic_id="topic-1",
    ):
        events.append(event)

    done = next(event for event in events if event["event"] == "done")
    assert done["data"]["turn_events"][0]["kind"] == "user_message"
    assert done["data"]["turn_events"][0]["sequence"] == 0
    assert done["data"]["turn_events"][0]["content"] == "stream current"


@pytest.mark.asyncio
async def test_handle_suspend_runs_sub_agent_and_appends_call_response():
    main_frame = _frame()
    main_frame.progress.iteration = 2
    main_frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="call",
            action_id="act-1",
            tool_kind="CALL",
            tool_name="helper",
            status="pending",
        )
    )
    sub_frame = _frame(depth=1, frame_id="frame-sub")
    sub_frame.progress.text_segments.append("sub reply")
    sub_frame.harvested_aliases.append("draft_sub")

    runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)),
        collect_tasks_by_run=MagicMock(return_value=[]),
        aliases_by_frame=MagicMock(return_value=["draft_runtime"]),
    )
    orchestrator = _orchestrator(main_frame, runtime=runtime)
    orchestrator._test_scheduler.fork_sub_frame.return_value = sub_frame

    engine_result = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="helper", task="summarize", context_refs=[]),
        suspend_assistant_text="<CALL helper>",
        suspend_action_id="act-1",
    )

    await orchestrator._handle_suspend(main_frame, engine_result, generation_options={"x": 1})

    orchestrator._test_scheduler.suspend_frame.assert_called_once_with(main_frame)
    orchestrator._test_scheduler.resume_frame.assert_called_once()
    runtime.run_frame.assert_awaited_once_with(
        frame=sub_frame,
        generation_options={"x": 1},
        cancel_event=None,
    )
    assert "draft_sub" in main_frame.harvested_aliases
    assert "draft_runtime" in sub_frame.harvested_aliases
    assert main_frame.progress.turn_events[0].status == "success"
    assert main_frame.progress.turn_events[-1].kind == "tool_result"
    assert main_frame.progress.turn_events[-1].status == "success"
    assert "sub reply" in main_frame.working_history[-1]["content"]


@pytest.mark.asyncio
async def test_handle_suspend_passes_cancel_event_to_sub_agent():
    main_frame = _frame()
    sub_frame = _frame(depth=1, frame_id="frame-sub")
    cancel_event = asyncio.Event()

    runtime = SimpleNamespace(
        run_frame=AsyncMock(return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)),
        collect_tasks_by_run=MagicMock(return_value=[]),
        cancel_tasks_by_run=MagicMock(return_value=[]),
        aliases_by_frame=MagicMock(return_value=[]),
    )
    orchestrator = _orchestrator(main_frame, runtime=runtime)
    orchestrator._test_scheduler.fork_sub_frame.return_value = sub_frame

    engine_result = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="helper", task="summarize", context_refs=[]),
    )

    await orchestrator._handle_suspend(
        main_frame,
        engine_result,
        generation_options={"x": 1},
        cancel_event=cancel_event,
    )

    runtime.run_frame.assert_awaited_once_with(
        frame=sub_frame,
        generation_options={"x": 1},
        cancel_event=cancel_event,
    )


@pytest.mark.asyncio
async def test_handle_suspend_emits_error_response_when_sub_agent_fails():
    main_frame = _frame()
    main_frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="call",
            action_id="act-1",
            tool_kind="CALL",
            tool_name="helper",
            status="pending",
        )
    )
    orchestrator = _orchestrator(main_frame)
    orchestrator._test_profile_resolver.resolve.side_effect = RuntimeError("missing helper")
    events = []

    async def emit(event):
        events.append(event)

    engine_result = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="helper", task="summarize"),
        suspend_action_id="act-1",
    )

    await orchestrator._handle_suspend(main_frame, engine_result, None, emit=emit)

    assert events[0]["event"] == "sub_agent_start"
    assert events[-1]["event"] == "sub_agent_end"
    assert events[-1]["data"]["status"] == "error"
    assert main_frame.progress.turn_events[-1].status == "error"
