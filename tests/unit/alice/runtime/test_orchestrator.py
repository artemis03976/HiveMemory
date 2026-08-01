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
from hivememory.alice.runtime.orchestrator import AgentOrchestrator
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    Identity,
    RuntimeScope,
    TurnEvent,
)
from hivememory.core.mtp.exceptions import AliasNotFoundError
from hivememory.core.mtp.models import MTPCallRequest
from hivememory.core.protocol.models import AgentRunStatus
from hivememory.system.model_registry import ModelNotFoundError


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
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
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
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
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
    runtime.finalize_run.assert_called_once()
    assert runtime.finalize_run.call_args.args[0] == "run-1"
    assert runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_agent_cancelled_cancels_pending_atoms_without_materialize_tasks():
    frame = _frame()
    cancel_event = asyncio.Event()
    cancel_event.set()
    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
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
    runtime.finalize_run.assert_called_once()
    assert runtime.finalize_run.call_args.args[0] == "run-1"
    assert runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_run_agent_budget_exhaustion_is_failed_and_cleans_pending_atoms():
    frame = _frame()
    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=frame.identity,
        topic_id="topic-1",
    )

    assert result.status == AgentRunStatus.FAILED.value
    assert result.materialize_tasks == []
    runtime.finalize_run.assert_called_once()
    assert runtime.finalize_run.call_args.args[0] == "run-1"
    assert runtime.finalize_run.call_args.args[1].status == FrameExecutionStatus.BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_run_agent_stream_done_preserves_failed_terminal_status():
    frame = _frame()

    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.BUDGET_EXHAUSTED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(frame, runtime=runtime)

    events = [
        event
        async for event in orchestrator.run_agent_stream(
            messages=[{"role": "user", "content": "stream"}],
            identity=frame.identity,
            topic_id="topic-1",
        )
    ]

    done = next(event for event in events if event["event"] == "done")
    assert done["data"]["status"] == AgentRunStatus.FAILED.value


@pytest.mark.asyncio
async def test_run_agent_records_current_user_message_before_execution():
    frame = _frame()
    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
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

    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
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
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(
            return_value=FrameProducts(artifact_aliases=("draft_sub", "draft_runtime"))
        ),
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
    assert "draft_runtime" in main_frame.harvested_aliases
    runtime.finalize_frame.assert_called_once()
    assert runtime.finalize_frame.call_args.args[0] is sub_frame
    assert runtime.finalize_frame.call_args.args[1].status == FrameExecutionStatus.COMPLETED
    assert main_frame.progress.turn_events[0].status == "success"
    assert main_frame.progress.turn_events[-1].kind == "tool_result"
    assert main_frame.progress.turn_events[-1].status == "success"
    assert "sub reply" in main_frame.working_history[-1]["content"]
    orchestrator._test_profile_resolver.resolve.assert_awaited_once_with(
        "helper",
        identity=main_frame.identity,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_status", "expected_call_status", "error_code"),
    [
        (FrameExecutionStatus.CANCELLED, "cancelled", None),
        (
            FrameExecutionStatus.BUDGET_EXHAUSTED,
            "error",
            "mtp.call_response.budget_exhausted",
        ),
        (
            FrameExecutionStatus.SUSPENDED,
            "error",
            "mtp.call_response.unexpected_suspend",
        ),
        (FrameExecutionStatus.FAILED, "error", "mtp.call_response.sub_agent_error"),
    ],
)
async def test_handle_suspend_maps_non_completed_child_terminal_status(
    terminal_status,
    expected_call_status,
    error_code,
):
    main_frame = _frame()
    main_frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="call",
            action_id="act-terminal",
            tool_kind="CALL",
            tool_name="helper",
            status="pending",
        )
    )
    sub_frame = _frame(depth=1, frame_id="frame-sub-terminal")
    sub_frame.progress.text_segments.append("partial reply")
    sub_frame.harvested_aliases.append("draft_should_not_harvest")
    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(
                status=terminal_status,
                error=(
                    RuntimeError("child failed")
                    if terminal_status == FrameExecutionStatus.FAILED
                    else None
                ),
            )
        ),
        run_frame_emitting=AsyncMock(
            return_value=FrameExecutionResult(
                status=terminal_status,
                error=(
                    RuntimeError("child failed")
                    if terminal_status == FrameExecutionStatus.FAILED
                    else None
                ),
            )
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(main_frame, runtime=runtime)
    orchestrator._test_scheduler.fork_sub_frame.return_value = sub_frame
    events = []

    async def emit(event):
        events.append(event)

    await orchestrator._handle_suspend(
        main_frame,
        FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="helper", task="summarize"),
            suspend_action_id="act-terminal",
        ),
        generation_options=None,
        emit=emit,
    )

    assert main_frame.progress.turn_events[0].status == expected_call_status
    assert main_frame.progress.turn_events[-1].status == expected_call_status
    assert "draft_should_not_harvest" not in main_frame.harvested_aliases
    assert events[-1]["data"]["terminal_status"] == terminal_status.value
    assert events[-1]["data"]["status"] == expected_call_status
    if error_code is not None:
        assert error_code in main_frame.working_history[-1]["content"]
    runtime.finalize_frame.assert_called_once()
    assert runtime.finalize_frame.call_args.args[0] is sub_frame
    assert runtime.finalize_frame.call_args.args[1].status == terminal_status


@pytest.mark.asyncio
async def test_handle_suspend_streaming_child_uses_terminal_result():
    main_frame = _frame()
    main_frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="call",
            action_id="act-stream-terminal",
            tool_kind="CALL",
            tool_name="helper",
            status="pending",
        )
    )
    sub_frame = _frame(depth=1, frame_id="frame-sub-stream-terminal")
    runtime = SimpleNamespace(
        run_frame_emitting=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.CANCELLED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(main_frame, runtime=runtime)
    orchestrator._test_scheduler.fork_sub_frame.return_value = sub_frame
    events = []

    async def emit(event):
        events.append(event)

    await orchestrator._handle_suspend(
        main_frame,
        FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="helper", task="summarize"),
            suspend_action_id="act-stream-terminal",
        ),
        generation_options=None,
        emit=emit,
    )

    runtime.run_frame_emitting.assert_awaited_once()
    assert main_frame.progress.turn_events[0].status == "cancelled"
    assert main_frame.progress.turn_events[-1].status == "cancelled"
    assert events[-1]["data"]["terminal_status"] == "cancelled"


@pytest.mark.asyncio
async def test_handle_suspend_passes_cancel_event_to_sub_agent():
    main_frame = _frame()
    sub_frame = _frame(depth=1, frame_id="frame-sub")
    cancel_event = asyncio.Event()

    runtime = SimpleNamespace(
        run_frame=AsyncMock(
            return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
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

    assert [event["event"] for event in events] == ["sub_agent_end"]
    assert events[0]["data"]["status"] == "error"
    assert events[0]["data"]["frame_id"] is None
    assert main_frame.progress.turn_events[0].status == "error"
    assert main_frame.progress.turn_events[-1].status == "error"


@pytest.mark.asyncio
async def test_handle_suspend_preserves_explicit_profile_not_found_error():
    main_frame = _frame()
    main_frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="call",
            action_id="act-1",
            tool_kind="CALL",
            tool_name="missing_doll",
            status="pending",
        )
    )
    orchestrator = _orchestrator(main_frame)
    orchestrator._test_profile_resolver.resolve.side_effect = AliasNotFoundError(
        message_key="mtp.call.profile_not_found",
        params={"agent_alias": "missing_doll"},
    )
    engine_result = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="missing_doll", task="summarize"),
        suspend_action_id="act-1",
    )

    await orchestrator._handle_suspend(main_frame, engine_result, None)

    content = main_frame.working_history[-1]["content"]
    assert 'code="mtp.alias.not_found"' in content
    assert "missing_doll" in content
    assert main_frame.progress.turn_events[0].status == "error"
    assert main_frame.progress.turn_events[-1].status == "error"
    orchestrator._test_scheduler.fork_sub_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_suspend_returns_stable_model_unavailable_error():
    main_frame = _frame()
    main_frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="call",
            action_id="act-model",
            tool_kind="CALL",
            tool_name="model_doll",
            status="pending",
        )
    )
    runtime = SimpleNamespace(
        run_frame=AsyncMock(),
        run_frame_emitting=AsyncMock(
            side_effect=ModelNotFoundError("missing-model is not registered")
        ),
        finalize_run=MagicMock(return_value=RuntimeProducts()),
        finalize_frame=MagicMock(return_value=FrameProducts()),
    )
    orchestrator = _orchestrator(main_frame, runtime=runtime)
    profile = AgentProfile(model_name="missing-model")
    orchestrator._test_profile_resolver.resolve.return_value = profile
    orchestrator._test_scheduler.fork_sub_frame.return_value = _frame(
        depth=1,
        frame_id="frame-sub-model",
    )
    events = []

    async def emit(event):
        events.append(event)

    engine_result = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="model_doll", task="summarize"),
        suspend_action_id="act-model",
    )

    await orchestrator._handle_suspend(main_frame, engine_result, None, emit=emit)

    content = main_frame.working_history[-1]["content"]
    assert 'code="mtp.system.service_unavailable"' in content
    assert "missing-model" in content
    assert events[-1]["data"]["error_code"] == "mtp.system.service_unavailable"
    assert main_frame.progress.turn_events[0].status == "error"
    assert main_frame.progress.turn_events[-1].status == "error"
