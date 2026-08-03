from __future__ import annotations

import asyncio

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
)
from hivememory.agent_runtime.output import MTPFinished, MTPStarted, TokenDelta
from hivememory.alice.orchestration.run_output import CallOutputFinished, CallOutputStarted
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.runtime.streaming import AgentRunStream, QueueAgentRunOutput
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope


def _frame() -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
        identity=Identity(user_id="user-1", agent_id="omni_doll"),
    )


@pytest.mark.asyncio
async def test_queue_output_projects_typed_frame_outputs_to_compatible_events() -> None:
    output = QueueAgentRunOutput("run-1")
    sink = output.for_frame(_frame(), action_id="call-1", scope="sub", depth=1)

    await sink.send(TokenDelta(content="hello"))
    await sink.send(
        MTPStarted(
            verb="READ",
            target="fact_1",
            args={"limit": 1},
            raw_text="<< READ | fact_1 >>",
            iteration=2,
            action_id="action-2",
        )
    )
    await sink.send(
        MTPFinished(
            verb="READ",
            target="fact_1",
            args={"limit": 1},
            raw_text="<< READ | fact_1 >>",
            status="success",
            iteration=2,
            action_id="action-2",
        )
    )

    events = [await output.receive() for _ in range(3)]

    assert [event["event"] for event in events] == ["token", "mtp_start", "mtp_result"]
    assert [event["data"]["stream_sequence"] for event in events] == [0, 1, 2]
    assert all(event["data"]["agent_run_id"] == "run-1" for event in events)
    assert all(event["data"]["frame_id"] == "frame-1" for event in events)
    assert all(event["data"]["scope"] == "sub" for event in events)
    assert events[0]["data"]["action_id"] == "call-1"
    assert events[1]["data"]["action_id"] == "action-2"


@pytest.mark.asyncio
async def test_queue_output_projects_call_boundaries_with_run_correlation() -> None:
    output = QueueAgentRunOutput("run-1")

    await output.call_started(
        CallOutputStarted(
            agent_id="helper",
            task="summarize",
            iteration=1,
            action_id="action-1",
            frame_id="frame-child",
        )
    )
    await output.call_finished(
        CallOutputFinished(
            status="success",
            final_text="done",
            iteration=1,
            action_id="action-1",
            frame_id="frame-child",
            agent_id="helper",
            terminal_status="completed",
        )
    )

    started = await output.receive()
    finished = await output.receive()

    assert started["event"] == "sub_agent_start"
    assert finished["event"] == "sub_agent_end"
    assert started["data"]["agent_run_id"] == "run-1"
    assert finished["data"]["agent_run_id"] == "run-1"
    assert started["data"]["frame_id"] == "frame-child"
    assert finished["data"]["terminal_status"] == "completed"
    assert [started["data"]["stream_sequence"], finished["data"]["stream_sequence"]] == [
        0,
        1,
    ]


@pytest.mark.asyncio
async def test_agent_run_stream_propagates_runner_error_after_queued_outputs() -> None:
    session = RunSession(agent_run_id="run-1")
    stream = AgentRunStream(session)

    async def runner() -> FrameExecutionResult:
        await stream.output.send_event("token", {"content": "before-error"})
        raise RuntimeError("runner failed")

    events = stream.events(runner())

    assert (await anext(events))["data"]["content"] == "before-error"
    with pytest.raises(RuntimeError, match="runner failed"):
        await anext(events)


@pytest.mark.asyncio
async def test_agent_run_stream_wakes_consumer_when_runner_is_cancelled() -> None:
    session = RunSession(agent_run_id="run-1")
    stream = AgentRunStream(session)

    async def runner() -> FrameExecutionResult:
        raise asyncio.CancelledError

    events = stream.events(runner())

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(anext(events), timeout=1)
    assert session.cancel_event.is_set()
