"""交互输出端口：帧输出、CALL 边界与压缩进度投影为交互事件，经 CPU 事件 sink 推送。"""

from __future__ import annotations

from typing import Any

import pytest

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionPolicy,
)
from hivememory.agent_runtime.output import MTPFinished, MTPStarted, TokenDelta
from hivememory.alice.interface.output import SinkRunOutput
from hivememory.alice.orchestration.output import CallOutputFinished, CallOutputStarted
from hivememory.core.models import OMNI_DOLL_PROFILE
from tests.helpers.call import StubOperationSubmitter
from tests.helpers.workspace import make_runtime_scope


def _frame() -> ExecutionFrame:
    return ExecutionFrame(
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        submit_operation=StubOperationSubmitter(),
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
        language="zh",
    )


def _recording_output() -> tuple[SinkRunOutput, list[dict[str, Any]]]:
    received: list[dict[str, Any]] = []

    async def sink(event: dict[str, Any]) -> None:
        received.append(event)

    output = SinkRunOutput(sink)
    output.bind_run("run-1")
    return output, received


@pytest.mark.asyncio
async def test_output_projects_typed_frame_outputs_to_compatible_events() -> None:
    output, received = _recording_output()
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

    assert [event["event"] for event in received] == ["token", "mtp_start", "mtp_result"]
    assert [event["data"]["stream_sequence"] for event in received] == [0, 1, 2]
    assert all(event["data"]["agent_run_id"] == "run-1" for event in received)
    assert all(event["data"]["frame_id"] == "frame-1" for event in received)
    assert all(event["data"]["scope"] == "sub" for event in received)
    assert received[0]["data"]["action_id"] == "call-1"
    assert received[1]["data"]["action_id"] == "action-2"


@pytest.mark.asyncio
async def test_output_projects_call_boundaries_with_run_correlation() -> None:
    output, received = _recording_output()

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

    started, finished = received
    assert started["event"] == "sub_agent_start"
    assert finished["event"] == "sub_agent_end"
    assert started["data"]["agent_run_id"] == "run-1"
    assert finished["data"]["agent_run_id"] == "run-1"
    assert started["data"]["frame_id"] == "frame-child"
    assert finished["data"]["terminal_status"] == "completed"
    assert [started["data"]["stream_sequence"], finished["data"]["stream_sequence"]] == [0, 1]


@pytest.mark.asyncio
async def test_output_pushes_compaction_progress_with_shared_sequence() -> None:
    """压缩进度与其他交互事件共用一条序号，内容为状态与进度数据。"""
    output, received = _recording_output()

    await output.context_compaction(status="started", data={"layer": "reduction"})
    await output.for_frame(_frame(), action_id=None, scope="main", depth=0).send(
        TokenDelta(content="hi")
    )

    assert received[0] == {
        "event": "context_compaction",
        "data": {
            "agent_run_id": "run-1",
            "status": "started",
            "layer": "reduction",
            "stream_sequence": 0,
        },
    }
    assert received[1]["data"]["stream_sequence"] == 1
