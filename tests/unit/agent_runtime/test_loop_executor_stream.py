import asyncio
from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.events import QueueFrameEventSink
from hivememory.agent_runtime.loop_executor import AgentLoopExecutor
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionStatus,
    GenerationResult,
    StreamChunk,
)
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope


def _frame() -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "task"}],
        topic_id="topic-1",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
    )


@pytest.mark.asyncio
async def test_stream_sink_receives_tokens_with_ordered_metadata() -> None:
    async def generate_stream(_messages, **_kwargs):
        yield StreamChunk(delta="partial", full_text="partial")
        yield StreamChunk(
            is_final=True,
            result=GenerationResult(text="complete", finish_reason="stop"),
        )

    worker_agent = MagicMock(generate_stream=generate_stream)
    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=MagicMock(),
        config=MagicMock(max_loop_iterations=2),
    )
    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=2)
    sink = QueueFrameEventSink(
        queue,
        metadata={
            "agent_run_id": "run-1",
            "frame_id": "frame-1",
            "action_id": None,
        },
    )

    result = await executor.execute_frame(_frame(), max_iterations=2, event_sink=sink)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert queue.get_nowait() == {
        "event": "token",
        "data": {
            "content": "partial",
            "agent_run_id": "run-1",
            "frame_id": "frame-1",
            "action_id": None,
            "stream_sequence": 0,
        },
    }
    assert sink.next_sequence == 1


@pytest.mark.asyncio
async def test_streaming_without_final_chunk_returns_failed_frame_result() -> None:
    async def generate_stream(_messages, **_kwargs):
        yield StreamChunk(delta="partial", full_text="partial")

    worker_agent = MagicMock(generate_stream=generate_stream)
    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=MagicMock(),
        config=MagicMock(max_loop_iterations=2),
    )
    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=2)

    result = await executor.execute_frame(
        _frame(),
        max_iterations=2,
        event_sink=QueueFrameEventSink(queue),
    )

    assert result.status == FrameExecutionStatus.FAILED
    assert isinstance(result.error, RuntimeError)
