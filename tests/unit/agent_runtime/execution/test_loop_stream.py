from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.execution.loop import AgentLoopExecutor
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionStatus,
    GenerationResult,
    StreamChunk,
)
from hivememory.agent_runtime.output import FrameOutput, TokenDelta
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope


def _frame() -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "task"}],
        topic_id="topic-1",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
    )


class RecordingFrameOutputSink:
    streams_tokens = True

    def __init__(self) -> None:
        self.outputs: list[FrameOutput] = []

    async def send(self, output: FrameOutput) -> None:
        self.outputs.append(output)


@pytest.mark.asyncio
async def test_stream_sink_receives_typed_token_outputs_in_order() -> None:
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
    sink = RecordingFrameOutputSink()

    result = await executor.execute_frame(_frame(), max_iterations=2, output_sink=sink)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert sink.outputs == [TokenDelta(content="partial")]


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
    result = await executor.execute_frame(
        _frame(),
        max_iterations=2,
        output_sink=RecordingFrameOutputSink(),
    )

    assert result.status == FrameExecutionStatus.FAILED
    assert isinstance(result.error, RuntimeError)
