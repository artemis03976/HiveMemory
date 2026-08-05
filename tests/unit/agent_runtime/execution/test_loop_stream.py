import asyncio
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


class TrackingChunkStream:
    """记录 AgentLoop 是否显式关闭其消费的 Worker stream。"""

    def __init__(
        self,
        chunks: list[StreamChunk] | None = None,
        *,
        block_after_chunks: bool = False,
        close_error: Exception | None = None,
    ) -> None:
        self._chunks = list(chunks or [])
        self._block_after_chunks = block_after_chunks
        self._close_error = close_error
        self.pull_started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamChunk:
        if self._chunks:
            return self._chunks.pop(0)
        if self._block_after_chunks:
            self.pull_started.set()
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


@pytest.mark.asyncio
async def test_stream_sink_receives_typed_token_outputs_in_order() -> None:
    stream = TrackingChunkStream(
        [
            StreamChunk(delta="partial", full_text="partial"),
            StreamChunk(
                is_final=True,
                result=GenerationResult(text="complete", finish_reason="stop"),
            ),
        ]
    )

    worker_agent = MagicMock(generate_stream=MagicMock(return_value=stream))
    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=MagicMock(),
        config=MagicMock(max_loop_iterations=2),
    )
    sink = RecordingFrameOutputSink()

    result = await executor.execute_frame(_frame(), max_iterations=2, output_sink=sink)

    assert result.status == FrameExecutionStatus.COMPLETED
    assert sink.outputs == [TokenDelta(content="partial")]
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_is_closed_when_output_sink_fails() -> None:
    class FailingFrameOutputSink:
        streams_tokens = True

        async def send(self, output: FrameOutput) -> None:
            raise RuntimeError("sink failed")

    stream = TrackingChunkStream([StreamChunk(delta="partial", full_text="partial")])
    worker_agent = MagicMock(generate_stream=MagicMock(return_value=stream))
    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=MagicMock(),
        config=MagicMock(max_loop_iterations=2),
    )

    with pytest.raises(RuntimeError, match="sink failed"):
        await executor.execute_frame(
            _frame(),
            max_iterations=2,
            output_sink=FailingFrameOutputSink(),
        )

    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_is_closed_when_pull_task_is_cancelled() -> None:
    stream = TrackingChunkStream(block_after_chunks=True)
    worker_agent = MagicMock(generate_stream=MagicMock(return_value=stream))
    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=MagicMock(),
        config=MagicMock(max_loop_iterations=2),
    )

    task = asyncio.create_task(
        executor.execute_frame(
            _frame(),
            max_iterations=2,
            output_sink=RecordingFrameOutputSink(),
        )
    )
    await stream.pull_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_close_error_does_not_replace_task_cancellation() -> None:
    stream = TrackingChunkStream(
        block_after_chunks=True,
        close_error=RuntimeError("close failed"),
    )
    worker_agent = MagicMock(generate_stream=MagicMock(return_value=stream))
    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=MagicMock(),
        config=MagicMock(max_loop_iterations=2),
    )

    task = asyncio.create_task(
        executor.execute_frame(
            _frame(),
            max_iterations=2,
            output_sink=RecordingFrameOutputSink(),
        )
    )
    await stream.pull_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.close_calls == 1


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
