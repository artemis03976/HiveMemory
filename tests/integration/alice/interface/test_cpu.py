"""真实 AliceCPU、AgentRunService 与交互输出端口的执行协作。

执行层以确定性 run_frame 替身隔离 LLM；执行骨架与取消收尾均真实。
"""

import asyncio

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.alice.interface.cpu import AliceCPU
from tests.helpers.alice import CollectedEvents, build_run_service
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.llm import ScriptedLLMService
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_access_composition, make_actor_access_record


async def _thread_access():
    return await make_access_composition([make_actor_access_record()]).authenticate_thread()


@pytest.mark.asyncio
async def test_cpu_pushes_events_then_returns_result_from_real_service():
    """真实执行服务的帧输出经 sink 推送，终态作为返回值交回。"""
    runtime, service = build_run_service(llm_service=ScriptedLLMService())

    async def complete(frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="执行完成"))
        frame.progress.text_segments.append("执行完成")
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    runtime.run_frame = complete
    cpu = AliceCPU(service, OperationsHarness().entry)
    cpu.start()
    events = CollectedEvents()
    result = await cpu.execute(
        make_input_manifest(), access=await _thread_access(), generation_options=None, events=events
    )
    assert [event["event"] for event in events.events] == ["token"]
    assert events.events[0]["data"]["content"] == "执行完成"
    assert result.status == "completed"
    assert result.final_text == "执行完成"


@pytest.mark.asyncio
async def test_cancelled_cpu_finishes_cleanup_before_returning_cancelled_result():
    """原生取消沿执行栈收尾完毕后，CPU 才以 cancelled 结果返回，而不是传播取消。"""
    runtime, service = build_run_service(llm_service=ScriptedLLMService())
    cleaned = asyncio.Event()

    async def running(frame, *, output_sink, **_kwargs):
        try:
            await output_sink.send(TokenDelta(content="开始"))
            await asyncio.Event().wait()
        finally:
            # 让清理跨越一次调度点，检验返回前确实等待了收尾。
            cleanup = asyncio.Event()
            asyncio.get_running_loop().call_soon(cleanup.set)
            await cleanup.wait()
            cleaned.set()

    runtime.run_frame = running
    cpu = AliceCPU(service, OperationsHarness().entry)
    cpu.start()
    events = CollectedEvents()
    execution = asyncio.create_task(
        cpu.execute(
            make_input_manifest(),
            access=await _thread_access(),
            generation_options=None,
            events=events,
        )
    )
    async with asyncio.timeout(5):
        await events.wait_for(lambda event: event["event"] == "token")
        execution.cancel()
        result = await execution
    assert cleaned.is_set()
    assert result.status == "cancelled"
