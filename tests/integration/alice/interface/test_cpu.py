"""真实 AliceCPU、AgentRunService 与流适配器的执行协作。

执行层以确定性 run_frame 替身隔离 LLM；执行骨架与流生命周期均真实。
"""

import asyncio

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.alice.interface.cpu import AliceCPU
from hivememory.alice.interface.stream import AgentRunStreamAdapter
from hivememory.workspace.contracts import CPUExecutionResult
from tests.helpers.alice import build_run_service
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.llm import ScriptedLLMService
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@pytest.mark.asyncio
async def test_cpu_stream_delivers_events_then_result_from_real_service():
    """真实执行服务的帧输出与类型化结果经 CPU 端口按序到达消费方。"""
    runtime, service = build_run_service(llm_service=ScriptedLLMService())

    async def complete(frame, *, output_sink, **_kwargs):
        await output_sink.send(TokenDelta(content="执行完成"))
        frame.progress.text_segments.append("执行完成")
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    runtime.run_frame = complete
    operations = OperationsHarness()
    cpu = AliceCPU(service, operations.entry, stream_adapter=AgentRunStreamAdapter())
    cpu.start()
    handle = cpu.execute(
        make_input_manifest(),
        access=await make_access_composition([make_actor_access_record()]).authenticate_thread(),
        generation_options=None,
        stream=True,
    )
    items = [item async for item in handle.output]
    assert [item["event"] for item in items[:-1]] == ["token"]
    assert items[0]["data"]["content"] == "执行完成"
    result = items[-1]
    assert isinstance(result, CPUExecutionResult)
    assert result.status == "completed"
    assert result.final_text == "执行完成"
    assert handle.result_slot.result is result


@pytest.mark.asyncio
async def test_closing_cpu_stream_cancels_and_joins_runner():
    """消费者关闭CPU 输出流时，返回前已执行 runner 的异步清理。"""
    runtime, service = build_run_service(llm_service=ScriptedLLMService())
    closed = asyncio.Event()

    async def running(frame, *, output_sink, **_kwargs):
        try:
            await output_sink.send(TokenDelta(content="开始"))
            await asyncio.Event().wait()
        finally:
            # 让清理跨越一次调度点，检验关闭者确实等待 runner 收尾。
            cleanup = asyncio.Event()
            asyncio.get_running_loop().call_soon(cleanup.set)
            await cleanup.wait()
            closed.set()

    runtime.run_frame = running
    operations = OperationsHarness()
    cpu = AliceCPU(service, operations.entry, stream_adapter=AgentRunStreamAdapter())
    cpu.start()
    async with asyncio.timeout(5):
        handle = cpu.execute(
            make_input_manifest(),
            access=await make_access_composition(
                [make_actor_access_record()]
            ).authenticate_thread(),
            generation_options=None,
            stream=True,
        )
        assert (await anext(handle.output))["data"]["content"] == "开始"
        await handle.output.aclose()
    assert closed.is_set()
    assert handle.result_slot.result.status == "cancelled"
