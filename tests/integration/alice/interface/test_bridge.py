"""真实全局总线、AliceBridge 与 AgentRunService 的两种执行路由协作。

执行层以确定性 run_frame 替身隔离 LLM；总线、执行骨架与流生命周期均真实。
"""

import asyncio

import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.agent_runtime.output import TokenDelta
from hivememory.alice.contracts import AliceRoutes
from hivememory.alice.interface.bridge import AliceBridge, AlicePublicApi
from hivememory.components.bus import GlobalSystemBus
from hivememory.workspace.contracts import CPUExecutionResult
from tests.helpers.chat_handoff import make_input_manifest
from tests.integration.alice.run.test_service import _build_service


async def _unused_operation(request):
    raise AssertionError("纯文本执行不应提交资源操作")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_execution_route_returns_completed_result_or_done_stream(stream):
    """同一真实执行骨架按所选路由交付结果或以 done 封口的事件流。"""
    runtime, service = _build_service()

    async def complete(frame, **_kwargs):
        frame.progress.text_segments.append("执行完成")
        return FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)

    runtime.run_frame = complete
    bus = GlobalSystemBus()
    bridge = AliceBridge(global_bus=bus, public_api=AlicePublicApi(agent=service))
    bridge.mount()
    selected = AliceRoutes.RUN_AGENT_STREAM if stream else AliceRoutes.RUN_AGENT
    try:
        result = await bus.request(
            selected, make_input_manifest(), submit_operation=_unused_operation
        )
        if stream:
            events = [event async for event in result]
            assert events[-1]["event"] == "done"
            assert events[-1]["data"]["status"] == "completed"
            assert events[-1]["data"]["final_text"] == "执行完成"
        else:
            assert isinstance(result, CPUExecutionResult)
            assert result.status == "completed"
            assert result.final_text == "执行完成"
        assert bus.list_unresolved_routes() == []
    finally:
        bridge.unmount()
    with pytest.raises(KeyError, match="not registered"):
        await bus.request(selected, make_input_manifest(), submit_operation=_unused_operation)


@pytest.mark.asyncio
async def test_closing_route_stream_cancels_and_joins_runner():
    """消费者关闭路由返回的流时，返回前已执行 runner 的异步清理。"""
    runtime, service = _build_service()
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
    bus = GlobalSystemBus()
    bridge = AliceBridge(global_bus=bus, public_api=AlicePublicApi(agent=service))
    bridge.mount()
    try:
        async with asyncio.timeout(5):
            events = await bus.request(
                AliceRoutes.RUN_AGENT_STREAM,
                make_input_manifest(),
                submit_operation=_unused_operation,
            )
            assert (await anext(events))["data"]["content"] == "开始"
            await events.aclose()
        assert closed.is_set()
    finally:
        bridge.unmount()
