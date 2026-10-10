"""``AliceCPU``（``CPUPort`` 的 Alice 实现）的单元测试。

被测边界：端口实现经全局总线调用统一执行路由——流式时交互事件原样透传、
``done`` 被转换为 CPU 中立的执行结果；非流式时只产出执行结果；关闭端口
迭代器会关闭 Alice 的事件流。总线以替身 handler 注册隔离。
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from hivememory.alice.application.cpu import AliceCPU
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models.reference import ReferenceResolution
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    ExecutionCredential,
    OperationRequest,
    ResolveReferencesRequest,
)
from tests.helpers.chat_handoff import make_input_manifest, make_mtp_turn_events

# done 事件数据：结果字段 + Alice 的运行元数据（不应进入执行结果）
_DONE_DATA = {
    **CPUExecutionResult(
        status=CPUExecutionStatus.COMPLETED,
        final_text="完成",
        turn_events=make_mtp_turn_events(),
        model_used="glm-4",
    ).model_dump(),
    "agent_run_id": "agent_run_1",
    "frame_id": "frame-1",
    "scope": "main",
    "depth": 0,
    "agent_id": "omni_doll",
    "process_id": "process-1",
    "stream_sequence": 2,
}


class _ReferenceEntry:
    """凭据绑定测试的外部入口替身，按凭据返回对应的引用结果。"""

    def __init__(self) -> None:
        self.labels: dict[ExecutionCredential, str] = {}

    async def execute[R](
        self, request: OperationRequest[R], *, credential: ExecutionCredential
    ) -> R:
        if not isinstance(request, ResolveReferencesRequest):
            raise TypeError(f"Unsupported request: {type(request).__name__}")
        return cast(
            R,
            [
                ReferenceResolution(
                    kind="not_found", requested_alias=f"{self.labels[credential]}:{alias}"
                )
                for alias in request.aliases
            ],
        )


async def _stream_route(**_kwargs):
    async def _events():
        yield {"event": "token", "data": {"content": "完成"}}
        yield {"event": "mtp_start", "data": {"verb": "SEARCH"}}
        yield {"event": "done", "data": _DONE_DATA}

    return _events()


@pytest.mark.asyncio
async def test_alice_cpu_stream_relays_events_and_converts_done() -> None:
    """流式：交互事件原样透传，done 被转换为执行结果且元数据不进入结果。"""
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, _stream_route)
    cpu = AliceCPU(bus, _ReferenceEntry())
    manifest = make_input_manifest(process_id="process-1")

    items = [
        item
        async for item in cpu.execute(
            manifest,
            generation_options={"temperature": 0.5},
            stream=True,
            credential=ExecutionCredential(),
        )
    ]

    assert items[:-1] == [
        {"event": "token", "data": {"content": "完成"}},
        {"event": "mtp_start", "data": {"verb": "SEARCH"}},
    ]
    result = items[-1]
    assert isinstance(result, CPUExecutionResult)
    # done 的数据被校验还原：turn_events 回到 TurnEvent 对象，元数据被忽略。
    assert result.status == CPUExecutionStatus.COMPLETED.value
    assert result.final_text == "完成"
    assert result.model_used == "glm-4"
    assert [event.kind for event in result.turn_events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
    ]
    assert result.model_extra is None


@pytest.mark.asyncio
async def test_alice_cpu_once_yields_single_execution_result() -> None:
    """非流式：只产出路由返回的执行结果一项。"""
    expected = CPUExecutionResult(final_text="完成", model_used="glm-4")
    bus = GlobalSystemBus()

    async def route(**kwargs):
        assert kwargs["input_manifest"].process_id == "process-once"
        assert kwargs["generation_options"] is None
        assert kwargs["stream"] is False
        return expected

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, route)
    cpu = AliceCPU(bus, _ReferenceEntry())

    items = [
        item
        async for item in cpu.execute(
            make_input_manifest(process_id="process-once"),
            generation_options=None,
            stream=False,
            credential=ExecutionCredential(),
        )
    ]

    assert items == [expected]


@pytest.mark.asyncio
async def test_closing_cpu_iterator_closes_alice_event_stream() -> None:
    """关闭端口迭代器会关闭 Alice 的事件流（Alice 以取消语义收口自己的 run）。"""
    bus = GlobalSystemBus()
    stream_closed = asyncio.Event()

    async def _events():
        try:
            yield {"event": "token", "data": {"content": "一"}}
            await asyncio.Event().wait()
        finally:
            stream_closed.set()

    async def stream_route(**_kwargs):
        return _events()

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, stream_route)
    cpu = AliceCPU(bus, _ReferenceEntry())

    iterator = cpu.execute(
        make_input_manifest(),
        generation_options=None,
        stream=True,
        credential=ExecutionCredential(),
    )
    first = await iterator.__anext__()
    assert first == {"event": "token", "data": {"content": "一"}}

    await iterator.aclose()

    assert stream_closed.is_set()


@pytest.mark.asyncio
async def test_cpu_binds_each_execution_credential_to_its_operation_submitter() -> None:
    """每次执行的提交函数使用自己的凭据，不能被下一次执行的凭据覆盖。"""
    bus = GlobalSystemBus()
    entry = _ReferenceEntry()
    first_credential = ExecutionCredential()
    second_credential = ExecutionCredential()
    entry.labels[first_credential] = "first"
    entry.labels[second_credential] = "second"
    submitters = []

    async def route(*, submit_operation, **_kwargs):
        submitters.append(submit_operation)
        references = await submit_operation(ResolveReferencesRequest(("fact",)))
        return CPUExecutionResult(final_text=references[0].requested_alias)

    bus.register(GlobalRoutes.ALICE_RUN_AGENT, route)
    cpu = AliceCPU(bus, entry)
    results = []
    for credential in (first_credential, second_credential):
        results.extend(
            [
                result
                async for result in cpu.execute(
                    make_input_manifest(),
                    credential=credential,
                    generation_options=None,
                    stream=False,
                )
            ]
        )

    assert [result.final_text for result in results] == ["first:fact", "second:fact"]
    first_references = await submitters[0](ResolveReferencesRequest(("again",)))
    assert first_references[0].requested_alias == "first:again"
