"""任务进程经 CPU 端口执行 Actor 阶段的行为测试。

被测边界：进程只经组合根注入的 ``CPUPort`` 调用 CPU，总线上不注册任何
Alice 路由；测试 CPU（``tests.helpers.cpu``）覆盖完成、自报取消/失败、
停止请求、协议错误与断流关闭五种场景。子系统路由用 GlobalSystemBus 上的
替身隔离。
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.contracts import CPUExecutionStatus
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_gateway_decision,
    make_mtp_turn_events,
    make_write_materialize_task,
)
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope
from tests.helpers.workspace_assets import make_ready_text_asset


def _u1_scope() -> IdentityScope:
    """Chat 是 Agent action：构造携带具体 Agent 的显式 scope。"""
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


def _memory_atom(title: str) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title=title,
            summary=f"{title} 摘要",
            tags=[],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=f"{title} 正文"),
    )


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(decision=make_gateway_decision())


async def _gateway_route(**_kwargs):
    return _decision_outcome()


async def _prepare_route(
    *,
    identity_scope,
    interaction_id,
    **_kwargs,
):
    return PreparedAgentRun(
        identity_scope=identity_scope,
        interaction_id=interaction_id,
        topic_id="topic-1",
        is_new_topic=False,
        retrieval_result=RetrievalResponse(),
        storage_available=True,
    )


async def _profile_route(agent_id, *, identity_scope):
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)


def _service(
    bus: GlobalSystemBus,
    cpu: ScriptedCPU,
    *,
    event_publisher: RuntimeEventPublisher | None = None,
    store: InMemoryWorkspaceAssetStore | None = None,
) -> TaskProcessService:
    return TaskProcessService(
        bus,
        event_publisher,
        cpu=cpu,
        asset_reader=store,
    )


async def _run_once(service: TaskProcessService, message: str, *, process_id: str, **kwargs):
    return await service.run_process(
        stream=False,
        message=message,
        identity_scope=_u1_scope(),
        process_id=process_id,
        **kwargs,
    )


async def _stream_events(
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
) -> list[dict]:
    return [
        event
        async for event in service.run_process(
            message=message,
            identity_scope=_u1_scope(),
            process_id=process_id,
            **kwargs,
        )
    ]


# ========== 测试 CPU 跑通完整进程（总线上无 Alice 路由） ==========


@pytest.mark.asyncio
async def test_test_cpu_completes_non_streaming_process_without_alice_routes() -> None:
    """测试 CPU 跑完非流式进程：finalize 收到的交互记录来自 CPU 执行结果。"""
    bus = GlobalSystemBus()
    turn_events = make_mtp_turn_events()
    write_task = make_write_materialize_task()
    finalize_kwargs: dict = {}
    cpu = ScriptedCPU(
        result=make_cpu_result(
            turn_events=turn_events,
            materialize_tasks=[write_task],
        )
    )

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await _run_once(_service(bus, cpu), "问题", process_id="process-cpu-once")

    assert result.kind == "agent"
    assert result.execution_result.status == CPUExecutionStatus.COMPLETED.value
    assert result.execution_result.final_text == "完成"
    assert result.execution_result.model_used == "glm-4"
    assert cpu.calls[0].stream is False
    # 总线上只有 Gateway 与 Patchouli 的公开路由，没有任何 Alice 路由。
    assert GlobalRoutes.ALICE_RUN_AGENT not in bus.list_routes()

    payload = finalize_kwargs["payload"]
    assert payload.assistant_final_text == "完成"
    assert payload.turn_events == turn_events
    assert payload.model_used == "glm-4"
    assert payload.materialize_tasks == [write_task]
    assert payload.mtp_traces == expected_mtp_traces()


@pytest.mark.asyncio
async def test_test_cpu_completes_streaming_process_and_relays_events() -> None:
    """测试 CPU 跑完流式进程：交互事件原样转交，done 不含 Alice 的统计字段。"""
    bus = GlobalSystemBus()
    interaction_event = {"event": "token", "data": {"content": "完成"}}
    cpu = ScriptedCPU(events=[interaction_event], result=make_cpu_result())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    events = await _stream_events(_service(bus, cpu), "问题", process_id="process-cpu-stream")

    assert cpu.calls[0].stream is True
    assert [event["event"] for event in events] == [
        "process_id",
        "topic_info",
        "memory_refs",
        "token",
        "run_status",
        "done",
    ]
    # 交互事件原样转交：内容与顺序不经过进程解释。
    assert events[3] == interaction_event
    done = events[-1]
    assert done["data"]["status"] == "completed"
    assert done["data"]["final_text"] == "完成"
    assert done["data"]["memory_task_ids"] == []
    assert "mtp_iterations" not in done["data"]
    assert "total_iterations" not in done["data"]
    assert cpu.closed is True


@pytest.mark.asyncio
async def test_test_cpu_receives_generation_options_and_manifest() -> None:
    """进程把输入清单与生成参数原样传给 CPU 端口。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))

    await _run_once(
        _service(bus, cpu),
        "问题",
        process_id="process-cpu-options",
        generation_options={"temperature": 0.3},
    )

    call = cpu.calls[0]
    assert call.manifest.process_id == "process-cpu-options"
    assert call.manifest.user_message == "问题"
    assert call.generation_options == {"temperature": 0.3}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_cpu_iterator_is_closed_before_finalize(stream: bool) -> None:
    """拿到终态结果后立即关闭 CPU 输出流：finalize 执行时 CPU 已释放自己的资源。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result())
    closed_at_finalize: list[bool] = []

    async def finalize(**_kwargs):
        closed_at_finalize.append(cpu.closed)
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))
    service = _service(bus, cpu)

    if stream:
        await _stream_events(service, "问题", process_id="process-cpu-close-early-stream")
    else:
        await _run_once(service, "问题", process_id="process-cpu-close-early-once")

    assert closed_at_finalize == [True]


# ========== CPU 自报取消与失败 ==========


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CPUExecutionStatus.CANCELLED, CPUExecutionStatus.FAILED])
async def test_cpu_self_reported_terminal_ends_process_without_finalize(
    status: CPUExecutionStatus,
) -> None:
    """CPU 自报取消/失败：进程以相应结局结束，不调用 finalize，并请求 cleanup。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=status))
    cleanup_calls: list = []

    async def cleanup(*, prepared_run):
        cleanup_calls.append(prepared_run)

    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    bus.register(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        AsyncMock(side_effect=AssertionError("finalize 不应被调用")),
    )

    result = await _run_once(_service(bus, cpu), "问题", process_id=f"process-cpu-{status.value}")

    assert result.execution_result.status == status.value
    assert len(cleanup_calls) == 1


# ========== 停止请求与协议错误 ==========


@pytest.mark.asyncio
async def test_stop_during_cpu_pull_cancels_process_and_closes_cpu_iterator() -> None:
    """CPU 在拉取时挂起：停止请求使进程以取消结束，阶段为 Actor 执行阶段。"""
    bus = GlobalSystemBus()
    sink = RecordingRuntimeEventSink()
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "一"}}],
        result=make_cpu_result(),
        hang_before_result=True,
    )
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))

    service = _service(bus, cpu, event_publisher=RuntimeEventPublisher(sink))
    process_id = "process-cpu-stop"
    task = asyncio.create_task(_stream_events(service, "问题", process_id=process_id))

    # 等 CPU 进入挂起点，再注入停止请求：stop 必须在 Actor 拉取期间生效。
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)
    stop_result = service.cancel_process(process_id, identity_scope=_u1_scope())
    events = await task

    assert stop_result.cancelled is True
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    assert cpu.closed is True
    cancelled = [
        event for event in sink.events if event.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    ]
    assert cancelled[-1].data == {"phase": "actor"}


@pytest.mark.asyncio
async def test_cpu_stream_without_terminal_result_fails_process_with_error_event() -> None:
    """测试 CPU 没有产出终态结果即结束（流式）：进程以失败结束，交付发出 error。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(events=[{"event": "token", "data": {"content": "半"}}])
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))

    events = await _stream_events(_service(bus, cpu), "问题", process_id="process-cpu-no-done")

    assert [event for event in events if event["event"] == "error"] == [
        {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}
    ]


@pytest.mark.asyncio
async def test_cpu_once_without_terminal_result_raises_protocol_error() -> None:
    """测试 CPU 没有产出终态结果即结束（非流式）：协议错误沿异常上抛。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))

    with pytest.raises(RuntimeError, match="终态执行结果"):
        await _run_once(_service(bus, cpu), "问题", process_id="process-cpu-no-done-ns")


@pytest.mark.asyncio
async def test_cpu_exception_fails_process() -> None:
    """CPU 抛出异常：进程按失败处理，异常沿非流式路径原样上抛。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(error=RuntimeError("cpu exploded"))
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))

    with pytest.raises(RuntimeError, match="cpu exploded"):
        await _run_once(_service(bus, cpu), "问题", process_id="process-cpu-error")


# ========== 断流关闭与租借释放 ==========


@pytest.mark.asyncio
async def test_delivery_closed_early_closes_cpu_iterator_and_releases_leases() -> None:
    """交付方提前关闭流式交付：测试 CPU 的迭代器被关闭，租借已释放。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _u1_scope()
    ref_a = make_ready_text_asset(store, scope, operation_id="op-a", content="正文甲")

    bus = GlobalSystemBus()
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "一"}}],
        result=make_cpu_result(),
        hang_before_result=True,
    )
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))

    service = _service(bus, cpu, store=store)
    stream = service.run_process(
        "带附件的消息",
        identity_scope=scope,
        process_id="process-cpu-closed",
        attachments=[AttachmentSelectionRequest(asset_ref=ref_a)],
    )
    seen: list[dict[str, Any]] = [await stream.__anext__()]
    # 消费到 CPU 的交互事件为止：此时 CPU 已挂起在终态之前，进程尚未发布终态。
    while seen[-1]["event"] != "token":
        seen.append(await stream.__anext__())

    await stream.aclose()

    assert cpu.closed is True
    assert store.close_and_clear().leases_cleared == 0
