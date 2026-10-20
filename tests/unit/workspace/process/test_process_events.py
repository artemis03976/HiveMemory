"""任务进程 ``chat.run.*`` 观测事件的投影契约。

驱动真实 TaskProcessService + RuntimeEventPublisher，只以 RecordingRuntimeEventSink
替换事件总线这一边界外端口；workspace_id / agent_id 观测标签由注册入口在
创建进程时用通过认证的声明绑定一次。子系统路由用 GlobalSystemBus 上的替身
注册，Actor 阶段以测试 CPU 替换 Alice。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.core.errors import AssetNotReadyError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    ResolvedAgentProfile,
    WorkspaceIdentity,
)
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUFailureReason
from hivememory.workspace.process import NonStreamingAgentOutcome, TaskProcessService
from hivememory.workspace.process.service import ProcessHandle
from hivememory.workspace.process.table import ProcessTable
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service, wait_for_process_exit
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

_USER = "u1"
_AGENT = "omni_doll"
_TOPIC_ID = "topic-events"


def _actor() -> ActorIdentity:
    """注册入口的 actor 声明（认证前不组装 IdentityScope）。"""
    return ActorIdentity(user_id=_USER, agent_id=_AGENT)


def _workspace() -> WorkspaceIdentity:
    """注册入口的请求进入 workspace 声明。"""
    return make_workspace_identity(owner_user_id=_USER)


def _composition() -> AccessTestComposition:
    """u1/omni_doll 全 operation 的访问组合：注册声明与请求方 context 的签发来源。"""
    return make_access_composition([make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT)])


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id=_TOPIC_ID,
            rewritten_query="问题",
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(),
            intent_type=IntentType.RAG,
        )
    )


async def _prepare(*, identity_scope, interaction_id, **_kwargs):
    return PreparedAgentRun(
        belong_to=(identity_scope).workspace_identity,
        interaction_id=interaction_id,
        topic_id=_TOPIC_ID,
        is_new_topic=False,
    )


async def _profile(agent_id, *, identity_scope, **_kwargs):
    return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)


def _bus_until_actor() -> GlobalSystemBus:
    bus = GlobalSystemBus()
    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _prepare)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    return bus


async def _service(
    bus: GlobalSystemBus,
    publisher=None,
    cpu=None,
    *,
    composition: AccessTestComposition | None = None,
    process_table: ProcessTable | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与阶段授权使用同一网关/授权者实例。"""
    composition = composition or _composition()
    service = make_task_process_service(
        bus,
        event_publisher=publisher,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        process_table=process_table,
    )
    return service, composition


async def _register(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
) -> ProcessHandle:
    """按组合的默认声明注册进程：观测标签在创建时用通过认证的声明绑定一次。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=_actor(),
        workspace=_workspace(),
        process_id=process_id,
        message=message,
    )


def _chat_events(sink: RecordingRuntimeEventSink) -> list[RuntimeEvent]:
    return [event for event in sink.events if event.event_type.startswith("chat.run.")]


@pytest.mark.asyncio
async def test_completed_stream_events_share_process_correlation_and_bind_topic() -> None:
    """完成的流式进程：生命周期事件共享进程关联字段，prepare 之后的事件关联 Topic。"""
    bus = _bus_until_actor()
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(
        PatchouliRoutes.FINALIZE_AGENT_RUN,
        AsyncMock(return_value=[SimpleNamespace(task_id="memory-task-1")]),
    )
    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(bus, RuntimeEventPublisher(sink), cpu=cpu)
    handle = await _register(composition, service, "问题", process_id="process-events")

    async for _ in service.run_process(handle, stream=True):
        pass

    events = _chat_events(sink)
    assert [(event.event_type, event.status) for event in events] == [
        (RuntimeEventType.CHAT_RUN_CREATED, "created"),
        (RuntimeEventType.CHAT_RUN_STATUS, "preparing"),
        (RuntimeEventType.CHAT_RUN_STATUS, "streaming"),
        (RuntimeEventType.CHAT_RUN_STATUS, "finalizing"),
        (RuntimeEventType.CHAT_RUN_COMPLETED, "completed"),
    ]
    assert [event.topic_id for event in events] == [None, None, _TOPIC_ID, _TOPIC_ID, _TOPIC_ID]
    assert {
        (
            event.process_id,
            event.trace_id,
            event.workspace_id,
            event.agent_id,
            event.task_type,
            event.subsystem,
            event.component,
        )
        for event in events
    } == {
        (
            "process-events",
            events[0].trace_id,
            _workspace().workspace_id,
            _AGENT,
            "foreground",
            "workspace",
            "task_process",
        )
    }
    assert events[0].trace_id.startswith("task-")
    assert events[-1].data == {}


@pytest.mark.asyncio
async def test_stop_during_gateway_cancels_process_in_gateway_phase() -> None:
    """Gateway 阶段的停止：进程以 gateway 阶段取消收口，事件沿用创建时绑定的关联字段。"""
    bus = GlobalSystemBus()
    gateway_started = asyncio.Event()

    async def gateway(**_kwargs):
        gateway_started.set()
        await asyncio.Event().wait()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(bus, RuntimeEventPublisher(sink), cpu=ScriptedCPU())
    handle = await _register(composition, service, "问题", process_id="process-stop")
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await gateway_started.wait()

    stop_result = service.cancel_process(handle, reason="user_requested")
    await task

    assert stop_result.cancelled is True
    events = _chat_events(sink)
    cancelled = events[-1]
    assert cancelled.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    assert (cancelled.status, cancelled.reason, cancelled.data) == (
        "cancelled",
        "user_requested",
        {"phase": "gateway"},
    )
    assert (cancelled.process_id, cancelled.workspace_id, cancelled.agent_id) == (
        "process-stop",
        _workspace().workspace_id,
        _AGENT,
    )
    assert cancelled.trace_id == events[0].trace_id


@pytest.mark.asyncio
async def test_cancel_of_uncontrollable_process_publishes_request_event_with_requestor_label() -> (
    None
):
    """取消不可控进程：发布 not_found 的停止请求判定，workspace 标签取请求方驻留摘要。"""
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(
        GlobalSystemBus(), RuntimeEventPublisher(sink), cpu=ScriptedCPU()
    )
    requestor = await composition.authenticate(agent_id=_AGENT)

    result = service.cancel_process("missing-process", access=requestor)

    assert result.cancelled is False
    assert result.status == "not_found"
    cancel_requested = [
        event
        for event in _chat_events(sink)
        if event.event_type == RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED
    ]
    assert len(cancel_requested) == 1
    event = cancel_requested[0]
    assert (
        event.process_id,
        event.workspace_id,
        event.status,
        event.reason,
        event.data,
    ) == (
        "missing-process",
        _workspace().workspace_id,
        "not_found",
        "user_requested",
        {"cancelled": False},
    )


@pytest.mark.asyncio
async def test_failed_event_carries_domain_error_code() -> None:
    """Workspace 领域错误的失败事件只携带安全错误码。"""
    bus = _bus_until_actor()
    cpu = ScriptedCPU(error=AssetNotReadyError("附件尚未就绪"))
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(bus, RuntimeEventPublisher(sink), cpu=cpu)
    handle = await _register(composition, service, "问题", process_id="p-domain")

    with pytest.raises(AssetNotReadyError):
        await service.run_process(handle, stream=False)

    failed = _chat_events(sink)[-1]
    assert failed.event_type == RuntimeEventType.CHAT_RUN_FAILED
    assert (failed.severity, failed.status, failed.message, failed.topic_id) == (
        "error",
        "failed",
        "workspace.asset.not_ready",
        _TOPIC_ID,
    )


@pytest.mark.asyncio
async def test_failed_event_does_not_expose_exception_text() -> None:
    """非领域异常的失败事件使用固定摘要，异常正文不进入公共观测信封。"""
    bus = _bus_until_actor()
    cpu = ScriptedCPU(error=RuntimeError("internal-secret-detail"))
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(bus, RuntimeEventPublisher(sink), cpu=cpu)
    handle = await _register(composition, service, "问题", process_id="p-internal")

    with pytest.raises(RuntimeError, match="internal-secret-detail"):
        await service.run_process(handle, stream=False)

    failed = _chat_events(sink)[-1]
    assert failed.event_type == RuntimeEventType.CHAT_RUN_FAILED
    assert failed.message == "Task process failed."
    assert "internal-secret-detail" not in failed.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, CPUFailureReason.TURN_CONTEXT_BUDGET_EXCEEDED])
async def test_cpu_failure_reason_reaches_done_and_runtime_event(reason):
    """CPU 自报失败透传稳定原因码，未填写时保留既有 done 与观测语义。"""
    bus = _bus_until_actor()
    sink = RecordingRuntimeEventSink()
    cpu = ScriptedCPU(result=make_cpu_result(status="failed", final_text="", failure_reason=reason))
    service, composition = await _service(bus, RuntimeEventPublisher(sink), cpu=cpu)
    handle = await _register(composition, service, "问题", process_id="cpu-failure")

    events = [event async for event in service.run_process(handle, stream=True)]

    done = events[-1]["data"]
    assert done["status"] == "failed"
    assert done["reason"] == "agent_run_failed"
    assert done.get("failure_reason") == reason
    failed = _chat_events(sink)[-1]
    assert failed.event_type == RuntimeEventType.CHAT_RUN_FAILED
    assert failed.message == reason


def _hanging_gateway_bus() -> tuple[GlobalSystemBus, asyncio.Event]:
    """Gateway 调用挂起的总线，返回 Gateway 已进入的事件。"""
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    return bus, started


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["stream", "once"])
async def test_delivery_leaving_before_terminal_stops_process_with_its_reason(
    delivery: str,
) -> None:
    """交付方在终态前离开（流被关闭或等待被取消）：以交付方给出的原因走普通停止路径，
    发布停止请求与带生效阶段的 cancelled 事件。"""
    bus, gateway_started = _hanging_gateway_bus()
    sink = RecordingRuntimeEventSink()
    table = ProcessTable()
    service, composition = await _service(bus, RuntimeEventPublisher(sink), process_table=table)
    handle = await _register(composition, service, "问题", process_id="process-left")

    if delivery == "stream":
        stream = service.run_process(handle, stream=True, detach_reason="client_disconnected")
        assert (await anext(stream))["event"] == "process_id"
        await gateway_started.wait()
        await stream.aclose()
    else:
        waiter = asyncio.create_task(
            service.run_process(handle, stream=False, detach_reason="client_disconnected")
        )
        await gateway_started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    await wait_for_process_exit(table, "process-left")

    events = _chat_events(sink)
    assert [(event.event_type, event.status, event.reason, event.data) for event in events[2:]] == [
        (
            RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
            "stop_requested",
            "client_disconnected",
            {"cancelled": True},
        ),
        (RuntimeEventType.CHAT_RUN_STATUS, "stop_requested", "client_disconnected", {}),
        (
            RuntimeEventType.CHAT_RUN_CANCELLED,
            "cancelled",
            "client_disconnected",
            {"phase": "gateway"},
        ),
    ]
    assert table.get("process-left") is None


@pytest.mark.asyncio
async def test_hard_cancelled_process_publishes_cancelled_with_close_reason() -> None:
    """停止后未能在关闭等待内结束、被系统关闭硬取消的进程：按“终态前关闭”形态收口。"""
    sink = RecordingRuntimeEventSink()
    cpu = ScriptedCPU(hang_before_result=True, linger_on_cancel=True)
    composition = _composition()
    service = make_task_process_service(
        _bus_until_actor(),
        event_publisher=RuntimeEventPublisher(sink),
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        shutdown_wait_seconds=0,
    )
    handle = await _register(composition, service, "问题", process_id="process-hard-cancel")
    stream = service.run_process(handle, stream=True)
    await anext(stream)
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)

    await service.shutdown()

    cancelled = _chat_events(sink)[-1]
    assert (cancelled.event_type, cancelled.status, cancelled.reason) == (
        RuntimeEventType.CHAT_RUN_CANCELLED,
        "cancelled",
        "system_shutdown",
    )
    assert cancelled.message == "Task process closed before terminal event."
    assert cancelled.data == {"close_reason": "system_shutdown"}
    assert [event["event"] async for event in stream] == ["memory_refs"]


class _RaisingSink(RecordingRuntimeEventSink):
    def emit(self, event: RuntimeEvent) -> None:
        raise RuntimeError("sink unavailable")


@pytest.mark.asyncio
async def test_event_sink_failure_does_not_change_chat_result() -> None:
    """观测是 best-effort 旁路：sink 失败时进程仍按业务结果完成。"""
    bus = _bus_until_actor()
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    cpu = ScriptedCPU(result=make_cpu_result())
    service, composition = await _service(bus, RuntimeEventPublisher(_RaisingSink()), cpu=cpu)
    handle = await _register(composition, service, "问题", process_id="process-sink-failure")

    result = await service.run_process(handle, stream=False)

    assert isinstance(result, NonStreamingAgentOutcome)
    assert result.execution_result.status == "completed"
