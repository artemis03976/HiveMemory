"""任务进程 ``chat.run.*`` 观测事件的投影契约。

驱动真实 TaskProcessService + RuntimeEventPublisher，只以 RecordingRuntimeEventSink
替换事件总线这一边界外端口；子系统路由用 GlobalSystemBus 上的替身注册，
Actor 阶段以测试 CPU 替换 Alice。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.core.errors import AssetNotReadyError
from hivememory.core.models import OMNI_DOLL_PROFILE, IdentityScope, ResolvedAgentProfile
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.process import NonStreamingAgentOutcome, TaskProcessService
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.workspace import make_identity_scope

_TOPIC_ID = "topic-events"


def _scope() -> IdentityScope:
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


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
        identity_scope=identity_scope,
        interaction_id=interaction_id,
        topic_id=_TOPIC_ID,
        is_new_topic=False,
        retrieval_result=RetrievalResponse(),
    )


async def _profile(agent_id, *, identity_scope):
    return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)


def _bus_until_actor() -> GlobalSystemBus:
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _prepare)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    return bus


def _chat_events(sink: RecordingRuntimeEventSink) -> list[RuntimeEvent]:
    return [event for event in sink.events if event.event_type.startswith("chat.run.")]


@pytest.mark.asyncio
async def test_completed_stream_events_share_process_correlation_and_bind_topic() -> None:
    """完成的流式进程：生命周期事件共享进程关联字段，prepare 之后的事件关联 Topic。"""
    bus = _bus_until_actor()
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        AsyncMock(return_value=[SimpleNamespace(task_id="memory-task-1")]),
    )
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))
    sink = RecordingRuntimeEventSink()
    service = TaskProcessService(bus, RuntimeEventPublisher(sink), cpu=cpu)

    async for _ in service.run_process(
        "问题", identity_scope=_scope(), process_id="process-events"
    ):
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
            _scope().workspace_identity.workspace_id,
            "omni_doll",
            "foreground",
            "system",
            "chat_application_service",
        )
    }
    assert events[0].trace_id.startswith("task-")
    assert events[-1].data == {"memory_task_ids": ["memory-task-1"]}


@pytest.mark.asyncio
async def test_stop_during_gateway_publishes_request_and_cancelled_phase() -> None:
    """Gateway 阶段的停止：先发布停止请求判定，进程随后以 gateway 阶段取消收口。"""
    bus = GlobalSystemBus()
    gateway_started = asyncio.Event()

    async def gateway(**_kwargs):
        gateway_started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    sink = RecordingRuntimeEventSink()
    service = TaskProcessService(bus, RuntimeEventPublisher(sink), cpu=ScriptedCPU())
    task = asyncio.create_task(
        service.run_process(
            "问题", stream=False, identity_scope=_scope(), process_id="process-stop"
        )
    )
    await gateway_started.wait()

    service.cancel_process("process-stop", identity_scope=_scope())
    await task

    events = _chat_events(sink)
    cancel_requested = next(
        event for event in events if event.event_type == RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED
    )
    assert (
        cancel_requested.process_id,
        cancel_requested.workspace_id,
        cancel_requested.status,
        cancel_requested.reason,
        cancel_requested.data,
    ) == (
        "process-stop",
        _scope().workspace_identity.workspace_id,
        "stop_requested",
        "user_requested",
        {"cancelled": True},
    )
    cancelled = events[-1]
    assert cancelled.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    assert (cancelled.status, cancelled.reason, cancelled.data) == (
        "cancelled",
        "user_requested",
        {"phase": "gateway"},
    )
    assert cancelled.trace_id == events[0].trace_id


@pytest.mark.asyncio
async def test_failed_event_carries_domain_error_code() -> None:
    """Workspace 领域错误的失败事件只携带安全错误码。"""
    bus = _bus_until_actor()
    cpu = ScriptedCPU(error=AssetNotReadyError("附件尚未就绪"))
    sink = RecordingRuntimeEventSink()
    service = TaskProcessService(bus, RuntimeEventPublisher(sink), cpu=cpu)

    with pytest.raises(AssetNotReadyError):
        await service.run_process(
            "问题", stream=False, identity_scope=_scope(), process_id="p-domain"
        )

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
    service = TaskProcessService(bus, RuntimeEventPublisher(sink), cpu=cpu)

    with pytest.raises(RuntimeError, match="internal-secret-detail"):
        await service.run_process(
            "问题", stream=False, identity_scope=_scope(), process_id="p-internal"
        )

    failed = _chat_events(sink)[-1]
    assert failed.event_type == RuntimeEventType.CHAT_RUN_FAILED
    assert failed.message == "Task process failed."
    assert "internal-secret-detail" not in failed.model_dump_json()


@pytest.mark.asyncio
async def test_stream_closed_before_terminal_publishes_cancelled_with_close_reason() -> None:
    """交付方在终态前关闭流：进程按断流取消收口并发布关闭原因。"""
    sink = RecordingRuntimeEventSink()
    service = TaskProcessService(
        GlobalSystemBus(),
        RuntimeEventPublisher(sink),
        cpu=ScriptedCPU(result=make_cpu_result()),
    )
    stream = service.run_process("问题", identity_scope=_scope(), process_id="process-closed")

    first = await stream.__anext__()
    await stream.aclose()

    assert first["event"] == "process_id"
    cancelled = _chat_events(sink)[-1]
    assert cancelled.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    assert (cancelled.status, cancelled.reason, cancelled.data) == (
        "cancelled",
        "stream_closed",
        {"close_reason": "stream_closed"},
    )
    assert service.process_status("process-closed", identity_scope=_scope()) is None


class _RaisingSink(RecordingRuntimeEventSink):
    def emit(self, event: RuntimeEvent) -> None:
        raise RuntimeError("sink unavailable")


@pytest.mark.asyncio
async def test_event_sink_failure_does_not_change_chat_result() -> None:
    """观测是 best-effort 旁路：sink 失败时进程仍按业务结果完成。"""
    bus = _bus_until_actor()
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    cpu = ScriptedCPU(result=make_cpu_result())
    service = TaskProcessService(bus, RuntimeEventPublisher(_RaisingSink()), cpu=cpu)

    result = await service.run_process(
        "问题",
        stream=False,
        identity_scope=_scope(),
        process_id="process-sink-failure",
    )

    assert isinstance(result, NonStreamingAgentOutcome)
    assert result.execution_result.status == "completed"
