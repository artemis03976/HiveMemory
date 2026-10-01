"""Phase 3F 主动聊天 Gateway 编排测试。

被测边界：``TaskProcessService`` 的四阶段编排在命令短路、完成、取消、失败
与断流路径上的行为；Actor 阶段以测试 CPU 替换 Alice（总线上不注册 Alice
路由），子系统路由用 GlobalSystemBus 上的替身隔离。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import AssetNotReadyError, WorkspaceDomainError
from hivememory.core.models import (
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnEvent,
    WorkspaceAssetRef,
)
from hivememory.core.protocol.gateway import (
    CommandExecutionStatus,
    CommandParseResult,
    CommandParseStatus,
    GatewayCommandOutcome,
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUExecutionStatus
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_mtp_turn_events,
    make_write_materialize_task,
)
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _u1_scope() -> IdentityScope:
    """Chat 是 Agent action：构造携带具体 Agent 的显式 scope。"""
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


async def _run_once(
    service: TaskProcessService,
    message: str,
    *,
    process_id: str | None = None,
    **kwargs,
):
    return await service.run_process(
        stream=False,
        message=message,
        identity_scope=_u1_scope(),
        process_id=process_id or f"process_{uuid4().hex}",
        **kwargs,
    )


async def _stream_events(
    service: TaskProcessService,
    message: str,
    *,
    process_id: str | None = None,
) -> list[dict]:
    return [
        event
        async for event in service.run_process(
            message=message,
            identity_scope=_u1_scope(),
            process_id=process_id or f"process_{uuid4().hex}",
        )
    ]


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id="topic-1",
            rewritten_query="原问题",
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(),
            intent_type=IntentType.RAG,
        )
    )


def _command_outcome() -> GatewayCommandOutcome:
    """Gateway 命令只解析：/clear 的解析成功结果（不含任何执行产物）。"""
    return GatewayCommandOutcome(
        command_parse_result=CommandParseResult(
            command_id="system.clear",
            raw_input="/clear",
            name="/clear",
            tokens=["/clear"],
            matched_alias="/clear",
            parse_status=CommandParseStatus.MATCHED,
        )
    )


def _memory_ref_atom() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="引用记忆",
            summary="引用摘要",
            tags=["t"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="引用正文"),
    )


def _assert_sealed_payload(payload, *, turn_events: list[TurnEvent], write_task) -> None:
    """完成的进程交给 finalize 的交互记录逐字段等于各阶段产出。"""
    decision = _decision_outcome().decision
    assert payload.user_message == "问题"
    assert payload.rewritten_query == decision.rewritten_query
    assert payload.worth_saving is True
    assert payload.assistant_final_text == "完成"
    assert payload.turn_events == turn_events
    assert payload.model_used == "glm-4"
    assert payload.materialize_tasks == [write_task]
    # 本测试未选择附件：附件编译的实际使用集合为空
    assert payload.used_attachments == []
    assert payload.mtp_traces == expected_mtp_traces()


def _scoped_prepared_route(
    *,
    memories: list[MemoryAtom] | None = None,
    topic_id: str = "topic-1",
    is_new_topic: bool = False,
):
    """prepare 替身：按请求 scope 构造真实 PreparedAgentRun（进程 CPU 分配的输入）。"""

    async def route(
        *,
        identity_scope,
        user_message,
        interaction_id,
        gateway_decision,
        **_kwargs,
    ):
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            user_message=user_message,
            gateway_decision=gateway_decision,
            topic_id=topic_id,
            is_new_topic=is_new_topic,
            topic_context=None,
            pool_topics=[],
            retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
            storage_available=True,
        )

    return route


def _profile_route():
    """PATCHOULI_GET_AGENT_PROFILE 替身：返回 builtin omni_doll Profile。"""
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    async def route(agent_id, *, identity_scope):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    return route


def _register_profile(bus: GlobalSystemBus) -> None:
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())


def _service(
    bus: GlobalSystemBus,
    cpu: ScriptedCPU | None = None,
) -> TaskProcessService:
    """构造被测服务：默认注入恒完成的测试 CPU。"""
    return TaskProcessService(bus, cpu=cpu or ScriptedCPU(result=make_cpu_result()))


@pytest.mark.asyncio
async def test_non_streaming_command_short_circuits_patchouli_and_cpu() -> None:
    """命令只解析不执行：进程把解析结果转为"暂不可用"终态，不调用 prepare/Profile/CPU。"""
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    cpu = ScriptedCPU(result=make_cpu_result())

    result = await _run_once(_service(bus, cpu), "/clear")

    assert result.kind == "command"
    assert result.command_execution_result.command_id == "system.clear"
    assert result.command_execution_result.status == CommandExecutionStatus.NOT_IMPLEMENTED
    assert result.command_execution_result.error_code == "command.unavailable"
    assert result.command_execution_result.client_action is None
    assert cpu.calls == []
    assert bus.list_routes() == [GlobalRoutes.GATEWAY_PROCESS]


@pytest.mark.asyncio
async def test_non_streaming_decision_uses_one_prepare_cpu_finalize_sequence() -> None:
    bus = GlobalSystemBus()
    calls: list[str] = []

    class _RecordingCPU(ScriptedCPU):
        def execute(self, manifest, *, generation_options=None, stream=False):
            calls.append("cpu")
            return super().execute(manifest, generation_options=generation_options, stream=stream)

    async def gateway(**_kwargs):
        calls.append("gateway")
        return _decision_outcome()

    async def prepare(**kwargs):
        calls.append("prepare")
        assert kwargs["gateway_decision"] == _decision_outcome().decision
        return await _scoped_prepared_route()(**kwargs)

    async def finalize(**_kwargs):
        calls.append("finalize")
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await _run_once(_service(bus, _RecordingCPU(result=make_cpu_result())), "问题")

    assert result.kind == "agent"
    assert result.execution_result.final_text == "完成"
    assert calls == ["gateway", "prepare", "cpu", "finalize"]


@pytest.mark.asyncio
async def test_completed_non_streaming_process_seals_interaction_payload() -> None:
    """完成的非流式进程：交给 finalize 的交互记录由进程封口，逐字段等于各阶段产出。"""
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

    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await _run_once(_service(bus, cpu), "问题")

    assert result.kind == "agent"
    _assert_sealed_payload(
        finalize_kwargs["payload"],
        turn_events=turn_events,
        write_task=write_task,
    )


@pytest.mark.asyncio
async def test_completed_streaming_process_seals_interaction_payload() -> None:
    """完成的流式进程：与非流式共用同一封口，CPU 终态结果逐字段进入交互记录。"""
    bus = GlobalSystemBus()
    turn_events = make_mtp_turn_events()
    write_task = make_write_materialize_task()
    finalize_kwargs: dict = {}
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完成"}}],
        result=make_cpu_result(
            turn_events=turn_events,
            materialize_tasks=[write_task],
        ),
    )

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    events = await _stream_events(_service(bus, cpu), "问题")

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    _assert_sealed_payload(
        finalize_kwargs["payload"],
        turn_events=turn_events,
        write_task=write_task,
    )


@pytest.mark.asyncio
async def test_streaming_command_emits_result_and_done_only() -> None:
    """流式命令请求：command_result 为"暂不可用"终态且不带客户端动作，随后 done 收口。"""
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_command_outcome()),
    )

    events = await _stream_events(_service(bus), "/clear")

    assert [event["event"] for event in events] == [
        "process_id",
        "command_result",
        "done",
    ]
    assert events[1]["data"]["status"] == "not_implemented"
    assert events[1]["data"]["error_code"] == "command.unavailable"
    assert events[1]["data"]["client_action"] is None
    assert events[2]["data"]["final_text"] == "系统指令 /clear 暂不可用。"


@pytest.mark.asyncio
async def test_completed_stream_uses_one_process_id_for_events_and_downstream_routes() -> None:
    """完成的流式进程：SSE 首个事件、finalizing 状态与 done 携带同一 process_id，
    并以它调用 Patchouli prepare（interaction_id）与 CPU（清单 process_id）；
    topic_info 与 memory_refs 由进程从 prepare 结果推导。"""
    bus = GlobalSystemBus()
    prepare_calls: list[dict] = []
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完成"}}],
        result=make_cpu_result(),
    )

    async def prepare(*, identity_scope, **kwargs):
        prepare_calls.append(kwargs)
        route = _scoped_prepared_route(memories=[_memory_ref_atom()])
        return await route(identity_scope=identity_scope, **kwargs)

    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    events = await _stream_events(
        _service(bus, cpu),
        "问题",
        process_id="process-complete",
    )

    assert events[0] == {"event": "process_id", "data": {"process_id": "process-complete"}}
    topic_info = next(event for event in events if event["event"] == "topic_info")
    assert topic_info["data"] == {
        "topic_id": "topic-1",
        "is_new": False,
        "pool_topics": [],
    }
    memory_refs = next(event for event in events if event["event"] == "memory_refs")
    assert memory_refs["data"]["memories"][0]["title"] == "引用记忆"
    assert [event["data"] for event in events if event["event"] == "run_status"] == [
        {"process_id": "process-complete", "status": "finalizing"}
    ]
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    assert events[-1]["data"]["process_id"] == "process-complete"
    assert [call["interaction_id"] for call in prepare_calls] == ["process-complete"]
    assert cpu.calls[0].manifest.process_id == "process-complete"


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    service = _service(bus)

    task = asyncio.create_task(_run_once(service, "问题", process_id="process-gateway"))
    await started.wait()
    stop_result = service.cancel_process("process-gateway", identity_scope=_u1_scope())
    result = await task

    assert stop_result.cancelled is True

    assert result.kind == "agent"
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_non_streaming_cancel_after_prepare_cleans_prepared_run() -> None:
    bus = GlobalSystemBus()
    prepared_holder: dict = {}
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.CANCELLED))

    async def prepare(**kwargs):
        prepared = await _scoped_prepared_route()(**kwargs)
        prepared_holder["prepared"] = prepared
        return prepared

    cleanup = AsyncMock(return_value=True)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        prepare,
    )
    _register_profile(bus)
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        cleanup,
    )

    result = await _run_once(_service(bus, cpu), "问题")

    assert result.kind == "agent"
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    cleanup.assert_awaited_once_with(prepared_run=prepared_holder["prepared"])


@pytest.mark.asyncio
async def test_non_streaming_failed_agent_run_is_not_rewritten_as_cancelled() -> None:
    bus = GlobalSystemBus()
    finalize = AsyncMock(return_value=[])
    cleanup = AsyncMock(return_value=True)
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    result = await _run_once(_service(bus, cpu), "问题")

    assert result.execution_result.status == CPUExecutionStatus.FAILED.value
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_failed_agent_run_preserves_failed_done_status() -> None:
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    events = await _stream_events(_service(bus, cpu), "问题")

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == CPUExecutionStatus.FAILED.value
    assert events[-1]["data"]["stopped"] is True
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_during_prepare_waits_for_prepare_then_skips_cpu_and_finalize() -> None:
    bus = GlobalSystemBus()
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    prepare_cancelled = False
    cpu = ScriptedCPU(result=make_cpu_result())

    async def prepare(*, identity_scope, **kwargs):
        nonlocal prepare_cancelled
        prepare_started.set()
        try:
            await release_prepare.wait()
        except asyncio.CancelledError:
            prepare_cancelled = True
            raise
        route = _scoped_prepared_route()
        return await route(identity_scope=identity_scope, **kwargs)

    finalize = AsyncMock()
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = _service(bus, cpu)

    task = asyncio.create_task(_run_once(service, "问题", process_id="process-prepare"))
    await prepare_started.wait()
    stop_result = service.cancel_process("process-prepare", identity_scope=_u1_scope())
    release_prepare.set()
    result = await task

    assert stop_result.cancelled is True
    assert prepare_cancelled is False
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    assert cpu.calls == []
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_stop_cancels_current_cpu_pull_and_closes_cpu_iterator() -> None:
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "late"}}],
        result=make_cpu_result(),
        hang_before_result=True,
    )

    finalize = AsyncMock()
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = _service(bus, cpu)

    task = asyncio.create_task(_stream_events(service, "问题", process_id="process-stream-cancel"))
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)
    stop_result = service.cancel_process("process-stream-cancel", identity_scope=_u1_scope())
    events = await task

    assert stop_result.cancelled is True
    assert cpu.closed is True
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_during_finalize_is_rejected_and_finalize_completes() -> None:
    bus = GlobalSystemBus()
    finalize_started = asyncio.Event()
    release_finalize = asyncio.Event()

    async def finalize(**_kwargs):
        finalize_started.set()
        await release_finalize.wait()
        return []

    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = _service(bus)

    task = asyncio.create_task(_run_once(service, "问题", process_id="process-finalize"))
    await finalize_started.wait()
    stop_result = service.cancel_process("process-finalize", identity_scope=_u1_scope())
    release_finalize.set()
    result = await task

    assert stop_result.cancelled is False
    assert stop_result.reason == "already_finalizing"
    assert result.execution_result.status == CPUExecutionStatus.COMPLETED.value
    cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_attachment_selection_without_reader_fails_allocation() -> None:
    """捕获装配遗漏：有附件选择但 Store 未注入时，分配显式失败且不进入 CPU。"""
    bus = GlobalSystemBus()
    cleanup = AsyncMock(return_value=True)
    cpu = ScriptedCPU(result=make_cpu_result())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    with pytest.raises(WorkspaceDomainError, match="附件读取能力"):
        await _run_once(
            _service(bus, cpu),
            "问题",
            process_id="process-no-reader",
            attachments=[
                AttachmentSelectionRequest(
                    asset_ref=WorkspaceAssetRef(token="ref-a", asset_id="asset-a"),
                ),
            ],
        )

    assert cpu.calls == []
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_workspace_domain_error_yields_safe_code() -> None:
    """捕获附件拒绝等 Workspace 领域错误丢失安全 code 或被包装成系统错误。"""
    bus = GlobalSystemBus()

    async def prepare(**_kwargs):
        raise AssetNotReadyError("附件尚未完成解析，不能在本轮使用")

    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=False))

    events = [
        event
        async for event in _service(bus).run_process(
            "问题",
            identity_scope=_u1_scope(),
            process_id="process-domain-error",
        )
    ]

    errors = [event for event in events if event["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "workspace.asset.not_ready"
    assert "附件" in errors[0]["data"]["message"]
