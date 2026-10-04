"""Phase 3F 主动聊天 Gateway 编排测试。

被测边界：``TaskProcessService`` 注册入口与四阶段编排在命令短路、完成、
取消、失败与断流路径上的行为。注册经真实认证网关完成两阶段认证（组合内
显式登记）并返回进程句柄；停止请求经句柄 stop 入口注入；Actor 阶段以
测试 CPU 替换 Alice（总线上不注册 Alice 路由），子系统路由用
GlobalSystemBus 上的替身隔离。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import (
    AssetNotReadyError,
    ScopeRequiredError,
    WorkspaceDomainError,
)
from hivememory.core.models import (
    ActorIdentity,
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnEvent,
    WorkspaceAssetRef,
    WorkspaceIdentity,
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
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessStatusSnapshot
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_mtp_turn_events,
    make_write_materialize_task,
)
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

_USER = "u1"
_AGENT = "omni_doll"


def _actor() -> ActorIdentity:
    """注册入口的 actor 声明（认证前不组装 IdentityScope）。"""
    return ActorIdentity(user_id=_USER, agent_id=_AGENT)


def _workspace() -> WorkspaceIdentity:
    """注册入口的请求进入 workspace 声明。"""
    return make_workspace_identity(owner_user_id=_USER)


def _expected_scope() -> IdentityScope:
    """认证声明经操作授权者授权规则应组装出的 IdentityScope。"""
    return IdentityScope(actor_identity=_actor(), workspace_identity=_workspace())


def _composition() -> AccessTestComposition:
    """u1/omni_doll 全 operation 的访问组合：注册声明与请求方 context 的签发来源。"""
    return make_access_composition([make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT)])


def _capture_issued_contexts(composition: AccessTestComposition) -> list[WorkspaceAccessContext]:
    """包装认证网关捕获每次签发的 context：注册入口签发即绑定，供失效断言使用。"""
    issued: list[WorkspaceAccessContext] = []
    original = composition.gateway.authenticate

    async def authenticate(**kwargs):
        context = await original(**kwargs)
        issued.append(context)
        return context

    composition.gateway.authenticate = authenticate  # type: ignore[method-assign]
    return issued


async def _service(
    bus: GlobalSystemBus,
    cpu: ScriptedCPU | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与阶段授权使用同一网关/授权者实例。"""
    composition = _composition()
    service = TaskProcessService(
        bus,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
    )
    return service, composition


async def _register(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
) -> ProcessHandle:
    """按组合的默认声明注册进程：两阶段认证由注册入口完成。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=_actor(),
        workspace=_workspace(),
        process_id=process_id,
        message=message,
        **kwargs,
    )


async def _run_once(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
):
    handle = await _register(composition, service, message, process_id=process_id, **kwargs)
    return await service.run_process(handle, stream=False)


async def _stream_events(
    composition: AccessTestComposition,
    service: TaskProcessService,
    message: str,
    *,
    process_id: str,
    **kwargs,
) -> list[dict]:
    handle = await _register(composition, service, message, process_id=process_id, **kwargs)
    return [event async for event in service.run_process(handle, stream=True)]


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
        interaction_id,
        **_kwargs,
    ):
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            topic_id=topic_id,
            is_new_topic=is_new_topic,
            topic_context=None,
            pool_topics=[],
            retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
            storage_available=True,
        )

    return route


def _recording_prepared_route(holder: dict):
    """记录 prepare 结果的替身：供关闭路径断言 cleanup 补偿的参数。"""
    base = _scoped_prepared_route()

    async def route(**kwargs):
        prepared = await base(**kwargs)
        holder["prepared"] = prepared
        return prepared

    return route


def _profile_route():
    """PATCHOULI_GET_AGENT_PROFILE 替身：返回 builtin omni_doll Profile。"""
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    async def route(agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    return route


def _register_profile(bus: GlobalSystemBus) -> None:
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())


@pytest.mark.asyncio
async def test_non_streaming_command_short_circuits_patchouli_and_cpu() -> None:
    """命令只解析不执行：进程把解析结果转为"暂不可用"终态，不调用 prepare/Profile/CPU。"""
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "/clear", process_id=f"process_{uuid4().hex}")

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

    service, composition = await _service(bus, _RecordingCPU(result=make_cpu_result()))
    result = await _run_once(composition, service, "问题", process_id="process-sequence")

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

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-seal-once")

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

    service, composition = await _service(bus, cpu)
    events = await _stream_events(composition, service, "问题", process_id="process-seal-stream")

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    _assert_sealed_payload(
        finalize_kwargs["payload"],
        turn_events=turn_events,
        write_task=write_task,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_keys"),
    [
        (
            CPUExecutionStatus.COMPLETED,
            {
                "process_id",
                "status",
                "final_text",
                "model_used",
                "stopped",
                "reason",
                "memory_task_ids",
                "pool_topics",
            },
        ),
        (
            CPUExecutionStatus.FAILED,
            {
                "process_id",
                "status",
                "final_text",
                "model_used",
                "stopped",
                "reason",
                "memory_task_ids",
            },
        ),
        (
            CPUExecutionStatus.CANCELLED,
            {
                "process_id",
                "status",
                "final_text",
                "model_used",
                "stopped",
                "reason",
                "memory_task_ids",
            },
        ),
    ],
)
async def test_streaming_done_omits_sealing_only_result_fields(
    status: CPUExecutionStatus,
    expected_keys: set[str],
) -> None:
    """流式 done 只携带交付方使用的执行结果字段：轮次事件与物化任务只用于封口，不下发。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(
        result=make_cpu_result(
            status=status,
            turn_events=make_mtp_turn_events(),
            materialize_tasks=[make_write_materialize_task()],
        ),
    )
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    service, composition = await _service(bus, cpu)
    events = await _stream_events(
        composition, service, "问题", process_id=f"process-done-{status.value}"
    )

    done = events[-1]
    assert done["event"] == "done"
    assert set(done["data"]) == expected_keys
    assert done["data"]["status"] == status.value
    assert done["data"]["final_text"] == "完成"
    assert done["data"]["model_used"] == "glm-4"


@pytest.mark.asyncio
async def test_streaming_command_emits_result_and_done_only() -> None:
    """流式命令请求：command_result 为"暂不可用"终态且不带客户端动作，随后 done 收口。"""
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_command_outcome()),
    )

    service, composition = await _service(bus)
    events = await _stream_events(
        composition, service, "/clear", process_id="process-stream-command"
    )

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

    service, composition = await _service(bus, cpu)
    events = await _stream_events(
        composition,
        service,
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
async def test_stage_routes_receive_only_authorizer_assembled_identity_scope() -> None:
    """阶段授权之后，Gateway、prepare 与结算后话题池读取只收到操作授权者组装的
    IdentityScope：无 access，也不取自 prepare 结果。"""
    bus = GlobalSystemBus()
    captured: dict[str, dict] = {}

    async def gateway(**kwargs):
        captured["gateway"] = kwargs
        return _decision_outcome()

    async def prepare(**kwargs):
        captured["prepare"] = kwargs
        return await _scoped_prepared_route()(**kwargs)

    async def topic_list(**kwargs):
        captured["topic_pool"] = kwargs
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, topic_list)

    service, composition = await _service(bus)
    # 话题池读取只发生在流式交付（结算后、done 之前）。
    await _stream_events(composition, service, "问题", process_id="process-scope-only")

    assert "access" not in captured["gateway"]
    assert captured["gateway"]["identity_scope"] == _expected_scope()
    assert "access" not in captured["prepare"]
    assert captured["prepare"]["identity_scope"] == _expected_scope()
    # 结算后的话题池读取同样只收到授权返回的 scope（不取自 prepared_run）。
    assert "access" not in captured["topic_pool"]
    assert captured["topic_pool"]["identity_scope"] == _expected_scope()


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    service, composition = await _service(bus)
    handle = await _register(composition, service, "问题", process_id="process-gateway")

    task = asyncio.create_task(service.run_process(handle, stream=False))
    await started.wait()
    stop_result = service.cancel_process(handle, reason="user_requested")
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

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-cancel-cleanup")

    assert result.kind == "agent"
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    # cleanup 只补偿本进程 prepare 的结果：不做阶段 operation 检查，也不接收凭据。
    # 注：交付收口会经 close_process 再次 close，cleanup 补偿当前会重复发布
    # （疑似生产缺陷，见迁移报告），这里固定补偿以本进程 prepared_run 发生。
    cleanup.assert_any_await(prepared_run=prepared_holder["prepared"])


@pytest.mark.asyncio
async def test_non_streaming_failed_agent_run_is_not_rewritten_as_cancelled() -> None:
    bus = GlobalSystemBus()
    finalize = AsyncMock(return_value=[])
    cleanup = AsyncMock(return_value=True)
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))
    prepared_holder: dict = {}
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-cpu-failed")

    assert result.execution_result.status == CPUExecutionStatus.FAILED.value
    finalize.assert_not_awaited()
    cleanup.assert_any_await(prepared_run=prepared_holder["prepared"])


@pytest.mark.asyncio
async def test_streaming_failed_agent_run_preserves_failed_done_status() -> None:
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED))
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    _register_profile(bus)
    prepared_holder: dict = {}
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service, composition = await _service(bus, cpu)
    events = await _stream_events(
        composition, service, "问题", process_id="process-cpu-failed-stream"
    )

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == CPUExecutionStatus.FAILED.value
    assert events[-1]["data"]["stopped"] is True
    cleanup.assert_any_await(prepared_run=prepared_holder["prepared"])


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
    prepared_holder: dict = {}

    async def recording_prepare(**kwargs):
        prepared = await prepare(**kwargs)
        prepared_holder["prepared"] = prepared
        return prepared

    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, recording_prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service, composition = await _service(bus, cpu)
    handle = await _register(composition, service, "问题", process_id="process-prepare")

    task = asyncio.create_task(service.run_process(handle, stream=False))
    await prepare_started.wait()
    stop_result = service.cancel_process(handle, reason="user_requested")
    release_prepare.set()
    result = await task

    assert stop_result.cancelled is True
    assert prepare_cancelled is False
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    assert cpu.calls == []
    finalize.assert_not_awaited()
    cleanup.assert_any_await(prepared_run=prepared_holder["prepared"])


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
    prepared_holder: dict = {}
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service, composition = await _service(bus, cpu)
    handle = await _register(composition, service, "问题", process_id="process-stream-cancel")

    task = asyncio.create_task(_collect_stream(service.run_process(handle, stream=True)))
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)
    stop_result = service.cancel_process(handle, reason="user_requested")
    events = await task

    assert stop_result.cancelled is True
    assert cpu.closed is True
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    finalize.assert_not_awaited()
    cleanup.assert_any_await(prepared_run=prepared_holder["prepared"])


async def _collect_stream(stream) -> list[dict]:
    return [event async for event in stream]


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
    service, composition = await _service(bus)
    handle = await _register(composition, service, "问题", process_id="process-finalize")

    task = asyncio.create_task(service.run_process(handle, stream=False))
    await finalize_started.wait()
    stop_result = service.cancel_process(handle, reason="user_requested")
    release_finalize.set()
    result = await task

    assert stop_result.cancelled is False
    assert stop_result.reason == "already_finalizing"
    assert result.execution_result.status == CPUExecutionStatus.COMPLETED.value
    cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_delivery_closes_process_and_invalidates_context() -> None:
    """交付结束自动收口：绑定 context 撤销（授权被拒），进程从表中注销。"""
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))

    service, composition = await _service(bus)
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    handle = await _register(composition, service, "问题", process_id="process-autoclose")

    snapshot = service.process_status("process-autoclose", access=requestor)
    assert snapshot == ProcessStatusSnapshot(
        process_id="process-autoclose",
        phase="created",
        status="running",
        reason=None,
    )

    result = await service.run_process(handle, stream=False)
    assert result.kind == "agent"

    assert service.process_status("process-autoclose", access=requestor) is None
    cancel_result = service.cancel_process("process-autoclose", access=requestor)
    assert cancel_result.cancelled is False
    assert cancel_result.status == "not_found"
    assert cancel_result.process_id == "process-autoclose"
    assert composition.gateway.describe_context(issued[0]) is None
    with pytest.raises(ScopeRequiredError) as excinfo:
        composition.authorizer.authorize_operation(
            issued[0],
            WorkspaceOperation.RESOURCE_READ,
            _workspace(),
        )
    assert excinfo.value.details["reason"] == "context_not_issued"


@pytest.mark.asyncio
async def test_attachment_selection_without_reader_fails_allocation() -> None:
    """捕获装配遗漏：有附件选择但 Store 未注入时，分配显式失败且不进入 CPU。"""
    bus = GlobalSystemBus()
    cleanup = AsyncMock(return_value=True)
    cpu = ScriptedCPU(result=make_cpu_result())
    prepared_holder: dict = {}
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service, composition = await _service(bus, cpu)
    with pytest.raises(WorkspaceDomainError, match="附件读取能力"):
        await _run_once(
            composition,
            service,
            "问题",
            process_id="process-no-reader",
            attachments=[
                AttachmentSelectionRequest(
                    asset_ref=WorkspaceAssetRef(token="ref-a", asset_id="asset-a"),
                ),
            ],
        )

    assert cpu.calls == []
    cleanup.assert_any_await(prepared_run=prepared_holder["prepared"])


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

    service, composition = await _service(bus)
    handle = await _register(composition, service, "问题", process_id="process-domain-error")
    events = [event async for event in service.run_process(handle, stream=True)]

    errors = [event for event in events if event["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "workspace.asset.not_ready"
    assert "附件" in errors[0]["data"]["message"]
