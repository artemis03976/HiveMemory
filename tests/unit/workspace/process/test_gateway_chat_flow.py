"""主动聊天任务进程的 Gateway 编排测试。

被测边界：``TaskProcessService`` 注册入口与四阶段编排在命令短路、完成、
取消与失败路径上的行为，以及各阶段停止的响应点。注册经真实认证网关完成两阶段认证（组合内
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
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import (
    AssetNotReadyError,
    ScopeRequiredError,
)
from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnEvent,
    WorkspaceIdentity,
    WriteFocus,
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
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUExecutionStatus, SubmitWriteIntentRequest
from hivememory.workspace.process.outputs import OUTPUT_CHANNEL_CAPACITY
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from tests.helpers.chat_handoff import (
    expected_mtp_traces,
    make_mtp_turn_events,
)
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import make_task_process_service, wait_for_process_exit
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
    process_table: ProcessTable | None = None,
    event_publisher: RuntimeEventPublisher | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与阶段授权使用同一网关/授权者实例。"""
    composition = _composition()
    service = make_task_process_service(
        bus,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        process_table=process_table,
        event_publisher=event_publisher,
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


def _assert_sealed_payload(payload, *, turn_events: list[TurnEvent], submitted) -> None:
    """完成的进程交给 finalize 的交互记录逐字段等于各阶段产出。"""
    decision = _decision_outcome().decision
    assert payload.user_message == "问题"
    assert payload.rewritten_query == decision.rewritten_query
    assert payload.worth_saving is True
    assert payload.assistant_final_text == "完成"
    assert payload.turn_events == [
        *turn_events,
        TurnEvent(kind="assistant_message", sequence=5, role="assistant", content="完成"),
    ]
    assert payload.model_used == "glm-4"
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
            belong_to=(identity_scope).workspace_identity,
            interaction_id=interaction_id,
            topic_id=topic_id,
            is_new_topic=is_new_topic,
            memories=tuple(memories or ()),
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
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route())


@pytest.mark.asyncio
async def test_non_streaming_command_short_circuits_patchouli_and_cpu() -> None:
    """命令只解析不执行：进程把解析结果转为"暂不可用"终态，不调用 prepare/Profile/CPU。"""
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "/clear", process_id=f"process_{uuid4().hex}")

    assert result.kind == "command"
    assert result.command_execution_result.command_id == "system.clear"
    assert result.command_execution_result.status == CommandExecutionStatus.NOT_IMPLEMENTED
    assert result.command_execution_result.error_code == "command.unavailable"
    assert result.command_execution_result.client_action is None
    assert cpu.calls == []
    assert bus.list_routes() == [GatewayPublicRoutes.PROCESS, PatchouliRoutes.MEMORY_INTENT_SUBMIT]


@pytest.mark.asyncio
async def test_non_streaming_decision_uses_one_prepare_cpu_finalize_sequence() -> None:
    bus = GlobalSystemBus()
    calls: list[str] = []

    class _RecordingCPU(ScriptedCPU):
        async def execute(self, manifest, **kwargs):
            calls.append("cpu")
            return await super().execute(manifest, **kwargs)

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

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

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
    submitted = []

    async def write(submit):
        submitted.append(
            await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="记住这一点")))
        )

    finalize_kwargs: dict = {}
    cpu = ScriptedCPU(
        result=make_cpu_result(
            turn_events=turn_events,
        ),
        operation_script=write,
    )

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-seal-once")

    assert result.kind == "agent"
    _assert_sealed_payload(
        finalize_kwargs["payload"],
        turn_events=turn_events,
        submitted=submitted,
    )


@pytest.mark.asyncio
async def test_completed_streaming_process_seals_interaction_payload() -> None:
    """完成的流式进程：与非流式共用同一封口，CPU 终态结果逐字段进入交互记录。"""
    bus = GlobalSystemBus()
    turn_events = make_mtp_turn_events()
    submitted = []

    async def write(submit):
        submitted.append(
            await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="记住这一点")))
        )

    finalize_kwargs: dict = {}
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完成"}}],
        result=make_cpu_result(
            turn_events=turn_events,
        ),
        operation_script=write,
    )

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    service, composition = await _service(bus, cpu)
    events = await _stream_events(composition, service, "问题", process_id="process-seal-stream")

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    _assert_sealed_payload(
        finalize_kwargs["payload"],
        turn_events=turn_events,
        submitted=submitted,
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
                "topic_id",
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
                "topic_id",
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
                "topic_id",
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
        ),
    )
    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

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
        GatewayPublicRoutes.PROCESS,
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
    done 的话题标注与 memory_refs 由进程从 prepare 结果推导。"""
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

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    service, composition = await _service(bus, cpu)
    events = await _stream_events(
        composition,
        service,
        "问题",
        process_id="process-complete",
    )

    assert events[0]["event"] == "process_id"
    assert events[0]["data"]["process_id"] == "process-complete"
    assert events[0]["data"]["session_id"].startswith("session_")
    assert events[-1]["data"]["topic_id"] == "topic-1"
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

    async def finalize(**kwargs):
        captured["finalize"] = kwargs
        return []

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, topic_list)

    service, composition = await _service(bus)
    # 流式交付只保留阶段调用，话题池由独立管理入口读取。
    await _stream_events(composition, service, "问题", process_id="process-scope-only")

    assert "access" not in captured["gateway"]
    assert captured["gateway"]["identity_scope"] == _expected_scope()
    assert "access" not in captured["prepare"]
    assert captured["prepare"]["identity_scope"] == _expected_scope()
    assert captured["finalize"]["identity_scope"] == _expected_scope()
    assert "access" not in captured["finalize"]
    # finalize 后不再请求话题池。
    assert "topic_pool" not in captured


class _AdmissionChangingAuthorizer:
    """阶段授权使用实时准入：Actor 执行后切换到已撤销的真实访问登记。"""

    def __init__(self, delegate):
        self._delegate = delegate
        self._revoked = make_access_composition([]).authorizer
        self.revoked = False

    def authorize_stage(self, access, target_workspace):
        authorizer = self._revoked if self.revoked else self._delegate
        return authorizer.authorize_stage(access, target_workspace)

    def authorize_operation(self, access, operation, target_workspace):
        return self._delegate.authorize_operation(access, operation, target_workspace)

    def get_stage_operations(self, access, target_workspace):
        authorizer = self._revoked if self.revoked else self._delegate
        return authorizer.get_stage_operations(access, target_workspace)

    def authorize_process_control(self, requestor, record_access):
        return self._delegate.authorize_process_control(requestor, record_access)


@pytest.mark.asyncio
async def test_finalize_reauthorizes_and_revoked_admission_closes_completed_cpu_process() -> None:
    """Actor 完成前准入撤销：不提交交互，进程失败且 cleanup 不越过准入检查。"""
    bus = GlobalSystemBus()
    finalized = []
    cleaned = []
    composition = _composition()
    authorizer = _AdmissionChangingAuthorizer(composition.authorizer)

    async def revoke_admission(_submit):
        authorizer.revoked = True

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=revoke_admission)

    async def finalize(**kwargs):
        finalized.append(kwargs)
        return []

    async def cleanup(**kwargs):
        cleaned.append(kwargs)

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=authorizer,
    )
    issued = _capture_issued_contexts(composition)
    with pytest.raises(ScopeRequiredError) as error:
        await _run_once(composition, service, "问题", process_id="process-finalize-denied")

    assert error.value.details == {"reason": "actor_not_admitted"}
    assert len(cpu.calls) == 1
    assert cpu.finished is True
    assert finalized == []
    assert cleaned == []
    assert composition.gateway.describe_context(issued[0]) is None


@pytest.mark.asyncio
async def test_cleanup_admission_denial_skips_route_and_still_closes_process(caplog) -> None:
    """Actor 失败前准入撤销：cleanup 只记录拒绝警告，进程注销与 context 失效照常。"""
    bus = GlobalSystemBus()
    cleaned = []
    composition = _composition()
    authorizer = _AdmissionChangingAuthorizer(composition.authorizer)

    async def revoke_admission(_submit):
        authorizer.revoked = True

    async def cleanup(**kwargs):
        cleaned.append(kwargs)

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)
    cpu = ScriptedCPU(
        result=make_cpu_result(status=CPUExecutionStatus.FAILED.value, final_text=""),
        operation_script=revoke_admission,
    )
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=authorizer,
    )
    issued = _capture_issued_contexts(composition)
    result = await _run_once(composition, service, "问题", process_id="process-cleanup-denied")

    assert result.execution_result.status == "failed"
    assert cleaned == []
    assert "清理 prepared run 失败" in caplog.text
    assert composition.gateway.describe_context(issued[0]) is None


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
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
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.CANCELLED, final_text=""))

    async def prepare(**kwargs):
        prepared = await _scoped_prepared_route()(**kwargs)
        prepared_holder["prepared"] = prepared
        return prepared

    cleanup = AsyncMock(return_value=True)
    bus.register(
        GatewayPublicRoutes.PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        PatchouliRoutes.PREPARE_AGENT_RUN,
        prepare,
    )
    _register_profile(bus)
    bus.register(
        PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN,
        cleanup,
    )

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-cancel-cleanup")

    assert result.kind == "agent"
    assert result.execution_result.status == CPUExecutionStatus.CANCELLED.value
    # cleanup 只在唯一的关闭流程中补偿一次本进程 prepare 的结果，scope 来自
    # 关闭前的阶段再授权。
    cleanup.assert_awaited_once_with(
        prepared_run=prepared_holder["prepared"], identity_scope=_expected_scope()
    )


@pytest.mark.asyncio
async def test_non_streaming_failed_agent_run_is_not_rewritten_as_cancelled() -> None:
    bus = GlobalSystemBus()
    finalize = AsyncMock(return_value=[])
    cleanup = AsyncMock(return_value=True)
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED, final_text=""))
    prepared_holder: dict = {}
    bus.register(
        GatewayPublicRoutes.PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        PatchouliRoutes.PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service, composition = await _service(bus, cpu)
    result = await _run_once(composition, service, "问题", process_id="process-cpu-failed")

    assert result.execution_result.status == CPUExecutionStatus.FAILED.value
    finalize.assert_not_awaited()
    cleanup.assert_any_await(
        prepared_run=prepared_holder["prepared"], identity_scope=_expected_scope()
    )


@pytest.mark.asyncio
async def test_streaming_cpu_self_reported_cancel_skips_finalize_and_reports_stopped_done() -> None:
    """CPU 自报取消且无输出：流式交付发出 cancelled 的 done，finalize 路由全程未被调用。"""
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.CANCELLED, final_text=""))
    finalize_calls: list = []

    async def finalize(**kwargs):
        finalize_calls.append(kwargs)
        return []

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    _register_profile(bus)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, cpu)
    events = await _stream_events(composition, service, "hello", process_id="process-cpu-cancel")

    done_events = [event for event in events if event["event"] == "done"]
    assert [(event["data"]["status"], event["data"]["stopped"]) for event in done_events] == [
        ("cancelled", True)
    ]
    assert finalize_calls == []


@pytest.mark.asyncio
async def test_streaming_failed_agent_run_preserves_failed_done_status() -> None:
    bus = GlobalSystemBus()
    cpu = ScriptedCPU(result=make_cpu_result(status=CPUExecutionStatus.FAILED, final_text=""))
    bus.register(
        GatewayPublicRoutes.PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    _register_profile(bus)
    prepared_holder: dict = {}
    bus.register(
        PatchouliRoutes.PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    cleanup = AsyncMock(return_value=True)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)

    service, composition = await _service(bus, cpu)
    events = await _stream_events(
        composition, service, "问题", process_id="process-cpu-failed-stream"
    )

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == CPUExecutionStatus.FAILED.value
    assert events[-1]["data"]["stopped"] is True
    cleanup.assert_any_await(
        prepared_run=prepared_holder["prepared"], identity_scope=_expected_scope()
    )


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

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, recording_prepare)
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)
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
    cleanup.assert_any_await(
        prepared_run=prepared_holder["prepared"], identity_scope=_expected_scope()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_on", ["cpu", "channel"])
async def test_actor_stop_takes_effect_while_waiting_on_cpu_or_channel(blocked_on: str) -> None:
    """Actor 中的停止立即生效：子 task 在等 CPU 的下一项，或因订阅者暂停阻塞在通道写入上。

    订阅者读到 process_id 后不再读取；通道写满时 CPU 不会被继续推进（背压），
    停止仍取消子 task、关闭 CPU 输出流，并按 Actor 阶段以取消收口。
    """
    bus = GlobalSystemBus()
    token = {"event": "token", "data": {"content": "late"}}
    cpu = ScriptedCPU(
        events=[token] * (1 if blocked_on == "cpu" else OUTPUT_CHANNEL_CAPACITY + 1),
        result=make_cpu_result(),
        hang_before_result=True,
    )
    finalize = AsyncMock()
    cleanup = AsyncMock(return_value=True)
    prepared_holder: dict = {}
    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        PatchouliRoutes.PREPARE_AGENT_RUN,
        _recording_prepared_route(prepared_holder),
    )
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)
    sink = RecordingRuntimeEventSink()
    table = ProcessTable()
    service, composition = await _service(
        bus, cpu, process_table=table, event_publisher=RuntimeEventPublisher(sink)
    )
    handle = await _register(composition, service, "问题", process_id="process-actor-stop")
    stream = service.run_process(handle, stream=True)
    first = await anext(stream)

    if blocked_on == "cpu":
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)
    else:
        await asyncio.wait_for(cpu.last_event_entered.wait(), timeout=1)
        assert cpu.hang_entered.is_set() is False
    stop_result = service.cancel_process(handle, reason="user_requested")
    await wait_for_process_exit(table, "process-actor-stop")
    events = [first, *[event async for event in stream]]

    assert stop_result.cancelled is True
    assert cpu.finished is True
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once_with(
        prepared_run=prepared_holder["prepared"], identity_scope=_expected_scope()
    )
    cancelled = [
        event for event in sink.events if event.event_type == RuntimeEventType.CHAT_RUN_CANCELLED
    ]
    assert [event.data for event in cancelled] == [{"phase": "actor"}]


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
    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        PatchouliRoutes.PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup)
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
async def test_streaming_workspace_domain_error_yields_safe_code() -> None:
    """捕获附件拒绝等 Workspace 领域错误丢失安全 code 或被包装成系统错误。"""
    bus = GlobalSystemBus()

    async def prepare(**_kwargs):
        raise AssetNotReadyError("secret-token@internal-host 附件尚未完成解析")

    bus.register(GatewayPublicRoutes.PROCESS, AsyncMock(return_value=_decision_outcome()))
    _register_profile(bus)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    bus.register(PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=False))

    service, composition = await _service(bus)
    handle = await _register(composition, service, "问题", process_id="process-domain-error")
    events = [event async for event in service.run_process(handle, stream=True)]

    errors = [event for event in events if event["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "workspace.asset.not_ready"
    assert errors[0]["data"]["message"] == "请求未能完成，请检查权限、资源和会话状态"
