"""主动聊天任务进程的 Gateway 编排测试。

被测边界：``TaskProcessService`` 注册入口与四阶段编排在命令短路、完成、
取消与失败路径上的行为，以及各阶段停止的响应点。注册经真实认证网关完成两阶段认证（组合内
显式登记）并返回进程句柄；停止请求以 process_id 加请求级 context 注入；Actor 阶段以
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
    make_mtp_turn_events,
)
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import (
    collect_events,
    done_data,
    error_code,
    make_task_process_service,
    wait_for_process_exit,
)
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
) -> list[dict]:
    """注册并经流式交付运行到结束，返回全部流式事件。"""
    handle = await _register(composition, service, message, process_id=process_id, **kwargs)
    return await collect_events(service, handle)


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
    """完成的进程交给 finalize 的交互记录逐字段等于 CPU 的执行结果与实际使用的附件。

    Gateway 的改写与价值预判不进入交互记录。
    """
    assert payload.turn_events == [
        *turn_events,
        TurnEvent(kind="assistant_message", sequence=3, content="完成"),
    ]
    assert payload.model_used == "glm-4"
    # 本测试未选择附件：附件编译的实际使用集合为空
    assert payload.used_attachments == []


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
async def test_command_short_circuits_patchouli_and_cpu() -> None:
    """命令只解析不执行：进程把解析结果转为"暂不可用"终态，只发出 command_result 与 done，
    不调用 prepare/Profile/CPU。"""
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    cpu = ScriptedCPU(result=make_cpu_result())

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "/clear", process_id=f"process_{uuid4().hex}")

    assert [event["event"] for event in events] == ["process_id", "command_result", "done"]
    command_result = events[1]["data"]
    assert command_result["command_id"] == "system.clear"
    assert command_result["status"] == CommandExecutionStatus.NOT_IMPLEMENTED.value
    assert command_result["error_code"] == "command.unavailable"
    assert command_result["client_action"] is None
    assert done_data(events)["final_text"] == "系统指令 /clear 暂不可用。"
    assert cpu.calls == []
    assert bus.list_routes() == [GatewayPublicRoutes.PROCESS, PatchouliRoutes.MEMORY_INTENT_SUBMIT]


@pytest.mark.asyncio
async def test_decision_uses_one_prepare_cpu_finalize_sequence() -> None:
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
    events = await _run_once(composition, service, "问题", process_id="process-sequence")

    assert done_data(events)["final_text"] == "完成"
    assert calls == ["gateway", "prepare", "cpu", "finalize"]


@pytest.mark.asyncio
async def test_model_prefetch_precedes_gateway_and_its_failure_does_not_affect_process() -> None:
    """进程在 Gateway 之前请求预取推理模型；预取失败只记录日志，进程照常完成。"""
    bus = GlobalSystemBus()
    calls: list[str] = []

    async def prefetch_models():
        calls.append("prefetch")
        raise RuntimeError("patchouli unavailable")

    async def gateway(**_kwargs):
        calls.append("gateway")
        return _decision_outcome()

    bus.register(PatchouliRoutes.PREFETCH_MODELS, prefetch_models)
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))

    service, composition = await _service(bus)
    events = await _run_once(composition, service, "问题", process_id="process-prefetch")

    assert done_data(events)["final_text"] == "完成"
    assert calls == ["prefetch", "gateway"]


@pytest.mark.asyncio
async def test_completed_process_seals_interaction_payload() -> None:
    """完成的进程：交给 finalize 的交互记录由进程封口，CPU 终态结果逐字段进入交互记录。"""
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
    events = await _run_once(composition, service, "问题", process_id="process-seal-stream")

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
    events = await _run_once(
        composition, service, "问题", process_id=f"process-done-{status.value}"
    )

    done = events[-1]
    assert done["event"] == "done"
    assert set(done["data"]) == expected_keys
    assert done["data"]["status"] == status.value
    assert done["data"]["final_text"] == "完成"
    assert done["data"]["model_used"] == "glm-4"


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
    events = await _run_once(
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
    await _run_once(composition, service, "问题", process_id="process-scope-only")

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
    events = await _run_once(composition, service, "问题", process_id="process-finalize-denied")

    assert error_code(events) == ScopeRequiredError.code
    assert len(cpu.calls) == 1
    assert cpu.finished is True
    assert finalized == []
    assert cleaned == []
    assert composition.is_revoked(issued[0]) is True


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
    events = await _run_once(composition, service, "问题", process_id="process-cleanup-denied")

    assert done_data(events)["status"] == "failed"
    assert cleaned == []
    assert "清理 prepared run 失败" in caplog.text
    assert composition.is_revoked(issued[0]) is True


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    service, composition = await _service(bus)
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, "问题", process_id="process-gateway")

    task = asyncio.create_task(collect_events(service, handle))
    await started.wait()
    stop_result = service.cancel_process("process-gateway", access=requestor)
    events = await task

    assert stop_result.cancelled is True
    assert done_data(events)["status"] == CPUExecutionStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_cpu_self_reported_cancel_skips_finalize_and_cleans_prepared_run() -> None:
    """CPU 自报取消且无输出：done 为 stopped 的 cancelled，不调用 finalize，并补偿 prepare 结果。"""
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
    finalize = AsyncMock(return_value=[])
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

    service, composition = await _service(bus, cpu)
    events = await _run_once(composition, service, "问题", process_id="process-cancel-cleanup")

    done = done_data(events)
    assert (done["status"], done["stopped"]) == (CPUExecutionStatus.CANCELLED.value, True)
    finalize.assert_not_awaited()
    # cleanup 只在唯一的关闭流程中补偿一次本进程 prepare 的结果，scope 来自
    # 关闭前的阶段再授权。
    cleanup.assert_awaited_once_with(
        prepared_run=prepared_holder["prepared"], identity_scope=_expected_scope()
    )


@pytest.mark.asyncio
async def test_failed_agent_run_keeps_failed_status_and_skips_finalize() -> None:
    """CPU 自报失败且无输出：done 保持 failed（不改记为取消），不调用 finalize，补偿 prepare 结果。"""
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
    events = await _run_once(composition, service, "问题", process_id="process-cpu-failed")

    done = done_data(events)
    assert (done["status"], done["stopped"]) == (CPUExecutionStatus.FAILED.value, True)
    finalize.assert_not_awaited()
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
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, "问题", process_id="process-prepare")

    task = asyncio.create_task(collect_events(service, handle))
    await prepare_started.wait()
    stop_result = service.cancel_process("process-prepare", access=requestor)
    release_prepare.set()
    events = await task

    assert stop_result.cancelled is True
    assert prepare_cancelled is False
    assert done_data(events)["status"] == CPUExecutionStatus.CANCELLED.value
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
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, "问题", process_id="process-actor-stop")
    stream = service.run_process(handle)
    first = await anext(stream)

    if blocked_on == "cpu":
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=1)
    else:
        await asyncio.wait_for(cpu.last_event_entered.wait(), timeout=1)
        assert cpu.hang_entered.is_set() is False
    stop_result = service.cancel_process("process-actor-stop", access=requestor)
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
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, "问题", process_id="process-finalize")

    task = asyncio.create_task(collect_events(service, handle))
    await finalize_started.wait()
    stop_result = service.cancel_process("process-finalize", access=requestor)
    release_finalize.set()
    events = await task

    assert stop_result.cancelled is False
    assert stop_result.reason == "already_finalizing"
    assert done_data(events)["status"] == CPUExecutionStatus.COMPLETED.value
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
    events = await collect_events(service, handle)

    errors = [event for event in events if event["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "workspace.asset.not_ready"
    assert errors[0]["data"]["message"] == "请求未能完成，请检查权限、资源和会话状态"
