"""任务进程 stop 控制契约测试。

覆盖两层契约：进程记录的 stop 语义（request_stop）与进程表作为唯一进程
注册表的登记/注销（以 process_id 为键登记任务进程、重复 process_id 拒绝、
注销只移除同一对象），以及注册入口句柄 API 的访问边界——注册认证失败
不创建、不登记进程；注册成功返回只暴露 process_id 的句柄；句柄按对象身份
判定有效，重新构造的句柄与进程关闭后的旧句柄不能运行、取消或关闭进程；
流未开始的 close_process 使 context 失效并从表中注销；交付以任何结局结束
后 context 都已失效；取消只有一个方法，句柄形式与控制请求形式发布相同
事件，句柄形式不经进程控制授权，错误的参数组合被拒绝；认证成功后登记
失败使已签发 context 失效。跨 workspace 请求方的控制请求统一按 not_found
呈现；context 失效的请求方以 ScopeRequiredError 拒绝。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink, RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import (
    AdmissionDeniedError,
    ScopeRequiredError,
    WorkspaceDomainError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    IdentityScope,
    ResolvedAgentProfile,
    WorkspaceIdentity,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.authentication import AccessGrantSummary
from hivememory.workspace.process.events import BoundProcessEvents
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessStatusSnapshot,
    ProcessTable,
)
from hivememory.workspace.process.task_process import _run_interruptible
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

_USER = "u1"
_AGENT = "omni_doll"
_WORKSPACE_ID = "main_workspace"


def _actor(*, user_id: str = _USER, agent_id: str = _AGENT) -> ActorIdentity:
    """注册入口的 actor 声明（认证前不组装 IdentityScope）。"""
    return ActorIdentity(user_id=user_id, agent_id=agent_id)


def _workspace(
    *, owner_user_id: str = _USER, workspace_id: str = _WORKSPACE_ID
) -> WorkspaceIdentity:
    """注册入口的请求进入 workspace 声明。"""
    return make_workspace_identity(owner_user_id=owner_user_id, workspace_id=workspace_id)


def _record(process_id: str) -> ProcessRecord:
    """记录级测试的进程记录：access 是不透明凭据，记录级 stop 语义不使用它。"""
    return ProcessRecord(
        process_id=process_id,
        access=WorkspaceAccessContext(),
        events=BoundProcessEvents(RuntimeEventPublisher(NullRuntimeEventSink())),
    )


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
    bus: GlobalSystemBus | None = None,
    *,
    composition: AccessTestComposition | None = None,
    cpu: ScriptedCPU | None = None,
    event_publisher: RuntimeEventPublisher | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与控制授权使用同一网关/授权者实例。"""
    bus = bus or GlobalSystemBus()
    composition = composition or _composition()
    service = TaskProcessService(
        bus,
        event_publisher,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
    )
    return service, composition


async def _register(
    composition: AccessTestComposition,
    service: TaskProcessService,
    *,
    process_id: str,
    actor: ActorIdentity | None = None,
    workspace: WorkspaceIdentity | None = None,
    message: str = "问题",
) -> ProcessHandle:
    """按组合的默认声明注册进程：两阶段认证由注册入口完成。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=actor or _actor(),
        workspace=workspace or _workspace(),
        process_id=process_id,
        message=message,
    )


def _bus_until_finalize() -> GlobalSystemBus:
    """Gateway → Profile → prepare → finalize 的替身总线：交付到 completed。"""
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(_agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, identity_scope, interaction_id, **_kwargs):
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            topic_id="topic-control",
            is_new_topic=False,
            retrieval_result=RetrievalResponse(),
        )

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=True))
    return bus


def _assert_context_invalidated(composition: AccessTestComposition, context) -> None:
    """绑定 context 已随进程关闭失效：诊断查询返回 None，授权兑现被拒绝。"""
    assert composition.gateway.describe_context(context) is None
    with pytest.raises(ScopeRequiredError) as excinfo:
        composition.authorizer.authorize_operation(
            context, WorkspaceOperation.RESOURCE_READ, _workspace()
        )
    assert excinfo.value.details["reason"] == "context_not_issued"


# ========== 进程记录的 stop 语义 ==========


@pytest.mark.asyncio
async def test_gateway_stop_cancels_bound_task_immediately() -> None:
    run = _record("process-1")
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    run.bind_phase(ProcessPhase.GATEWAY, task)

    result = run.request_stop()

    assert result.accepted is True
    assert result.reason == "user_requested"
    assert run.outcome is ProcessOutcome.STOP_REQUESTED
    assert run.phase is ProcessPhase.GATEWAY
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_prepare_stop_records_request_without_cancelling_prepare_task() -> None:
    run = _record("process-2")
    run.enter_phase(ProcessPhase.PREPARE)
    blocker = asyncio.Event()
    prepare_task = asyncio.create_task(blocker.wait())

    result = run.request_stop("during_prepare")

    assert result.accepted is True
    assert run.outcome is ProcessOutcome.STOP_REQUESTED
    assert run.stop_reason == "during_prepare"
    assert prepare_task.cancelled() is False
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task


def test_finalize_and_terminal_stop_are_rejected() -> None:
    finalizing = _record("process-3")
    finalizing.enter_phase(ProcessPhase.FINALIZE)
    result = finalizing.request_stop()
    assert result.accepted is False
    assert result.reason == "already_finalizing"
    assert finalizing.outcome is ProcessOutcome.RUNNING

    terminal = _record("process-4")
    terminal.phase = ProcessPhase.TERMINAL
    terminal.outcome = ProcessOutcome.COMPLETED
    result = terminal.request_stop()
    assert result.accepted is False
    assert result.reason == "already_terminal"
    assert terminal.outcome is ProcessOutcome.COMPLETED


def test_repeated_stop_keeps_first_reason_and_does_not_cancel_again() -> None:
    run = _record("process-5")
    task = MagicMock()
    task.done.return_value = False
    run.bind_phase(ProcessPhase.ACTOR, task)

    first = run.request_stop("first_reason")
    second = run.request_stop("second_reason")

    assert first.accepted is True
    assert second.accepted is True
    assert second.reason == "first_reason"
    task.cancel.assert_called_once_with()


def test_stop_after_bound_task_finished_is_accepted_without_second_cancel() -> None:
    run = _record("process-7")
    task = MagicMock()
    task.done.return_value = True
    run.bind_phase(ProcessPhase.GATEWAY, task)

    result = run.request_stop("late_stop")

    assert result.accepted is True
    assert result.reason == "late_stop"
    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_owner_task_cancellation_is_not_translated_to_chat_run_cancelled() -> None:
    run = _record("process-8")
    blocker = asyncio.Event()

    async def operation():
        await blocker.wait()

    task = asyncio.create_task(_run_interruptible(run, ProcessPhase.GATEWAY, operation))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.outcome is ProcessOutcome.RUNNING


# ========== 进程表：登记、取回与注销 ==========


def _process_stand_in(process_id: str) -> SimpleNamespace:
    """进程表级测试的任务进程替身：进程表只经 ``record.process_id`` 寻址。"""
    return SimpleNamespace(record=_record(process_id))


def test_table_get_returns_registered_process_and_close_removes_it() -> None:
    """进程表登记任务进程本身：按 process_id 原样取回同一对象，注销后取回为 None。"""
    table = ProcessTable()
    process = _process_stand_in("process-1")
    table.register(process)

    assert table.get("process-1") is process
    assert table.get("missing") is None

    table.close(process)
    assert table.get("process-1") is None


def test_table_rejects_duplicate_process_id_without_overwriting() -> None:
    """防止重复 process_id 覆盖既有进程并把控制权转给后注册者。"""
    table = ProcessTable()
    original = _process_stand_in("collision")
    table.register(original)

    with pytest.raises(WorkspaceDomainError, match="拒绝覆盖"):
        table.register(_process_stand_in("collision"))

    assert table.get("collision") is original


def test_table_close_only_removes_the_same_process_object() -> None:
    """注销按对象身份：同 process_id 的另一个对象不能把已登记的进程注销掉。"""
    table = ProcessTable()
    registered = _process_stand_in("process-shared-id")
    table.register(registered)

    table.close(_process_stand_in("process-shared-id"))

    assert table.get("process-shared-id") is registered


# ========== 注册入口：认证、签发即绑定、登记 ==========


@pytest.mark.asyncio
async def test_register_rejects_system_actor_without_registering_process() -> None:
    """任务进程必须由具体 Agent 执行：保留 system 声明在认证前被拒绝且不登记。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)

    with pytest.raises(WorkspaceDomainError) as excinfo:
        await _register(
            composition,
            service,
            process_id="process-system-actor",
            actor=_actor(agent_id=SYSTEM_AGENT_ID),
        )

    assert excinfo.value.details == {"agent_id": SYSTEM_AGENT_ID}
    assert service.process_status("process-system-actor", access=requestor) is None


@pytest.mark.asyncio
async def test_register_authentication_failure_does_not_create_or_register_process() -> None:
    """认证失败（actor 与 workspace owner 不一致）：抛 AdmissionDeniedError，不创建、不登记。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)

    with pytest.raises(AdmissionDeniedError) as excinfo:
        await _register(
            composition,
            service,
            process_id="process-auth-fail",
            actor=_actor(user_id="u2"),
        )

    assert excinfo.value.details["reason"] == "actor_not_owner"
    assert service.process_status("process-auth-fail", access=requestor) is None


@pytest.mark.asyncio
async def test_register_success_returns_handle_and_binds_issued_context_to_process() -> None:
    """注册成功：返回句柄，签发 context 绑定本进程并已登记到表内。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)

    await _register(composition, service, process_id="process-registered")

    assert [context is not None for context in issued] == [True]
    bound_context = issued[0]
    assert service.process_status("process-registered", access=requestor) == ProcessStatusSnapshot(
        process_id="process-registered",
        phase="created",
        status="running",
        reason=None,
    )
    # 绑定 context 经操作授权者按注册声明组装可信 scope。
    scope = composition.authorizer.authorize_operation(
        bound_context, WorkspaceOperation.RESOURCE_READ, _workspace()
    )
    assert scope == IdentityScope(actor_identity=_actor(), workspace_identity=_workspace())
    # 签发即绑定：授予记录的运行绑定是本进程的 process_id（诊断查询）。
    assert composition.gateway.describe_context(bound_context) == AccessGrantSummary(
        actor_user_id=_USER,
        agent_id=_AGENT,
        workspace_id=_WORKSPACE_ID,
        principal_id=composition.principal.principal_id,
        run_type="task_process",
        run_id="process-registered",
    )


@pytest.mark.asyncio
async def test_process_handle_exposes_only_process_id() -> None:
    """句柄是入口 adapter 唯一引用：只暴露 process_id，不暴露记录、context 或进程容器。"""
    service, composition = await _service()
    handle = await _register(composition, service, process_id="process-opaque")

    assert handle.process_id == "process-opaque"
    assert [name for name in dir(handle) if not name.startswith("_")] == ["process_id"]
    assert not hasattr(handle, "record")
    assert not hasattr(handle, "access")
    assert not hasattr(handle, "task")
    assert not hasattr(handle, "request")


@pytest.mark.asyncio
async def test_close_process_before_stream_invalidates_context_and_deregisters() -> None:
    """流从未开始：close_process 使 context 失效（兑现被拒）并从表中注销。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    handle = await _register(composition, service, process_id="process-never-run")

    await service.close_process(handle)

    _assert_context_invalidated(composition, issued[0])
    cancel_result = service.cancel_process("process-never-run", access=requestor)
    assert cancel_result.cancelled is False
    assert cancel_result.status == "not_found"
    assert service.process_status("process-never-run", access=requestor) is None


@pytest.mark.asyncio
async def test_close_process_is_idempotent() -> None:
    """close_process 幂等：重复调用是空操作，不改变收口后的可见状态。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, process_id="process-close-twice")

    await service.close_process(handle)
    await service.close_process(handle)

    assert service.process_status("process-close-twice", access=requestor) is None


# ========== 注册入口：认证成功后登记失败的 context 失效 ==========


@pytest.mark.asyncio
async def test_registration_failure_after_authentication_invalidates_issued_context() -> None:
    """认证通过后、登记完成前失败（重复 process_id）：已签发 context 失效，无半注册记录。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    await _register(composition, service, process_id="process-dup")

    with pytest.raises(WorkspaceDomainError, match="拒绝覆盖"):
        await _register(composition, service, process_id="process-dup")

    # 第二次注册签发的 context 已随登记失败失效；不留悬挂凭据。
    _assert_context_invalidated(composition, issued[1])
    # 第一次注册的 context 不受影响，进程表内没有半注册记录：
    # 既有进程仍以原声明可控、可查。
    assert composition.gateway.describe_context(issued[0]) == AccessGrantSummary(
        actor_user_id=_USER,
        agent_id=_AGENT,
        workspace_id=_WORKSPACE_ID,
        principal_id=composition.principal.principal_id,
        run_type="task_process",
        run_id="process-dup",
    )
    assert service.process_status("process-dup", access=requestor) == ProcessStatusSnapshot(
        process_id="process-dup",
        phase="created",
        status="running",
        reason=None,
    )


# ========== 句柄 API：有效性、运行、取消与关闭 ==========


def _cancel_requested_events(sink: RecordingRuntimeEventSink) -> list:
    return [
        event
        for event in sink.events
        if event.event_type == RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED
    ]


@pytest.mark.asyncio
async def test_run_process_with_closed_handle_raises_domain_error() -> None:
    """进程关闭后的旧句柄：run_process 显式失败（process_handle_unknown）。"""
    service, composition = await _service()
    handle = await _register(composition, service, process_id="process-closed-handle")
    await service.close_process(handle)

    for kwargs in ({"stream": True}, {"stream": False}):
        with pytest.raises(WorkspaceDomainError) as excinfo:
            service.run_process(handle, **kwargs)
        assert excinfo.value.details["reason"] == "process_handle_unknown"
        assert excinfo.value.details["process_id"] == "process-closed-handle"


@pytest.mark.asyncio
async def test_reconstructed_handle_cannot_run_cancel_or_close_the_process() -> None:
    """按 process_id 重新构造的句柄无效：不能运行、取消或关闭进程，进程不受影响。"""
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(event_publisher=RuntimeEventPublisher(sink))
    requestor = await composition.authenticate(agent_id=_AGENT)
    await _register(composition, service, process_id="process-forged")
    # 入口 adapter 拿不到进程对象：只能用一个 process_id 相同的替身去构造句柄。
    forged = ProcessHandle(SimpleNamespace(record=SimpleNamespace(process_id="process-forged")))

    with pytest.raises(WorkspaceDomainError) as excinfo:
        service.run_process(forged, stream=False)
    assert excinfo.value.details["reason"] == "process_handle_unknown"
    assert service.cancel_process(forged, reason="client_disconnected") == CancelResult(
        process_id="process-forged",
        cancelled=False,
        status="not_found",
        reason="client_disconnected",
    )
    await service.close_process(forged)

    assert _cancel_requested_events(sink) == []
    assert service.process_status("process-forged", access=requestor) == ProcessStatusSnapshot(
        process_id="process-forged",
        phase="created",
        status="running",
        reason=None,
    )


@pytest.mark.asyncio
async def test_stale_handle_cannot_control_a_reregistered_process() -> None:
    """进程关闭后同一 process_id 重新登记：旧句柄不能取消或关闭新进程。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    stale = await _register(composition, service, process_id="process-reused-id")
    await service.close_process(stale)
    await _register(composition, service, process_id="process-reused-id")

    result = service.cancel_process(stale, reason="client_disconnected")
    await service.close_process(stale)

    assert result.status == "not_found"
    assert service.process_status("process-reused-id", access=requestor) == ProcessStatusSnapshot(
        process_id="process-reused-id",
        phase="created",
        status="running",
        reason=None,
    )


@pytest.mark.asyncio
async def test_handle_cancel_after_delivery_reports_not_found_without_event() -> None:
    """交付结束已收口的进程：句柄已失效，取消按 not_found 收口且不发布事件。"""
    bus = _bus_until_finalize()
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(bus, event_publisher=RuntimeEventPublisher(sink))
    handle = await _register(composition, service, process_id="process-cancelled-late")

    result = await service.run_process(handle, stream=False)
    assert result.kind == "agent"

    cancel_result = service.cancel_process(handle)
    assert cancel_result == CancelResult(
        process_id="process-cancelled-late",
        cancelled=False,
        status="not_found",
        reason="user_requested",
    )
    assert _cancel_requested_events(sink) == []


@pytest.mark.asyncio
async def test_control_request_not_found_publishes_event_with_requestor_label() -> None:
    """控制请求找不到进程：返回 not_found，并发布带请求方观测标签的事件。"""
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(event_publisher=RuntimeEventPublisher(sink))
    requestor = await composition.authenticate(agent_id=_AGENT)

    result = service.cancel_process("process-missing", access=requestor)

    assert result.status == "not_found"
    events = _cancel_requested_events(sink)
    assert [(event.process_id, event.workspace_id, event.status) for event in events] == [
        ("process-missing", _WORKSPACE_ID, "not_found")
    ]


@pytest.mark.asyncio
async def test_cancel_rejects_mismatched_arguments() -> None:
    """错误组合在接线处失败：句柄形式不接受 access，按 process_id 取消必须提供 access。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, process_id="process-arguments")

    with pytest.raises(TypeError, match="不接受 access"):
        service.cancel_process(handle, access=requestor)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="必须提供请求级 access"):
        service.cancel_process("process-arguments")  # type: ignore[call-overload]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_handle", [True, False], ids=["handle", "control_plane"])
async def test_accepted_cancel_publishes_identical_events(use_handle: bool) -> None:
    """句柄形式与控制请求形式的取消共用 stop 记录与事件发布：事件序列逐字段相同。"""
    gateway_started = asyncio.Event()
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        gateway_started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(
        bus, cpu=ScriptedCPU(result=make_cpu_result()), event_publisher=RuntimeEventPublisher(sink)
    )
    requestor = await composition.authenticate(agent_id=_AGENT)
    handle = await _register(composition, service, process_id="process-stop-parity")
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await gateway_started.wait()

    if use_handle:
        stop_result = service.cancel_process(handle, reason="user_requested")
    else:
        stop_result = service.cancel_process("process-stop-parity", access=requestor)
    await task

    assert stop_result.cancelled is True
    assert stop_result.reason == "user_requested"
    events = [event for event in sink.events if event.event_type.startswith("chat.run.")]
    assert [(event.event_type, event.status, event.reason, event.data) for event in events] == [
        (RuntimeEventType.CHAT_RUN_CREATED, "created", None, {}),
        (RuntimeEventType.CHAT_RUN_STATUS, "preparing", None, {}),
        (
            RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
            "stop_requested",
            "user_requested",
            {"cancelled": True},
        ),
        (RuntimeEventType.CHAT_RUN_STATUS, "stop_requested", "user_requested", {}),
        (RuntimeEventType.CHAT_RUN_CANCELLED, "cancelled", "user_requested", {"phase": "gateway"}),
    ]
    # 两个入口都使用注册时绑定的观测标签，请求方声明不重建身份坐标。
    assert {(event.process_id, event.workspace_id, event.agent_id) for event in events} == {
        ("process-stop-parity", _WORKSPACE_ID, _AGENT)
    }


@pytest.mark.asyncio
async def test_handle_cancel_skips_process_control_authorization() -> None:
    """句柄形式的取消不经进程控制授权：未签发的伪造 access 无法取消，持有句柄即可取消。"""
    gateway_started = asyncio.Event()
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        gateway_started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    service, composition = await _service(bus)
    handle = await _register(composition, service, process_id="process-handle-stop")
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await gateway_started.wait()

    # 取消入口需要经网关签发的请求级 context：伪造（未签发）凭据被拒绝。
    with pytest.raises(ScopeRequiredError) as excinfo:
        service.cancel_process("process-handle-stop", access=WorkspaceAccessContext())
    assert excinfo.value.details["reason"] == "context_not_issued"

    # 持有句柄即为生命周期所有者：不提交任何 access 也能取消自己的进程。
    stop_result = service.cancel_process(handle, reason="client_disconnected")
    await task

    assert stop_result.cancelled is True
    assert stop_result.reason == "client_disconnected"


# ========== 交付结束自动收口：context 随任何结局失效 ==========


@pytest.mark.asyncio
async def test_completed_delivery_invalidates_bound_context() -> None:
    """completed 结局：交付结束自动 close_process，绑定 context 失效、进程注销。"""
    bus = _bus_until_finalize()
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    service, composition = await _service(bus)
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    handle = await _register(composition, service, process_id="process-end-completed")

    result = await service.run_process(handle, stream=False)

    assert result.kind == "agent"
    _assert_context_invalidated(composition, issued[0])
    assert service.process_status("process-end-completed", access=requestor) is None


@pytest.mark.asyncio
async def test_failed_delivery_invalidates_bound_context() -> None:
    """失败结局（编排异常沿非流式上抛）：交付结束仍收口，绑定 context 失效。"""
    bus = _bus_until_finalize()
    service, composition = await _service(bus, cpu=ScriptedCPU(error=RuntimeError("cpu exploded")))
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    handle = await _register(composition, service, process_id="process-end-failed")

    with pytest.raises(RuntimeError, match="cpu exploded"):
        await service.run_process(handle, stream=False)

    _assert_context_invalidated(composition, issued[0])
    assert service.process_status("process-end-failed", access=requestor) is None


@pytest.mark.asyncio
async def test_cancelled_delivery_invalidates_bound_context() -> None:
    """取消结局：以句柄取消后进程以取消收口，绑定 context 失效。"""
    gateway_started = asyncio.Event()
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        gateway_started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    service, composition = await _service(bus)
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    handle = await _register(composition, service, process_id="process-end-cancelled")
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await gateway_started.wait()

    stop_result = service.cancel_process(handle, reason="user_requested")
    result = await task

    assert stop_result.cancelled is True
    assert result.execution_result.status == "cancelled"
    _assert_context_invalidated(composition, issued[0])
    assert service.process_status("process-end-cancelled", access=requestor) is None


# ========== 控制面：请求方与进程记录的驻留坐标比对 ==========


@pytest.mark.asyncio
async def test_control_requests_from_other_workspace_are_not_found() -> None:
    """跨 workspace 请求方：取消与状态查询统一 not_found，不泄露进程存在性。"""
    composition = make_access_composition(
        [
            make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT),
            make_actor_access_record(
                owner_user_id="u2",
                agent_id="other_agent",
                workspace_id="other_workspace",
            ),
        ]
    )
    service, composition = await _service(composition=composition)
    await _register(composition, service, process_id="process-shared")
    foreign_requestor = await composition.authenticate(
        agent_id="other_agent",
        user_id="u2",
        workspace=_workspace(owner_user_id="u2", workspace_id="other_workspace"),
    )

    rejected = service.cancel_process("process-shared", access=foreign_requestor)
    assert rejected.cancelled is False
    assert rejected.status == "not_found"
    assert rejected.process_id == "process-shared"
    assert service.process_status("process-shared", access=foreign_requestor) is None

    # 同驻留坐标的请求方仍可控：状态可读（not_found 不等于进程不存在）。
    owner_requestor = await composition.authenticate(agent_id=_AGENT)
    assert service.process_status(
        "process-shared", access=owner_requestor
    ) == ProcessStatusSnapshot(
        process_id="process-shared",
        phase="created",
        status="running",
        reason=None,
    )


@pytest.mark.asyncio
async def test_control_requests_with_invalidated_requestor_context_raise_scope_required() -> None:
    """context 失效的请求方：取消与状态查询以 ScopeRequiredError（context_not_issued）拒绝。"""
    service, composition = await _service()
    await _register(composition, service, process_id="process-controlled")
    requestor = await composition.authenticate(agent_id=_AGENT)
    composition.gateway.invalidate_context(requestor)

    with pytest.raises(ScopeRequiredError) as cancel_excinfo:
        service.cancel_process("process-controlled", access=requestor)
    assert cancel_excinfo.value.details["reason"] == "context_not_issued"

    with pytest.raises(ScopeRequiredError) as status_excinfo:
        service.process_status("process-controlled", access=requestor)
    assert status_excinfo.value.details["reason"] == "context_not_issued"
