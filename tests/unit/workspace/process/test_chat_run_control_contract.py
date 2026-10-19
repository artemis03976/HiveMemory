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

另覆盖进程的执行所有权：输出通道的容量与离开语义、订阅者暂停时进程仍自行
推进，以及五种结局共用的唯一退出路径——交付方离开只退订，不等待进程关闭。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink, RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.access import (
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import (
    AdmissionDeniedError,
    ScopeRequiredError,
    WorkspaceDomainError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AttachmentSelectionRequest,
    ExecutionLabels,
    IdentityScope,
    ResolvedAgentProfile,
    WorkspaceIdentity,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import AccessGrantSummary
from hivememory.workspace.process.events import BoundProcessEvents
from hivememory.workspace.process.outputs import (
    OUTPUT_CHANNEL_CAPACITY,
    ActorEvent,
    ProcessOutputChannel,
    RunCancelled,
)
from hivememory.workspace.process.runner import _run_interruptible
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessTable,
)
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import (
    make_task_process_service,
    process_state,
    wait_for_process_exit,
)
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)
from tests.helpers.workspace_assets import make_ready_text_asset

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
    """记录级测试的进程记录：记录级 stop 语义不使用 access。

    context 只能由认证一侧签发；记录级测试是同步的，直接经认证一侧的
    准入签发一份绑定本进程的 context（跳过第 1 阶段，不影响 stop 语义）。
    """
    access = _composition().authenticator.admit(
        actor=_actor(),
        workspace=_workspace(),
        principal=CallerPrincipal("local-process:test"),
        binding=RunBinding.for_task_process(process_id),
    )
    return ProcessRecord(
        process_id=process_id,
        access=access,
        events=BoundProcessEvents(
            RuntimeEventPublisher(NullRuntimeEventSink()),
            labels=ExecutionLabels(agent_id=_AGENT, workspace_id=_workspace().workspace_id),
        ),
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
    process_table: ProcessTable | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与控制授权使用同一网关/授权者实例。"""
    bus = bus or GlobalSystemBus()
    composition = composition or _composition()
    service = make_task_process_service(
        bus,
        event_publisher=event_publisher,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        process_table=process_table,
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


def _bus_until_finalize(cleanup: AsyncMock | None = None) -> GlobalSystemBus:
    """Gateway → Profile → prepare → cleanup 的替身总线；finalize 由调用方登记。"""
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(_agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, identity_scope, interaction_id, **_kwargs):
        return PreparedAgentRun(
            belong_to=(identity_scope).workspace_identity,
            interaction_id=interaction_id,
            topic_id="topic-control",
            is_new_topic=False,
        )

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    bus.register(
        PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup or AsyncMock(return_value=True)
    )
    return bus


def _assert_context_invalidated(composition: AccessTestComposition, context) -> None:
    """绑定 context 已随进程关闭撤销：诊断查询返回 None，授权被拒绝。"""
    assert composition.gateway.describe_context(context) is None
    with pytest.raises(ScopeRequiredError) as excinfo:
        composition.authorizer.authorize_operation(
            context, WorkspaceOperation.MEMORY_READ, _workspace()
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


def test_stop_after_cancelled_terminal_returns_first_reason_instead_of_terminal() -> None:
    """已以取消结束的进程仍按已接受的 stop 应答首次原因，而不是 already_terminal。"""
    run = _record("process-cancelled")
    run.request_stop("first_reason")
    run.mark_cancelled()

    result = run.request_stop("second_reason")

    assert (result.accepted, result.reason) == (True, "first_reason")
    assert run.outcome is ProcessOutcome.CANCELLED


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
    """进程 task 自身被取消（系统关闭的硬取消）原样传播，不翻译为停止。"""
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
    table = ProcessTable()
    service, composition = await _service(process_table=table)

    with pytest.raises(WorkspaceDomainError) as excinfo:
        await _register(
            composition,
            service,
            process_id="process-system-actor",
            actor=_actor(agent_id=SYSTEM_AGENT_ID),
        )

    assert excinfo.value.details == {"agent_id": SYSTEM_AGENT_ID}
    assert table.get("process-system-actor") is None


@pytest.mark.asyncio
async def test_register_authentication_failure_does_not_create_or_register_process() -> None:
    """认证失败（actor 与 workspace owner 不一致）：抛 AdmissionDeniedError，不创建、不登记。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)

    with pytest.raises(AdmissionDeniedError) as excinfo:
        await _register(
            composition,
            service,
            process_id="process-auth-fail",
            actor=_actor(user_id="u2"),
        )

    assert excinfo.value.details["reason"] == "actor_not_owner"
    assert table.get("process-auth-fail") is None


@pytest.mark.asyncio
async def test_register_success_returns_handle_and_binds_issued_context_to_process() -> None:
    """注册成功：返回句柄，签发 context 绑定本进程并已登记到表内。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    issued = _capture_issued_contexts(composition)

    await _register(composition, service, process_id="process-registered")

    assert [context is not None for context in issued] == [True]
    bound_context = issued[0]
    assert process_state(table, "process-registered") == ("created", "running", None)
    # 绑定 context 经操作授权者按注册声明组装可信 scope。
    scope = composition.authorizer.authorize_operation(
        bound_context, WorkspaceOperation.MEMORY_READ, _workspace()
    )
    assert scope == IdentityScope(actor_identity=_actor(), workspace_identity=_workspace())
    # 签发即绑定：授予内容的运行绑定是本进程的 process_id（诊断查询）。
    assert composition.gateway.describe_context(bound_context) == AccessGrantSummary(
        actor_user_id=_USER,
        agent_id=_AGENT,
        workspace_id=_WORKSPACE_ID,
        principal_id=composition.principal.principal_id,
        run_type="task_process",
        run_id="process-registered",
    )


@pytest.mark.asyncio
async def test_process_handle_exposes_only_process_and_session_ids() -> None:
    """句柄是入口 adapter 唯一引用：只暴露 process_id/session_id，不暴露记录、context 或进程容器。"""
    service, composition = await _service()
    handle = await _register(composition, service, process_id="process-opaque")

    assert handle.process_id == "process-opaque"
    assert [name for name in dir(handle) if not name.startswith("_")] == [
        "process_id",
        "session_id",
    ]
    assert not hasattr(handle, "record")
    assert not hasattr(handle, "access")
    assert not hasattr(handle, "task")
    assert not hasattr(handle, "request")


@pytest.mark.asyncio
async def test_close_process_before_stream_invalidates_context_and_deregisters() -> None:
    """流从未开始：close_process 撤销 context（授权被拒）并从表中注销。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    requestor = await composition.authenticate(agent_id=_AGENT)
    issued = _capture_issued_contexts(composition)
    handle = await _register(composition, service, process_id="process-never-run")

    await service.close_process(handle)

    _assert_context_invalidated(composition, issued[0])
    cancel_result = service.cancel_process("process-never-run", access=requestor)
    assert cancel_result.cancelled is False
    assert cancel_result.status == "not_found"
    assert table.get("process-never-run") is None


@pytest.mark.asyncio
async def test_close_process_is_idempotent() -> None:
    """close_process 幂等：重复调用是空操作，不改变收口后的可见状态。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    handle = await _register(composition, service, process_id="process-close-twice")

    await service.close_process(handle)
    await service.close_process(handle)

    assert table.get("process-close-twice") is None


# ========== 注册入口：认证成功后登记失败的 context 失效 ==========


@pytest.mark.asyncio
async def test_registration_failure_after_authentication_invalidates_issued_context() -> None:
    """认证通过后、登记完成前失败（重复 process_id）：已签发 context 失效，无半注册记录。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
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
    assert process_state(table, "process-dup") == ("created", "running", None)


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
    table = ProcessTable()
    service, composition = await _service(
        event_publisher=RuntimeEventPublisher(sink), process_table=table
    )
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
    assert process_state(table, "process-forged") == ("created", "running", None)


@pytest.mark.asyncio
async def test_stale_handle_cannot_control_a_reregistered_process() -> None:
    """进程关闭后同一 process_id 重新登记：旧句柄不能取消或关闭新进程。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    stale = await _register(composition, service, process_id="process-reused-id")
    await service.close_process(stale)
    await _register(composition, service, process_id="process-reused-id")

    result = service.cancel_process(stale, reason="client_disconnected")
    await service.close_process(stale)

    assert result.status == "not_found"
    assert process_state(table, "process-reused-id") == ("created", "running", None)


@pytest.mark.asyncio
async def test_handle_cancel_after_delivery_reports_not_found_without_event() -> None:
    """交付结束已收口的进程：句柄已失效，取消按 not_found 收口且不发布事件。"""
    bus = _bus_until_finalize()
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
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

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
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
    """句柄形式的取消不经进程控制授权：已撤销的 access 无法取消，持有句柄即可取消。"""
    gateway_started = asyncio.Event()
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        gateway_started.set()
        await asyncio.Event().wait()

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    service, composition = await _service(bus)
    handle = await _register(composition, service, process_id="process-handle-stop")
    task = asyncio.create_task(service.run_process(handle, stream=False))
    await gateway_started.wait()

    # 控制请求形式需要有效的请求级 context：已撤销的凭据被拒绝。
    revoked = await composition.authenticate(agent_id=_AGENT)
    composition.gateway.invalidate_context(revoked)
    with pytest.raises(ScopeRequiredError) as excinfo:
        service.cancel_process("process-handle-stop", access=revoked)
    assert excinfo.value.details["reason"] == "context_not_issued"

    # 持有句柄即为生命周期所有者：不提交任何 access 也能取消自己的进程。
    stop_result = service.cancel_process(handle, reason="client_disconnected")
    await task

    assert stop_result.cancelled is True
    assert stop_result.reason == "client_disconnected"


# ========== 输出通道 ==========


def _token(content: str) -> ActorEvent:
    return ActorEvent({"event": "token", "data": {"content": content}})


@pytest.mark.asyncio
async def test_output_channel_blocks_interaction_event_until_subscriber_reads() -> None:
    """订阅者在线且交互事件已满容量：写入阻塞，读取一项后放行，顺序不变。"""
    channel = ProcessOutputChannel(capacity=1)
    await channel.send(_token("一"))
    blocked = asyncio.create_task(channel.send(_token("二")))
    await asyncio.sleep(0)

    assert blocked.done() is False
    assert await channel.receive() == _token("一")
    await asyncio.wait_for(blocked, timeout=1)
    assert await channel.receive() == _token("二")


@pytest.mark.asyncio
async def test_output_channel_terminal_and_end_marker_bypass_capacity() -> None:
    """交互事件占满容量时，终态产出与结束标记仍立即写入，订阅者按顺序读到。"""
    channel = ProcessOutputChannel(capacity=1)
    await channel.send(_token("一"))
    terminal = RunCancelled(reason="user_requested")

    await asyncio.wait_for(channel.send(terminal), timeout=1)
    channel.end()

    assert [await channel.receive() for _ in range(3)] == [_token("一"), terminal, None]


@pytest.mark.asyncio
async def test_output_channel_detach_releases_blocked_writer_and_drops_later_outputs() -> None:
    """订阅者离开：阻塞的写入被唤醒，缓存与此后的写入都被丢弃。"""
    channel = ProcessOutputChannel(capacity=1)
    await channel.send(_token("一"))
    blocked = asyncio.create_task(channel.send(_token("二")))
    await asyncio.sleep(0)

    channel.detach()
    await asyncio.wait_for(blocked, timeout=1)
    await channel.send(RunCancelled(reason="stream_closed"))
    channel.end()

    assert await channel.receive() is None


# ========== 执行所有权与唯一退出路径 ==========


@pytest.mark.asyncio
async def test_paused_subscriber_does_not_hold_back_process_within_channel_capacity() -> None:
    """订阅者读到 process_id 后暂停：进程照常推进到 finalize 并关闭，交互事件缓存在通道中。"""
    bus = _bus_until_finalize()
    finalize = AsyncMock(return_value=[])
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
    token = {"event": "token", "data": {"content": "字"}}
    table = ProcessTable()
    service, composition = await _service(
        bus,
        cpu=ScriptedCPU(events=[token] * OUTPUT_CHANNEL_CAPACITY, result=make_cpu_result()),
        process_table=table,
    )
    handle = await _register(composition, service, process_id="process-paused")
    stream = service.run_process(handle, stream=True)

    first = await anext(stream)
    await wait_for_process_exit(table, "process-paused")

    finalize.assert_awaited_once()
    assert table.get("process-paused") is None
    events = [first, *[event async for event in stream]]
    assert [event["event"] for event in events] == [
        "process_id",
        "memory_refs",
        *["token"] * OUTPUT_CHANNEL_CAPACITY,
        "run_status",
        "done",
    ]
    assert events[-1]["data"]["status"] == "completed"


_STOPPING_OUTCOMES = {"user_stop", "delivery_left", "hard_cancel"}


async def _drive_to_exit(
    outcome: str,
    service: TaskProcessService,
    handle: ProcessHandle,
    *,
    cpu: ScriptedCPU,
    table: ProcessTable,
) -> list[dict]:
    """以给定结局驱动进程直到进程 task 结束，返回交付方读到的事件。"""
    stream = service.run_process(handle, stream=True)
    events = [await anext(stream)]
    if outcome in _STOPPING_OUTCOMES:
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
    if outcome == "user_stop":
        assert service.cancel_process(handle).cancelled is True
    elif outcome == "delivery_left":
        await stream.aclose()
        await wait_for_process_exit(table, handle.process_id)
        return events
    elif outcome == "hard_cancel":
        await service.shutdown()
    # 结束标记在进程关闭之后写入：读到它时进程 task 已经完成退出路径。
    events.extend([event async for event in stream])
    return events


def _thread_revoked(composition: AccessTestComposition, cpu: ScriptedCPU) -> bool:
    """CPU 收到的主线程 context 是否已撤销、不能再提交 actor 操作。"""
    return composition.gateway.describe_context(cpu.calls[0].access) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "last_event"),
    [
        ("completed", "done"),
        ("user_stop", "done"),
        ("delivery_left", "process_id"),
        ("failed", "error"),
        ("hard_cancel", "token"),
    ],
)
async def test_every_outcome_exits_through_single_close_path(outcome: str, last_event: str) -> None:
    """五种结局都经唯一退出路径：线程 context 已撤销，租借、进程 context、
    进程登记与会话认领各释放一次，输出通道在关闭后结束（硬取消时没有终态）。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    requestor = await composition.authenticate(agent_id=_AGENT)
    asset_scope = composition.authorizer.authorize_operation(
        requestor, WorkspaceOperation.MEMORY_READ, _workspace()
    )
    asset = make_ready_text_asset(store, asset_scope, operation_id="op-exit", content="附件正文")
    issued = _capture_issued_contexts(composition)
    invalidated: list[WorkspaceAccessContext] = []
    invalidate = composition.gateway.invalidate_context

    def recording_invalidate(context: WorkspaceAccessContext) -> None:
        invalidated.append(context)
        invalidate(context)

    composition.gateway.invalidate_context = recording_invalidate  # type: ignore[method-assign]
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    released: list[str] = []
    release = sessions.release

    def recording_release(belong_to, session_id, process_id) -> None:
        released.append(process_id)
        release(belong_to, session_id, process_id)

    sessions.release = recording_release  # type: ignore[method-assign]
    bus = _bus_until_finalize()
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "一"}}],
        result=make_cpu_result(),
        error=RuntimeError("cpu exploded") if outcome == "failed" else None,
        hang_before_result=outcome in _STOPPING_OUTCOMES,
        linger_on_cancel=outcome == "hard_cancel",
    )
    table = ProcessTable()
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
        asset_reader=store,
        process_table=table,
        sessions=sessions,
        shutdown_wait_seconds=0,
    )
    handle = await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=_actor(),
        workspace=_workspace(),
        process_id="process-exit",
        message="问题",
        attachments=[AttachmentSelectionRequest(asset_ref=asset)],
    )

    events = await _drive_to_exit(outcome, service, handle, cpu=cpu, table=table)

    assert events[-1]["event"] == last_event
    assert _thread_revoked(composition, cpu) is True
    assert store.close_and_clear().leases_cleared == 0
    assert [context is issued[0] for context in invalidated] == [True]
    _assert_context_invalidated(composition, issued[0])
    assert table.get("process-exit") is None
    assert released == ["process-exit"]
    assert sessions.get(_workspace(), handle.session_id).active_process_id is None


class _SlowStoppingCPU:
    """推送一条交互事件后挂起；被取消时在收尾处挂起，直到测试放行后传播取消。"""

    def __init__(self) -> None:
        self.cancel_entered = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def execute(self, manifest, *, access, generation_options, events):
        await events({"event": "token", "data": {"content": "一"}})
        try:
            await asyncio.Event().wait()
        finally:
            self.cancel_entered.set()
            await self.release_cancel.wait()


@pytest.mark.asyncio
async def test_delivery_leaving_does_not_wait_for_process_close() -> None:
    """交付方离开只退订并请求停止：CPU 收尾挂起时交付立即结束，进程放行后自行 cleanup 并注销。"""
    cleanup = AsyncMock(return_value=True)
    bus = _bus_until_finalize(cleanup)
    cpu = _SlowStoppingCPU()
    table = ProcessTable()
    service, composition = await _service(bus, cpu=cpu, process_table=table)  # type: ignore[arg-type]
    handle = await _register(composition, service, process_id="process-detached")
    stream = service.run_process(handle, stream=True)
    while (await anext(stream))["event"] != "token":
        pass

    await asyncio.wait_for(stream.aclose(), timeout=1)
    await asyncio.wait_for(cpu.cancel_entered.wait(), timeout=1)

    assert process_state(table, "process-detached") == ("actor", "stop_requested", "stream_closed")
    cleanup.assert_not_awaited()
    cpu.release_cancel.set()
    await wait_for_process_exit(table, "process-detached")
    assert table.get("process-detached") is None
    assert [call.kwargs["prepared_run"].interaction_id for call in cleanup.await_args_list] == [
        "process-detached"
    ]


# ========== 控制面：请求方与进程记录的驻留坐标比对 ==========


@pytest.mark.asyncio
async def test_control_requests_from_other_workspace_are_not_found() -> None:
    """跨 workspace 请求方：取消统一 not_found，不泄露进程存在性，进程照常保持运行。"""
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
    table = ProcessTable()
    service, composition = await _service(composition=composition, process_table=table)
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
    # not_found 不等于进程不存在：进程仍登记且未被停止。
    assert process_state(table, "process-shared") == ("created", "running", None)


@pytest.mark.asyncio
async def test_control_requests_with_invalidated_requestor_context_raise_scope_required() -> None:
    """context 失效的请求方：取消以 ScopeRequiredError（context_not_issued）拒绝。"""
    service, composition = await _service()
    await _register(composition, service, process_id="process-controlled")
    requestor = await composition.authenticate(agent_id=_AGENT)
    composition.gateway.invalidate_context(requestor)

    with pytest.raises(ScopeRequiredError) as cancel_excinfo:
        service.cancel_process("process-controlled", access=requestor)
    assert cancel_excinfo.value.details["reason"] == "context_not_issued"
