"""任务进程 stop 控制契约测试。

覆盖两层契约：进程记录的 stop 语义（request_stop）与进程表作为唯一进程
注册表的登记/注销（以 process_id 为键登记任务进程、重复 process_id 拒绝、
注销只移除同一对象），以及注册入口句柄 API 的访问边界——执行者或系统的
root 工作 actor 任一认证失败都不创建、不登记进程；注册成功的进程记录持有
两个工作 actor，返回只暴露 process_id 的句柄；句柄按对象身份判定有效，重新
构造的句柄与进程关闭后的旧句柄不能运行或关闭进程；流未开始的 close_process
从表中注销；认证成功后登记失败不留半注册记录。取消以 process_id 加请求方
的工作 actor 发起，属于 root，事件使用注册时绑定的观测标签；驻留在其他
workspace 或不是 system 的请求方统一按 not_found 呈现。

另覆盖进程的执行所有权：输出通道的容量与离开语义、订阅者暂停时进程仍自行
推进，以及五种结局共用的唯一退出路径——交付方离开只退订，不等待进程关闭。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink, RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.access import CallerPrincipal, WorkingActor
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import (
    AdmissionDeniedError,
    ThreadEndedError,
    WorkspaceDomainError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AttachmentSelectionRequest,
    ExecutionLabels,
    ResolvedAgentProfile,
    WorkspaceIdentity,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.access import SYSTEM_PRINCIPAL_ID
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.contracts import ResolveReferencesRequest
from hivememory.workspace.contracts.errors import SessionBusyError
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
    PatchouliTurnStub,
    collect_events,
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


def _working_actor(agent_id: str, *, principal_id: str = "local-process:test") -> WorkingActor:
    """驻留在 u1 默认 Workspace 的工作 actor 值。"""
    return WorkingActor(
        principal=CallerPrincipal(principal_id),
        actor=_actor(agent_id=agent_id),
        workspace=_workspace(),
    )


def _record(process_id: str) -> ProcessRecord:
    """记录级测试的进程记录：记录级 stop 语义不读取工作 actor，直接以值构造。"""
    return ProcessRecord(
        process_id=process_id,
        actor=_working_actor(_AGENT),
        system_actor=_working_actor(SYSTEM_AGENT_ID, principal_id=SYSTEM_PRINCIPAL_ID),
        labels=ExecutionLabels(agent_id=_AGENT, workspace_id=_workspace().workspace_id),
        events=BoundProcessEvents(RuntimeEventPublisher(NullRuntimeEventSink())),
    )


def _composition(
    *, admit_system: bool = True, register_system_principal: bool = True
) -> AccessTestComposition:
    """u1/omni_doll 全 operation 与 u1 的 system 准入：注册声明与取消请求方的认证来源。

    两个开关分别模拟访问登记缺少 system 记录、接入登记漏掉系统自身的 principal。
    """
    records = [make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT)]
    if admit_system:
        records.append(make_actor_access_record(owner_user_id=_USER, agent_id=SYSTEM_AGENT_ID))
    return make_access_composition(records, register_system_principal=register_system_principal)


async def _system_requestor(composition: AccessTestComposition) -> WorkingActor:
    """取消请求方：与管理入口一样以 (user, system) 认证得到的工作 actor。"""
    return await composition.authenticate(agent_id=SYSTEM_AGENT_ID)


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


def _turn_bus() -> tuple[GlobalSystemBus, PatchouliTurnStub]:
    """Gateway → Profile → Patchouli 一轮的替身总线，返回记录预检索、话题与提交的替身。"""
    bus = GlobalSystemBus()

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(_agent_id, *, identity_scope, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile)
    return bus, PatchouliTurnStub().install(bus)


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


# ========== 注册入口：认证执行者与系统、登记 ==========


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
@pytest.mark.parametrize(
    ("composition_options", "actor", "reason"),
    [
        pytest.param({}, _actor(user_id="u2"), "actor_not_owner", id="executor-not-owner"),
        pytest.param(
            {"register_system_principal": False},
            _actor(),
            "unknown_principal",
            id="system-principal-unregistered",
        ),
        pytest.param(
            {"admit_system": False}, _actor(), "actor_not_admitted", id="system-unadmitted"
        ),
    ],
)
async def test_register_authentication_failure_does_not_create_or_register_process(
    composition_options: dict, actor: ActorIdentity, reason: str
) -> None:
    """执行者或系统的 root 工作 actor 任一认证失败：抛 AdmissionDeniedError，不创建、不登记。

    自定义接入登记漏掉系统自身的 principal、或访问登记缺少该用户 system 的记录时，
    chat 注册与执行者认证失败一样被拒。
    """
    table = ProcessTable()
    service, composition = await _service(
        composition=_composition(**composition_options), process_table=table
    )

    with pytest.raises(AdmissionDeniedError) as excinfo:
        await _register(composition, service, process_id="process-auth-fail", actor=actor)

    assert excinfo.value.details["reason"] == reason
    assert table.get("process-auth-fail") is None


@pytest.mark.asyncio
async def test_register_success_records_executor_and_system_working_actors() -> None:
    """注册成功：进程已登记，记录持有入口来源认证的执行者与系统自身来源认证的 system。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)

    await _register(composition, service, process_id="process-registered")

    assert process_state(table, "process-registered") == ("created", "running", None)
    record = table.get("process-registered").record
    assert record.actor == _working_actor(_AGENT)
    assert record.system_actor == _working_actor(SYSTEM_AGENT_ID, principal_id=SYSTEM_PRINCIPAL_ID)


@pytest.mark.asyncio
async def test_process_handle_exposes_only_process_and_session_ids() -> None:
    """句柄是入口 adapter 唯一引用：只暴露 process_id/session_id，不暴露记录、工作 actor 或进程容器。"""
    service, composition = await _service()
    handle = await _register(composition, service, process_id="process-opaque")

    assert handle.process_id == "process-opaque"
    assert [name for name in dir(handle) if not name.startswith("_")] == [
        "process_id",
        "session_id",
    ]
    assert not hasattr(handle, "record")
    assert not hasattr(handle, "actor")
    assert not hasattr(handle, "task")
    assert not hasattr(handle, "request")


@pytest.mark.asyncio
async def test_close_process_before_stream_deregisters() -> None:
    """流从未开始：close_process 从表中注销，之后的取消按 not_found 呈现。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    requestor = await _system_requestor(composition)
    handle = await _register(composition, service, process_id="process-never-run")

    await service.close_process(handle)

    cancel_result = service.cancel_process("process-never-run", actor=requestor)
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


# ========== 注册入口：认证成功后登记失败 ==========


@pytest.mark.asyncio
async def test_registration_failure_after_authentication_leaves_no_registration() -> None:
    """认证通过后、登记完成前失败（会话忙碌）：无半注册记录，已有进程与会话认领不受影响。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    first = await _register(composition, service, process_id="process-holder")

    with pytest.raises(SessionBusyError):
        await service.register_process(
            adapter="local",
            principal=composition.principal,
            actor=_actor(),
            workspace=_workspace(),
            process_id="process-rejected",
            message="问题",
            session_id=first.session_id,
        )

    assert table.get("process-rejected") is None
    # 第一次注册的进程与会话认领不受影响，既有进程仍可查。
    assert process_state(table, "process-holder") == ("created", "running", None)


# ========== 句柄 API：有效性、运行与关闭 ==========


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

    with pytest.raises(WorkspaceDomainError) as excinfo:
        service.run_process(handle)
    assert excinfo.value.details["reason"] == "process_handle_unknown"
    assert excinfo.value.details["process_id"] == "process-closed-handle"


@pytest.mark.asyncio
async def test_reconstructed_handle_cannot_run_or_close_the_process() -> None:
    """按 process_id 重新构造的句柄无效：不能运行或关闭进程，进程不受影响。"""
    sink = RecordingRuntimeEventSink()
    table = ProcessTable()
    service, composition = await _service(
        event_publisher=RuntimeEventPublisher(sink), process_table=table
    )
    await _register(composition, service, process_id="process-forged")
    # 入口 adapter 拿不到进程对象：只能用一个 process_id 相同的替身去构造句柄。
    forged = ProcessHandle(SimpleNamespace(record=SimpleNamespace(process_id="process-forged")))

    with pytest.raises(WorkspaceDomainError) as excinfo:
        service.run_process(forged)
    assert excinfo.value.details["reason"] == "process_handle_unknown"
    await service.close_process(forged)

    assert _cancel_requested_events(sink) == []
    assert process_state(table, "process-forged") == ("created", "running", None)


@pytest.mark.asyncio
async def test_stale_handle_cannot_control_a_reregistered_process() -> None:
    """进程关闭后同一 process_id 重新登记：旧句柄不能关闭新进程。"""
    table = ProcessTable()
    service, composition = await _service(process_table=table)
    stale = await _register(composition, service, process_id="process-reused-id")
    await service.close_process(stale)
    await _register(composition, service, process_id="process-reused-id")

    await service.close_process(stale)

    assert process_state(table, "process-reused-id") == ("created", "running", None)


@pytest.mark.asyncio
async def test_control_request_not_found_publishes_event() -> None:
    """控制请求找不到进程：返回 not_found，并以请求中的 process_id 发布取消请求事件。"""
    sink = RecordingRuntimeEventSink()
    service, composition = await _service(event_publisher=RuntimeEventPublisher(sink))
    requestor = await _system_requestor(composition)

    result = service.cancel_process("process-missing", actor=requestor)

    assert result == CancelResult(
        process_id="process-missing",
        cancelled=False,
        status="not_found",
        reason="user_requested",
    )
    events = _cancel_requested_events(sink)
    assert [(event.process_id, event.status) for event in events] == [
        ("process-missing", "not_found")
    ]


@pytest.mark.asyncio
async def test_accepted_cancel_publishes_stop_events_with_registered_labels() -> None:
    """用户停止经 stop 记录与事件发布的唯一实现：事件使用注册时绑定的观测标签。"""
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
    requestor = await _system_requestor(composition)
    handle = await _register(composition, service, process_id="process-stop-parity")
    task = asyncio.create_task(collect_events(service, handle))
    await gateway_started.wait()

    stop_result = service.cancel_process("process-stop-parity", actor=requestor)
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
    # 事件使用注册时绑定的观测标签，请求方声明不重建身份坐标。
    assert {(event.process_id, event.workspace_id, event.agent_id) for event in events} == {
        ("process-stop-parity", _WORKSPACE_ID, _AGENT)
    }


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
    await channel.send(RunCancelled(reason="client_disconnected"))
    channel.end()

    assert await channel.receive() is None


# ========== 执行所有权与唯一退出路径 ==========


@pytest.mark.asyncio
async def test_paused_subscriber_does_not_hold_back_process_within_channel_capacity() -> None:
    """订阅者读到 process_id 后暂停：进程照常推进到交互提交并关闭，交互事件缓存在通道中。"""
    bus, stub = _turn_bus()
    token = {"event": "token", "data": {"content": "字"}}
    table = ProcessTable()
    service, composition = await _service(
        bus,
        cpu=ScriptedCPU(events=[token] * OUTPUT_CHANNEL_CAPACITY, result=make_cpu_result()),
        process_table=table,
    )
    handle = await _register(composition, service, process_id="process-paused")
    stream = service.run_process(handle)

    first = await anext(stream)
    await wait_for_process_exit(table, "process-paused")

    assert len(stub.submissions) == 1
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
    requestor: WorkingActor,
) -> list[dict]:
    """以给定结局驱动进程直到进程 task 结束，返回交付方读到的事件。"""
    stream = service.run_process(handle)
    events = [await anext(stream)]
    if outcome in _STOPPING_OUTCOMES:
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
    if outcome == "user_stop":
        assert service.cancel_process(handle.process_id, actor=requestor).cancelled is True
    elif outcome == "delivery_left":
        await stream.aclose()
        await wait_for_process_exit(table, handle.process_id)
        return events
    elif outcome == "hard_cancel":
        await service.shutdown()
    # 结束标记在进程关闭之后写入：读到它时进程 task 已经完成退出路径。
    events.extend([event async for event in stream])
    return events


async def _main_thread_ended(cpu: ScriptedCPU) -> bool:
    """CPU 收到的主线程是否已结束、不能再提交 actor 操作。"""
    try:
        await cpu.operation_entry.execute(
            ResolveReferencesRequest(("missing",)), thread=cpu.calls[0].thread.handle
        )
    except ThreadEndedError:
        return True
    return False


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
    """五种结局都经唯一退出路径：线程已结束，租借、进程登记与会话认领各释放一次，
    输出通道在关闭后结束（硬取消时没有终态）。"""
    store = InMemoryWorkspaceAssetStore()
    composition = _composition()
    requestor = await _system_requestor(composition)
    asset = make_ready_text_asset(store, _workspace(), operation_id="op-exit", content="附件正文")
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    released: list[str] = []
    release = sessions.release

    def recording_release(belong_to, session_id, process_id) -> None:
        released.append(process_id)
        release(belong_to, session_id, process_id)

    sessions.release = recording_release  # type: ignore[method-assign]
    bus, _stub = _turn_bus()
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

    events = await _drive_to_exit(
        outcome, service, handle, cpu=cpu, table=table, requestor=requestor
    )

    assert events[-1]["event"] == last_event
    assert await _main_thread_ended(cpu) is True
    assert store.close_and_clear().leases_cleared == 0
    assert table.get("process-exit") is None
    assert released == ["process-exit"]
    assert sessions.get(_workspace(), handle.session_id).active_process_id is None


class _SlowStoppingCPU:
    """推送一条交互事件后挂起；被取消时在收尾处挂起，直到测试放行后传播取消。"""

    def __init__(self) -> None:
        self.cancel_entered = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def execute(self, manifest, *, thread, model_selection, events):
        await events({"event": "token", "data": {"content": "一"}})
        try:
            await asyncio.Event().wait()
        finally:
            self.cancel_entered.set()
            await self.release_cancel.wait()


@pytest.mark.asyncio
async def test_delivery_leaving_does_not_wait_for_process_close() -> None:
    """交付方离开只退订并请求停止：CPU 收尾挂起时交付立即结束，进程放行后自行关闭并注销，
    丢弃本轮、不提交交互。"""
    bus, stub = _turn_bus()
    cpu = _SlowStoppingCPU()
    table = ProcessTable()
    service, composition = await _service(bus, cpu=cpu, process_table=table)  # type: ignore[arg-type]
    handle = await _register(composition, service, process_id="process-detached")
    stream = service.run_process(handle)
    while (await anext(stream))["event"] != "token":
        pass

    await asyncio.wait_for(stream.aclose(), timeout=1)
    await asyncio.wait_for(cpu.cancel_entered.wait(), timeout=1)

    assert process_state(table, "process-detached") == (
        "actor",
        "stop_requested",
        "client_disconnected",
    )
    cpu.release_cancel.set()
    await wait_for_process_exit(table, "process-detached")
    assert table.get("process-detached") is None
    assert stub.submissions == []


# ========== 控制面：取消的 root 授权 ==========


async def _foreign_system(composition: AccessTestComposition) -> WorkingActor:
    """驻留在 u2 的其他 workspace 的 system。"""
    return await composition.authenticate(
        agent_id=SYSTEM_AGENT_ID,
        user_id="u2",
        workspace=_workspace(owner_user_id="u2", workspace_id="other_workspace"),
    )


async def _same_workspace_agent(composition: AccessTestComposition) -> WorkingActor:
    """驻留在同一 workspace 的具体 Agent：不是 system，不能执行 root 操作。"""
    return await composition.authenticate(agent_id=_AGENT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_requestor",
    [
        pytest.param(_foreign_system, id="other-workspace"),
        pytest.param(_same_workspace_agent, id="not-system"),
    ],
)
async def test_cancel_denied_by_root_authorization_is_not_found(make_requestor) -> None:
    """取消属于 root：驻留在其他 workspace 或不是 system 的请求方统一 not_found。

    不泄露进程存在性，进程照常保持运行。
    """
    composition = make_access_composition(
        [
            make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT),
            make_actor_access_record(owner_user_id=_USER, agent_id=SYSTEM_AGENT_ID),
            make_actor_access_record(
                owner_user_id="u2",
                agent_id=SYSTEM_AGENT_ID,
                workspace_id="other_workspace",
            ),
        ]
    )
    table = ProcessTable()
    service, composition = await _service(composition=composition, process_table=table)
    await _register(composition, service, process_id="process-shared")

    rejected = service.cancel_process("process-shared", actor=await make_requestor(composition))

    assert rejected.cancelled is False
    assert rejected.status == "not_found"
    assert rejected.process_id == "process-shared"
    # not_found 不等于进程不存在：进程仍登记且未被停止。
    assert process_state(table, "process-shared") == ("created", "running", None)
