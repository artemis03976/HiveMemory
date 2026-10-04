"""任务进程 stop 控制契约测试。

覆盖两层契约：进程记录与进程表的 stop 语义（request_stop、登记/注销、
重复 process_id 拒绝），以及注册入口与控制面经 guard 的访问边界——注册
认证失败不创建、不登记进程；注册成功即持有签发 context；流未开始的
close_process 使 context 失效并从表中注销；跨 workspace 请求方的控制请求
统一按 not_found 呈现；context 失效的请求方以 ScopeRequiredError 拒绝。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import (
    AdmissionDeniedError,
    ScopeRequiredError,
    WorkspaceDomainError,
)
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity
from hivememory.workspace.access import AccessGrantSummary
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.process.table import (
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessStatusSnapshot,
    ProcessTable,
)
from hivememory.workspace.process.task_process import _run_interruptible
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
    return ProcessRecord(process_id=process_id, access=WorkspaceAccessContext())


def _composition() -> AccessTestComposition:
    """u1/omni_doll 全 operation 的访问组合：注册声明与请求方 context 的签发来源。"""
    return make_access_composition([make_actor_access_record(owner_user_id=_USER, agent_id=_AGENT)])


async def _service(
    bus: GlobalSystemBus | None = None,
    *,
    composition: AccessTestComposition | None = None,
    cpu: ScriptedCPU | None = None,
) -> tuple[TaskProcessService, AccessTestComposition]:
    """构造被测服务与配套认证组合：注册与控制授权使用同一 guard/gateway 实例。"""
    bus = bus or GlobalSystemBus()
    composition = composition or _composition()
    service = TaskProcessService(
        bus,
        cpu=cpu or ScriptedCPU(result=make_cpu_result()),
        access_gateway=composition.gateway,
        access_guard=composition.guard,
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
):
    """按组合的默认声明注册进程：两阶段认证由注册入口完成。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=actor or _actor(),
        workspace=workspace or _workspace(),
        process_id=process_id,
        message=message,
    )


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


def test_table_get_returns_registered_record_and_close_removes_it() -> None:
    """进程表按 process_id 原样取回记录；注销后取回为 None，不做 scope 过滤。"""
    table = ProcessTable()
    record = _record("process-1")
    table.register(record)

    assert table.get("process-1") is record
    assert table.get("missing") is None

    table.close(record)
    assert table.get("process-1") is None


def test_table_rejects_duplicate_process_id_without_overwriting() -> None:
    """防止重复 process_id 覆盖既有进程记录并把控制权转给后注册者。"""
    table = ProcessTable()
    original = _record("collision")
    table.register(original)

    with pytest.raises(WorkspaceDomainError, match="拒绝覆盖"):
        table.register(_record("collision"))

    assert table.get("collision") is original


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
async def test_register_success_registers_record_with_issued_context() -> None:
    """注册成功：进程已在表内，record.access 是签发即绑定本进程的 context。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)

    process = await _register(composition, service, process_id="process-registered")

    assert service.process_status("process-registered", access=requestor) == ProcessStatusSnapshot(
        process_id="process-registered",
        phase="created",
        status="running",
        reason=None,
    )
    # record.access 经 guard 授权返回按注册声明组装的可信 scope。
    scope = composition.guard.authorize_operation(
        process.record.access, WorkspaceOperation.RESOURCE_READ, _workspace()
    )
    assert scope == IdentityScope(actor_identity=_actor(), workspace_identity=_workspace())
    # 签发即绑定：授予记录的运行绑定是本进程的 process_id（诊断查询）。
    assert composition.guard.describe(process.record.access) == AccessGrantSummary(
        actor_user_id=_USER,
        agent_id=_AGENT,
        workspace_id=_WORKSPACE_ID,
        principal_id=composition.principal.principal_id,
        run_type="task_process",
        run_id="process-registered",
    )


@pytest.mark.asyncio
async def test_close_process_before_stream_invalidates_context_and_deregisters() -> None:
    """流从未开始：close_process 使 context 失效（guard 拒绝）并从表中注销。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    process = await _register(composition, service, process_id="process-never-run")

    await service.close_process(process)

    with pytest.raises(ScopeRequiredError) as excinfo:
        composition.guard.authorize_operation(
            process.record.access, WorkspaceOperation.RESOURCE_READ, _workspace()
        )
    assert excinfo.value.details["reason"] == "context_not_issued"
    cancel_result = service.cancel_process("process-never-run", access=requestor)
    assert cancel_result.cancelled is False
    assert cancel_result.status == "not_found"
    assert service.process_status("process-never-run", access=requestor) is None


@pytest.mark.asyncio
async def test_close_process_is_idempotent() -> None:
    """close_process 幂等：重复调用是空操作，不改变收口后的可见状态。"""
    service, composition = await _service()
    requestor = await composition.authenticate(agent_id=_AGENT)
    process = await _register(composition, service, process_id="process-close-twice")

    await service.close_process(process)
    await service.close_process(process)

    assert service.process_status("process-close-twice", access=requestor) is None


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
    process = await _register(composition, service, process_id="process-shared")
    foreign_requestor = await composition.authenticate(
        agent_id="other_agent",
        user_id="u2",
        workspace=_workspace(owner_user_id="u2", workspace_id="other_workspace"),
    )

    rejected = service.cancel_process("process-shared", access=foreign_requestor)
    assert rejected.cancelled is False
    assert rejected.status == "not_found"
    assert rejected.process_id == "process-shared"
    assert process.record.outcome is ProcessOutcome.RUNNING
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
    composition.guard.invalidate(requestor)

    with pytest.raises(ScopeRequiredError) as cancel_excinfo:
        service.cancel_process("process-controlled", access=requestor)
    assert cancel_excinfo.value.details["reason"] == "context_not_issued"

    with pytest.raises(ScopeRequiredError) as status_excinfo:
        service.process_status("process-controlled", access=requestor)
    assert status_excinfo.value.details["reason"] == "context_not_issued"
