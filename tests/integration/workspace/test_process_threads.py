"""真实进程表、线程管理与认证授权的派生边界。"""

import pytest

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import RunBinding, WorkspaceOperation
from hivememory.core.errors import ScopeRequiredError, ThreadContextRevokedError
from hivememory.core.models import AgentProfile
from hivememory.workspace.process.events import TaskProcessEventEmitter
from hivememory.workspace.process.table import ProcessPhase, ProcessRecord, ProcessTable
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess
from hivememory.workspace.process.threads import ProcessThreads
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_execution_labels,
)


def profile_of(agent_id):
    """被调用方图纸只需身份：派生以它的 agent_id 做 Workspace 准入。"""
    return AgentProfile(agent_id=agent_id)


async def make_process_threads():
    """用户级准入允许 READ，被调用方的精确登记只允许 WRITE。"""
    access = make_access_composition(
        [
            make_actor_access_record(
                agent_id=None, allowed_operations=[WorkspaceOperation.MEMORY_READ]
            ),
            make_actor_access_record(
                agent_id="worker", allowed_operations=[WorkspaceOperation.MEMORY_WRITE]
            ),
        ]
    )
    context = await access.authenticate(binding=RunBinding.for_task_process("p"))
    events = TaskProcessEventEmitter(RuntimeEventPublisher(NullRuntimeEventSink()))
    process = TaskProcess(
        record=ProcessRecord(
            process_id="p",
            access=context,
            events=events.for_process(process_id="p", labels=make_execution_labels(), trace_id="t"),
        ),
        request=ProcessRequest(
            session_id="test-session", message="派生", target_workspace=access.default_workspace
        ),
        trace_id="t",
    )
    table = ProcessTable()
    table.register(process)
    threads = ProcessThreads(table, lifecycle=access.gateway, authorizer=access.authorizer)
    process.record.enter_phase(ProcessPhase.ACTOR)
    main = threads.derive_main_thread(process)
    return access, process, threads, main


@pytest.mark.asyncio
async def test_child_snapshots_follow_exact_callee_record_and_all_threads_are_revoked():
    """多个子线程各自登记，精确记录覆盖用户级权限，统一撤销仍保留进程授权。"""
    access, process, threads, main = await make_process_threads()
    first = threads.derive_child_thread(access=main, process_id="p", profile=profile_of("worker"))
    second = threads.derive_child_thread(access=main, process_id="p", profile=profile_of("other"))
    assert first.agent_profile.agent_id == "worker"
    assert first.allowed_operations == frozenset({WorkspaceOperation.MEMORY_WRITE})
    assert second.allowed_operations == frozenset({WorkspaceOperation.MEMORY_READ})
    assert access.gateway.describe_context(first.access).thread_id == "child-1"
    assert access.gateway.describe_context(second.access).thread_id == "child-2"
    threads.revoke_threads(process.working_set)
    threads.revoke_threads(process.working_set)
    for context in (main, first.access, second.access):
        with pytest.raises(ThreadContextRevokedError):
            access.authorizer.validate_thread_context(context)
    assert (
        access.authorizer.authorize_stage(
            process.record.access, access.default_workspace
        ).actor_identity.agent_id
        == "test_agent"
    )


async def from_child_thread(access, process, threads, main):
    """子线程不能再派生。"""
    child = threads.derive_child_thread(access=main, process_id="p", profile=profile_of("worker"))
    return child.access, "p"


async def from_other_process(access, process, threads, main):
    """另一进程的线程不是本进程登记的主线程。"""
    return await access.authenticate_thread(process_id="other"), "p"


async def from_unregistered_process(access, process, threads, main):
    """进程关联指向未登记的进程。"""
    return main, "missing"


async def after_entering_finalize(access, process, threads, main):
    """进程已离开 Actor 阶段。"""
    process.record.enter_phase(ProcessPhase.FINALIZE)
    return main, "p"


async def after_main_revoked(access, process, threads, main):
    """主线程仍登记，但 context 已被单独撤销。"""
    access.gateway.revoke_thread_context(main)
    return main, "p"


async def after_cpu_end(access, process, threads, main):
    """CPU 执行结束，全部线程已撤销并移出登记。"""
    threads.revoke_threads(process.working_set)
    return main, "p"


async def after_stop_requested(access, process, threads, main):
    """用户已请求停止进程。"""
    process.record.request_stop()
    return main, "p"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prepare, error",
    [
        pytest.param(from_child_thread, ScopeRequiredError, id="child"),
        pytest.param(from_other_process, ScopeRequiredError, id="other-process"),
        pytest.param(from_unregistered_process, ScopeRequiredError, id="missing"),
        pytest.param(after_entering_finalize, ScopeRequiredError, id="finalize"),
        pytest.param(after_main_revoked, ThreadContextRevokedError, id="revoked"),
        pytest.param(after_cpu_end, ScopeRequiredError, id="ended"),
        pytest.param(after_stop_requested, ScopeRequiredError, id="stopped"),
    ],
)
async def test_derivation_rejects_invalid_caller_or_phase_without_registration(prepare, error):
    """递归、错进程、结束或停止后派生都拒绝，已有线程登记不变。"""
    access, process, threads, main = await make_process_threads()
    caller, process_id = await prepare(access, process, threads, main)
    before = dict(process.working_set.thread_contexts)
    with pytest.raises(error):
        threads.derive_child_thread(
            access=caller, process_id=process_id, profile=profile_of("worker")
        )
    assert process.working_set.thread_contexts == before


@pytest.mark.asyncio
async def test_derivation_rejects_profile_without_agent_identity():
    """没有身份的图纸无法确定被调用方，拒绝且不签发、不登记。"""
    access, process, threads, main = await make_process_threads()
    before = dict(process.working_set.thread_contexts)
    with pytest.raises(ValueError, match="agent_id"):
        threads.derive_child_thread(access=main, process_id="p", profile=AgentProfile())
    assert process.working_set.thread_contexts == before
