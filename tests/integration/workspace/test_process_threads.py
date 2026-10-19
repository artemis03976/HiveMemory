"""真实进程表、线程表与认证授权协作：线程的派生、结束与唯一的存活判定。"""

from types import SimpleNamespace

import pytest

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import RunBinding, WorkspaceOperation
from hivememory.core.errors import AdmissionDeniedError, ScopeRequiredError, ThreadEndedError
from hivememory.core.models import AgentProfile
from hivememory.workspace.process.events import TaskProcessEventEmitter
from hivememory.workspace.process.table import ProcessPhase, ProcessRecord, ProcessTable
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess
from hivememory.workspace.process.threads import ProcessThreads, ThreadState
from hivememory.workspace.thread_ports import ResolvedThread
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_execution_labels,
)


def profile_of(agent_id):
    """被调用方图纸只需身份：派生以它的 agent_id 做 Workspace 准入。"""
    return AgentProfile(agent_id=agent_id)


async def make_process(access, table, process_id="p"):
    """已认证并登记到进程表、处于 Actor 阶段且已冻结派发话题的任务进程。"""
    context = await access.authenticate(binding=RunBinding.for_task_process(process_id))
    events = TaskProcessEventEmitter(RuntimeEventPublisher(NullRuntimeEventSink()))
    process = TaskProcess(
        record=ProcessRecord(
            process_id=process_id,
            access=context,
            events=events.for_process(
                process_id=process_id, labels=make_execution_labels(), trace_id="t"
            ),
        ),
        request=ProcessRequest(
            session_id="test-session", message="派生", target_workspace=access.default_workspace
        ),
        trace_id="t",
    )
    process.working_set.dispatch_topic_id = f"topic-{process_id}"
    table.register(process)
    process.record.enter_phase(ProcessPhase.ACTOR)
    return process


async def make_process_threads():
    """用户级准入允许 READ，被调用方的精确登记只允许 WRITE；主线程已派生。

    返回带 ``access``、``table``、``process``、``threads``、``main`` 属性的命名空间。
    """
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
    table = ProcessTable()
    process = await make_process(access, table)
    threads = ProcessThreads(table, lifecycle=access.gateway, authorizer=access.authorizer)
    main = threads.derive_main(process)
    return SimpleNamespace(access=access, table=table, process=process, threads=threads, main=main)


def registered(process):
    """线程表的可观察登记：按派生顺序的（标识, 父线程, 状态）。"""
    return [(t.thread_id, t.parent_id, t.state) for t in process.threads.snapshot()]


@pytest.mark.asyncio
async def test_derivation_registers_threads_in_order_with_callee_snapshots():
    """主线程与子线程依次登记；子线程标识按派生顺序分配，快照按被调用方的精确登记读取。"""
    chain = await make_process_threads()
    process, threads, main = chain.process, chain.threads, chain.main
    first = threads.derive_child(parent=main, profile=profile_of("worker"))
    second = threads.derive_child(parent=main, profile=profile_of("other"))

    assert first.agent_profile.agent_id == "worker"
    assert first.allowed_operations == frozenset({WorkspaceOperation.MEMORY_WRITE})
    assert second.allowed_operations == frozenset({WorkspaceOperation.MEMORY_READ})
    assert registered(process) == [
        ("main", None, ThreadState.RUNNING),
        ("child-1", "main", ThreadState.RUNNING),
        ("child-2", "main", ThreadState.RUNNING),
    ]
    assert threads.resolve(second.access) == ResolvedThread(
        process_id="p", thread_id="child-2", dispatch_topic_id="topic-p"
    )


@pytest.mark.asyncio
async def test_end_all_revokes_every_thread_idempotently_and_keeps_registrations():
    """结束撤销全部线程凭据并保留登记；重复结束是空操作，标识不复用，进程授权不受影响。"""
    chain = await make_process_threads()
    access, process, threads, main = chain.access, chain.process, chain.threads, chain.main
    child = threads.derive_child(parent=main, profile=profile_of("worker"))

    threads.end_all(process)
    threads.end_all(process)

    assert registered(process) == [
        ("main", None, ThreadState.ENDED),
        ("child-1", "main", ThreadState.ENDED),
    ]
    for context in (main, child.access):
        assert access.gateway.describe_context(context) is None
    assert process.threads.next_child_id() == "child-2"
    assert (
        access.authorizer.authorize_stage(
            process.record.access, access.default_workspace
        ).actor_identity.agent_id
        == "test_agent"
    )


@pytest.mark.asyncio
async def test_main_thread_is_derived_only_once_per_process():
    """主线程唯一：已有登记时再次派生以接线缺陷拒绝，登记不变。"""
    chain = await make_process_threads()
    before = registered(chain.process)
    with pytest.raises(ScopeRequiredError, match="已经登记"):
        chain.threads.derive_main(chain.process)
    assert registered(chain.process) == before


def after_stop_requested(chain):
    """用户的停止已被接受，凭据尚未撤销、线程登记仍为运行。"""
    chain.process.record.request_stop()
    return chain.main


def after_threads_ended(chain):
    """Actor 阶段收尾已结束全部线程。"""
    chain.threads.end_all(chain.process)
    return chain.main


def unregistered_context(chain):
    """同一进程另行签发的主线程 context：凭据有效，但不是线程表登记的那一个。"""
    return chain.access.gateway.derive_thread_context(chain.process.record.access)


def process_context(chain):
    """进程 context 不是线程 context。"""
    return chain.process.record.access


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prepare, error",
    [
        pytest.param(after_stop_requested, ThreadEndedError, id="stopped"),
        pytest.param(after_threads_ended, ThreadEndedError, id="ended"),
        pytest.param(unregistered_context, ScopeRequiredError, id="unregistered"),
        pytest.param(process_context, ScopeRequiredError, id="process-context"),
    ],
)
async def test_liveness_is_judged_once_for_resolution_and_derivation(prepare, error):
    """解析与派生共用同一个存活判定：线程已结束以生命周期错误拒绝，接线缺陷另行区分。

    停止只改变进程结局、结束撤销凭据，两者都是“线程已结束”；登记的不是这一个
    context、不是线程 context 属于接线缺陷。拒绝时线程表不变。
    """
    chain = await make_process_threads()
    caller = prepare(chain)
    before = registered(chain.process)

    with pytest.raises(error):
        chain.threads.resolve(caller)
    with pytest.raises(error):
        chain.threads.derive_child(parent=caller, profile=profile_of("worker"))
    assert registered(chain.process) == before


@pytest.mark.asyncio
async def test_resolution_ignores_phase_but_requires_frozen_dispatch_topic():
    """存活不读取阶段；进程没有派发话题时线程不可能合法存在，以接线缺陷拒绝。"""
    chain = await make_process_threads()
    chain.process.record.enter_phase(ProcessPhase.FINALIZE)
    assert chain.threads.resolve(chain.main).thread_id == "main"

    chain.process.working_set.dispatch_topic_id = None
    with pytest.raises(ScopeRequiredError, match="派发话题"):
        chain.threads.resolve(chain.main)


@pytest.mark.asyncio
async def test_only_main_thread_can_derive():
    """子线程不能再派生：拒绝且不签发、不登记。"""
    chain = await make_process_threads()
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    before = registered(chain.process)
    with pytest.raises(ScopeRequiredError, match="主线程"):
        chain.threads.derive_child(parent=child.access, profile=profile_of("other"))
    assert registered(chain.process) == before


@pytest.mark.asyncio
async def test_thread_of_another_process_cannot_derive_into_it():
    """线程只属于自己的进程：另一进程的主线程派生只登记到它自己的线程表。"""
    chain = await make_process_threads()
    other = await make_process(chain.access, chain.table, process_id="q")
    other_main = chain.threads.derive_main(other)

    child = chain.threads.derive_child(parent=other_main, profile=profile_of("worker"))

    assert chain.access.gateway.describe_context(child.access).run_id == "q"
    assert [t.thread_id for t in other.threads.snapshot()] == ["main", "child-1"]
    assert [t.thread_id for t in chain.process.threads.snapshot()] == ["main"]


class _FailingSnapshotAuthorizer:
    """真实授权者的代理：只让操作快照读取失败，模拟签发之后的失败。"""

    def __init__(self, inner):
        self._inner = inner
        self.failed_contexts = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_stage_operations(self, access, target_workspace):
        self.failed_contexts.append(access)
        raise RuntimeError("snapshot unavailable")


@pytest.mark.asyncio
async def test_failed_derivation_leaves_no_registration_or_valid_credential():
    """准入拒绝、图纸缺少身份与快照读取失败都不留下登记；签发后的失败撤销刚签发的凭据。

    失败不消耗子线程编号，下一次成功派生仍得到 ``child-1``。
    """
    chain = await make_process_threads()
    access, process, threads, main = chain.access, chain.process, chain.threads, chain.main
    before = registered(process)

    with pytest.raises(AdmissionDeniedError):
        threads.derive_child(parent=main, profile=profile_of("system"))
    with pytest.raises(ValueError, match="agent_id"):
        threads.derive_child(parent=main, profile=AgentProfile())
    failing = _FailingSnapshotAuthorizer(access.authorizer)
    failing_threads = ProcessThreads(chain.table, lifecycle=access.gateway, authorizer=failing)
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        failing_threads.derive_child(parent=main, profile=profile_of("worker"))

    assert registered(process) == before
    (issued,) = failing.failed_contexts
    assert access.gateway.describe_context(issued) is None
    threads.derive_child(parent=main, profile=profile_of("worker"))
    assert [t.thread_id for t in process.threads.snapshot()] == ["main", "child-1"]
