"""真实进程表、线程表与认证授权协作：线程的派生、结束、唯一的存活判定、派发话题与 CPU 边界上的线程寿命。"""

import asyncio
from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.access import WorkingActor, WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AdmissionDeniedError, ScopeRequiredError, ThreadEndedError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    ExecutionLabels,
    ResolvedAgentProfile,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.errors import TopicBusyError
from hivememory.patchouli.contracts.interaction_submission import InteractionSubmitResult
from hivememory.workspace.capability.root.topic import RootTopicService
from hivememory.workspace.contracts import ResolveReferencesRequest, ThreadHandle
from hivememory.workspace.process.events import TaskProcessEventEmitter
from hivememory.workspace.process.table import ProcessPhase, ProcessRecord, ProcessTable
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess
from hivememory.workspace.process.threads import ProcessThreads, ThreadState
from hivememory.workspace.thread_ports import ResolvedThread
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import (
    PatchouliTurnStub,
    collect_events,
    done_data,
    make_task_process_service,
)
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_execution_labels,
    make_workspace_read_view,
)

#: Gateway 为这一轮给出的新话题决定：派发话题要等第一次写入时建立。
NEW_TOPIC_DECISION = make_gateway_decision(
    target_topic_id="NEW_TOPIC", new_topic_title="新话题", new_topic_summary="新话题摘要"
)


def profile_of(agent_id):
    """被调用方图纸只需身份：派生以它的 agent_id 做 Workspace 准入。"""
    return AgentProfile(agent_id=agent_id)


class TopicPreparation:
    """Patchouli 话题准备公开路由的替身：记录每次请求，可挂起，可按序失败。

    ``release`` 清除时请求停在路由内，``entered`` 在请求到达时置位；``failures``
    中的异常按序抛出，用完后以 ``topic_id`` 应答。
    """

    def __init__(self):
        self.topic_id = "topic-created"
        self.calls = []
        self.failures = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def prepare(self, *, identity_scope, target_topic_id, **kwargs):
        self.calls.append(
            {"identity_scope": identity_scope, "target_topic_id": target_topic_id, **kwargs}
        )
        self.entered.set()
        await self.release.wait()
        if self.failures:
            raise self.failures.pop(0)
        return self.topic_id


async def make_process(access, table, process_id="p"):
    """以认证得到的执行者与系统 root 工作 actor 登记一个处于 Actor 阶段的进程。

    Gateway 为这一轮给出新话题决定，派发话题尚未确定。
    """
    events = TaskProcessEventEmitter(RuntimeEventPublisher(NullRuntimeEventSink()))
    labels = make_execution_labels()
    process = TaskProcess(
        record=ProcessRecord(
            process_id=process_id,
            actor=await access.authenticate(),
            system_actor=await access.authenticate_system(),
            labels=labels,
            events=events.for_process(process_id=process_id, labels=labels, trace_id="t"),
        ),
        request=ProcessRequest(
            session_id="test-session", message="派生", target_workspace=access.default_workspace
        ),
        trace_id="t",
    )
    process.working_set.gateway_decision = NEW_TOPIC_DECISION
    table.register(process)
    process.record.enter_phase(ProcessPhase.ACTOR)
    return process


async def make_process_threads():
    """用户级准入允许 READ，被调用方的精确登记只允许 WRITE，``blocked`` 的登记已禁用；
    主线程已派生，话题准备经真实 root 话题 API 到达替身路由。

    返回带 ``access``、``table``、``process``、``threads``、``topics``、``main`` 属性的
    命名空间，``main`` 是交给 CPU 驱动的主线程句柄，``topics`` 是话题准备替身。
    """
    access = make_access_composition(
        [
            make_actor_access_record(
                agent_id=None, allowed_operations=[WorkspaceOperation.MEMORY_READ]
            ),
            make_actor_access_record(
                agent_id="worker", allowed_operations=[WorkspaceOperation.MEMORY_WRITE]
            ),
            make_actor_access_record(agent_id="blocked", enabled=False, allowed_operations=[]),
            make_actor_access_record(agent_id=SYSTEM_AGENT_ID),
        ]
    )
    bus = GlobalSystemBus()
    topics = TopicPreparation()
    bus.register(PatchouliRoutes.PREPARE_TOPIC, topics.prepare)
    table = ProcessTable()
    process = await make_process(access, table)
    threads = ProcessThreads(
        table,
        admission=access.gateway,
        authorizer=access.authorizer,
        topics=RootTopicService(bus, operation_authorizer=access.authorizer),
    )
    main = threads.derive_main(process, profile_of("test_agent")).handle
    return SimpleNamespace(
        access=access, table=table, process=process, threads=threads, topics=topics, main=main
    )


def registered(process):
    """线程表的可观察登记：按派生顺序的（标识, 父线程, 状态）。"""
    return [(t.thread_id, t.parent_id, t.state) for t in process.threads.snapshot()]


def working_actor(access, agent_id):
    """与执行者同一调用来源、驻留同一 Workspace 的工作 actor。"""
    return WorkingActor(
        principal=access.principal,
        actor=ActorIdentity(user_id=access.default_workspace.owner_user_id, agent_id=agent_id),
        workspace=access.default_workspace,
    )


@pytest.mark.asyncio
async def test_derivation_registers_threads_in_order_with_their_own_descriptors():
    """主线程与子线程依次登记，各自的线程描述与工作 actor 取自己的身份。

    子线程标识按派生顺序分配，快照按被调用方的精确登记读取；主线程标签是注册
    标签，子线程标签是被调用方与注册时的 workspace。解析得到的工作 actor 是
    线程自己的身份：主线程为执行者，子线程为被调用方。
    """
    chain = await make_process_threads()
    process, threads, main = chain.process, chain.threads, chain.main
    first = threads.derive_child(parent=main, profile=profile_of("worker"))
    second = threads.derive_child(parent=main, profile=profile_of("other"))

    main_descriptor = process.threads.get("main").descriptor
    assert main_descriptor.labels == process.record.labels
    assert main_descriptor.agent_profile == profile_of("test_agent")
    assert main_descriptor.allowed_operations == frozenset({WorkspaceOperation.MEMORY_READ})
    assert first.descriptor.thread_id == "child-1"
    assert first.descriptor.agent_profile.agent_id == "worker"
    assert first.descriptor.labels == ExecutionLabels(
        agent_id="worker", workspace_id=main_descriptor.labels.workspace_id
    )
    assert first.descriptor.allowed_operations == frozenset({WorkspaceOperation.MEMORY_WRITE})
    assert second.descriptor.allowed_operations == frozenset({WorkspaceOperation.MEMORY_READ})
    assert registered(process) == [
        ("main", None, ThreadState.RUNNING),
        ("child-1", "main", ThreadState.RUNNING),
        ("child-2", "main", ThreadState.RUNNING),
    ]
    assert threads.resolve(main).actor == process.record.actor
    assert threads.resolve(second.handle) == ResolvedThread(
        process_id="p", thread_id="child-2", actor=working_actor(chain.access, "other")
    )


@pytest.mark.asyncio
async def test_child_derivation_after_gateway_closes_keeps_main_principal_and_workspace():
    """网关关闭只拒绝新认证：已有进程仍能为子线程准入，子工作 actor 只换 Agent。"""
    chain = await make_process_threads()
    chain.access.gateway.close()

    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))

    resolved = chain.threads.resolve(child.handle)
    assert resolved.actor == working_actor(chain.access, "worker")
    assert resolved.actor.principal == chain.process.record.actor.principal
    assert resolved.actor.workspace == chain.process.record.actor.workspace


@pytest.mark.asyncio
async def test_end_all_ends_every_thread_idempotently_and_keeps_registrations():
    """结束全部线程并保留登记；重复结束是空操作，标识不复用，已结束的线程不再解析。"""
    chain = await make_process_threads()
    process, threads, main = chain.process, chain.threads, chain.main
    child = threads.derive_child(parent=main, profile=profile_of("worker"))

    threads.end_all(process)
    threads.end_all(process)

    assert registered(process) == [
        ("main", None, ThreadState.ENDED),
        ("child-1", "main", ThreadState.ENDED),
    ]
    for handle in (main, child.handle):
        with pytest.raises(ThreadEndedError):
            threads.resolve(handle)
    assert process.threads.next_child_id() == "child-2"


@pytest.mark.asyncio
async def test_main_thread_is_derived_only_once_per_process():
    """主线程唯一：已有登记时再次派生以接线缺陷拒绝，登记不变。"""
    chain = await make_process_threads()
    before = registered(chain.process)
    with pytest.raises(ScopeRequiredError, match="已经登记"):
        chain.threads.derive_main(chain.process, profile_of("test_agent"))
    assert registered(chain.process) == before


def after_stop_requested(chain):
    """用户的停止已被接受，线程登记仍为运行。"""
    chain.process.record.request_stop()
    return chain.main


def after_threads_ended(chain):
    """Actor 阶段收尾已结束全部线程。"""
    chain.threads.end_all(chain.process)
    return chain.main


def after_process_closed(chain):
    """进程已从进程表注销，线程登记仍为运行。"""
    chain.table.close(chain.process)
    return chain.main


def foreign_handle(chain):
    """按主线程的标识另造的句柄：标识相同，但不是线程表登记的那一个。"""
    return ThreadHandle(process_id=chain.main.process_id, thread_id=chain.main.thread_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prepare, error",
    [
        pytest.param(after_stop_requested, ThreadEndedError, id="stopped"),
        pytest.param(after_threads_ended, ThreadEndedError, id="ended"),
        pytest.param(after_process_closed, ThreadEndedError, id="process-closed"),
        pytest.param(foreign_handle, ScopeRequiredError, id="foreign-handle"),
    ],
)
async def test_liveness_is_judged_once_for_resolution_derivation_and_dispatch_topic(prepare, error):
    """解析、派生与取得派发话题共用同一个存活判定：线程未结束且进程仍在运行，接线缺陷另行区分。

    停止只改变进程结局、结束只标记线程、注销使进程离开进程表，三者都是“线程
    已结束”；登记的不是这一个句柄属于接线缺陷。拒绝时线程表不变，也不为这一轮
    确定话题。
    """
    chain = await make_process_threads()
    caller = prepare(chain)
    before = registered(chain.process)

    with pytest.raises(error):
        chain.threads.resolve(caller)
    with pytest.raises(error):
        chain.threads.derive_child(parent=caller, profile=profile_of("worker"))
    with pytest.raises(error):
        await chain.threads.dispatch_topic(caller)
    assert registered(chain.process) == before
    assert chain.topics.calls == []


@pytest.mark.asyncio
async def test_resolution_ignores_phase_and_does_not_require_dispatch_topic():
    """存活不读取阶段，也不要求派发话题：主线程从 PREPARE 起即可解析，解析不为这一轮确定话题。"""
    chain = await make_process_threads()
    for phase in (ProcessPhase.PREPARE, ProcessPhase.FINALIZE):
        chain.process.record.enter_phase(phase)
        assert chain.threads.resolve(chain.main).thread_id == "main"
    assert chain.topics.calls == []


@pytest.mark.asyncio
async def test_concurrent_dispatch_topic_requests_determine_the_topic_once_as_system():
    """主线程与子线程同时取得派发话题：这一轮只确定一次，二者得到同一个话题。

    话题确定是系统级操作，以进程记录的系统 root 工作 actor 执行，按 Gateway 的
    新话题决定以其标题与摘要建立；确定之后派发话题冻结，此后的请求直接返回。
    """
    chain = await make_process_threads()
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    chain.topics.release.clear()

    waiters = [
        asyncio.create_task(chain.threads.dispatch_topic(handle))
        for handle in (chain.main, child.handle)
    ]
    await asyncio.wait_for(chain.topics.entered.wait(), timeout=1)
    chain.topics.release.set()

    assert await asyncio.gather(*waiters) == ["topic-created", "topic-created"]
    assert await chain.threads.dispatch_topic(child.handle) == "topic-created"
    (call,) = chain.topics.calls
    system = ActorIdentity(
        user_id=chain.access.default_workspace.owner_user_id, agent_id=SYSTEM_AGENT_ID
    )
    assert call["identity_scope"].actor_identity == system
    assert call["identity_scope"].workspace_identity == chain.access.default_workspace
    assert (call["target_topic_id"], call["new_topic_title"], call["new_topic_summary"]) == (
        "NEW_TOPIC",
        "新话题",
        "新话题摘要",
    )


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_topic_determination_for_other_threads():
    """一个写入在等待话题确定时被取消：确定照常完成并冻结，另一线程的写入得到话题。

    被取消的等待方仍等确定结束才离开，不留下无人持有的确定任务。
    """
    chain = await make_process_threads()
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    chain.topics.release.clear()
    cancelled = asyncio.create_task(chain.threads.dispatch_topic(chain.main))
    waiting = asyncio.create_task(chain.threads.dispatch_topic(child.handle))
    await asyncio.wait_for(chain.topics.entered.wait(), timeout=1)

    cancelled.cancel()
    chain.topics.release.set()

    assert await asyncio.wait_for(waiting, timeout=1) == "topic-created"
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert await chain.threads.dispatch_topic(chain.main) == "topic-created"
    assert len(chain.topics.calls) == 1


@pytest.mark.asyncio
async def test_failed_topic_determination_reaches_its_waiter_and_next_request_retries():
    """话题确定失败时 Patchouli 的错误原样交回这次请求，派发话题不冻结；下一次请求重新确定。"""
    chain = await make_process_threads()
    chain.topics.failures.append(TopicBusyError("话题被占用"))

    with pytest.raises(TopicBusyError):
        await chain.threads.dispatch_topic(chain.main)
    assert await chain.threads.dispatch_topic(chain.main) == "topic-created"
    assert len(chain.topics.calls) == 2


@pytest.mark.asyncio
async def test_only_main_thread_can_derive():
    """子线程不能再派生：拒绝且不登记。"""
    chain = await make_process_threads()
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    before = registered(chain.process)
    with pytest.raises(ScopeRequiredError, match="主线程"):
        chain.threads.derive_child(parent=child.handle, profile=profile_of("other"))
    assert registered(chain.process) == before


@pytest.mark.asyncio
async def test_thread_of_another_process_cannot_derive_into_it():
    """线程只属于自己的进程：另一进程的主线程派生只登记到它自己的线程表。"""
    chain = await make_process_threads()
    other = await make_process(chain.access, chain.table, process_id="q")
    other_main = chain.threads.derive_main(other, profile_of("test_agent")).handle

    child = chain.threads.derive_child(parent=other_main, profile=profile_of("worker"))

    assert chain.threads.resolve(child.handle).process_id == "q"
    assert [t.thread_id for t in other.threads.snapshot()] == ["main", "child-1"]
    assert [t.thread_id for t in chain.process.threads.snapshot()] == ["main"]


class _FailingSnapshotAuthorizer:
    """真实授权者的代理：只让操作快照读取失败，模拟准入之后、登记之前的失败。"""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def allowed_operations(self, actor):
        raise RuntimeError("snapshot unavailable")


def derive_as(agent_id):
    """以给定被调用方派生：未准入与保留 system 都由准入拒绝。"""

    def derive(chain):
        chain.threads.derive_child(parent=chain.main, profile=profile_of(agent_id))

    return derive


def derive_without_agent_id(chain):
    """图纸缺少身份。"""
    chain.threads.derive_child(parent=chain.main, profile=AgentProfile())


def derive_with_failing_snapshot(chain):
    """被调用方准入通过，读取操作快照失败。"""
    failing = ProcessThreads(
        chain.table,
        admission=chain.access.gateway,
        authorizer=_FailingSnapshotAuthorizer(chain.access.authorizer),
        topics=RootTopicService(GlobalSystemBus(), operation_authorizer=chain.access.authorizer),
    )
    failing.derive_child(parent=chain.main, profile=profile_of("worker"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "derive, error, reason",
    [
        pytest.param(
            derive_as("system"), AdmissionDeniedError, "system_actor_not_allowed", id="system"
        ),
        pytest.param(
            derive_as("blocked"), AdmissionDeniedError, "actor_not_admitted", id="unadmitted"
        ),
        pytest.param(derive_without_agent_id, ValueError, None, id="no-agent-id"),
        pytest.param(derive_with_failing_snapshot, RuntimeError, None, id="snapshot-failure"),
    ],
)
async def test_failed_derivation_leaves_no_registration(derive, error, reason):
    """准入拒绝、图纸缺少身份与快照读取失败都不留下登记，也不消耗子线程编号。

    被调用方准入失败时明确拒绝，不退回调用方身份；下一次成功派生仍得到
    ``child-1``。
    """
    chain = await make_process_threads()
    before = registered(chain.process)

    with pytest.raises(error) as raised:
        derive(chain)

    if reason is not None:
        assert raised.value.details["reason"] == reason
    assert registered(chain.process) == before
    assert chain.process.threads.next_child_id() == "child-1"
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    assert child.descriptor.thread_id == "child-1"


@pytest.mark.asyncio
async def test_end_child_ends_only_that_child_idempotently():
    """CALL 返回时结束子线程：记为结束，之后解析与重复结束分别拒绝与空操作。

    主线程与其他子线程不受影响；标识不复用。
    """
    chain = await make_process_threads()
    first = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    second = chain.threads.derive_child(parent=chain.main, profile=profile_of("other"))

    chain.threads.end_child(first.handle)
    chain.threads.end_child(first.handle)

    assert registered(chain.process) == [
        ("main", None, ThreadState.RUNNING),
        ("child-1", "main", ThreadState.ENDED),
        ("child-2", "main", ThreadState.RUNNING),
    ]
    with pytest.raises(ThreadEndedError):
        chain.threads.resolve(first.handle)
    assert chain.threads.resolve(second.handle).thread_id == "child-2"
    third = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    assert third.descriptor.thread_id == "child-3"


@pytest.mark.asyncio
async def test_end_child_after_stop_still_ends_child():
    """停止被接受后线程已视为结束，CPU 的结束照常记为结束，不按存活拒绝。"""
    chain = await make_process_threads()
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    chain.process.record.request_stop()

    chain.threads.end_child(child.handle)

    assert chain.process.threads.get("child-1").state is ThreadState.ENDED


@pytest.mark.asyncio
async def test_end_child_after_process_closed_is_a_no_op():
    """进程注销后线程随之结束：CPU 迟到的结束是空操作，不按接线缺陷拒绝。"""
    chain = await make_process_threads()
    child = chain.threads.derive_child(parent=chain.main, profile=profile_of("worker"))
    chain.table.close(chain.process)

    chain.threads.end_child(child.handle)

    with pytest.raises(ThreadEndedError):
        chain.threads.resolve(child.handle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        pytest.param(lambda chain: chain.main, id="main-thread"),
        pytest.param(foreign_handle, id="foreign-handle"),
    ],
)
async def test_end_child_rejects_main_thread_and_foreign_handle(target):
    """CPU 不能结束主线程，也不能以另造的句柄结束线程：都属于接线缺陷，登记不变。"""
    chain = await make_process_threads()
    before = registered(chain.process)
    with pytest.raises(ScopeRequiredError):
        chain.threads.end_child(target(chain))
    assert registered(chain.process) == before
    assert chain.threads.resolve(chain.main).thread_id == "main"


# ---------------------------------------------------------------------------
# CPU 边界上的线程寿命：真实注册入口、执行器与操作入口
# ---------------------------------------------------------------------------

_EXECUTOR = ActorIdentity(user_id="u1", agent_id="omni_doll")
_SYSTEM = ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID)


@pytest.fixture
def process_chain():
    """仅替换 Gateway 与 Patchouli 的公开路由，认证、授权、执行器和入口均为真实对象。

    交互提交的替身在提交时核对：CPU 已经结束、提交以系统的 root scope 到达、
    CPU 持有的主线程与子线程句柄都已不可用，再记入 ``submitted``。
    """
    access = make_access_composition(
        [
            make_actor_access_record(owner_user_id="u1", agent_id=None),
            make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
        ]
    )
    bus = GlobalSystemBus()
    runtime = make_workspace_read_view(bus)
    table = ProcessTable()
    threads = ProcessThreads(
        table,
        admission=access.gateway,
        authorizer=access.authorizer,
        topics=RootTopicService(bus, operation_authorizer=access.authorizer),
    )
    chain = SimpleNamespace(
        access=access,
        bus=bus,
        runtime=runtime,
        submitted=[],
        table=table,
        threads=threads,
        children=[],
    )

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(*_args, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def submit(*, identity_scope, interaction_id, requested_topic_id, **_kwargs):
        assert chain.cpu.finished
        assert identity_scope.actor_identity == _SYSTEM
        assert identity_scope.workspace_identity == access.default_workspace
        for handle in (chain.cpu.calls[0].thread.handle, *chain.children):
            with pytest.raises(ThreadEndedError):
                await chain.cpu.operation_entry.execute(
                    ResolveReferencesRequest(("missing",)), thread=handle
                )
        chain.submitted.append(interaction_id)
        return InteractionSubmitResult(
            interaction_id=interaction_id, state="succeeded", topic_id=requested_topic_id
        )

    PatchouliTurnStub().install(bus)
    bus.unregister(PatchouliRoutes.INTERACTION_SUBMIT)
    for route, handler in (
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.INTERACTION_SUBMIT, submit),
    ):
        bus.register(route, handler)
    yield chain
    runtime.close()
    access.gateway.close()


async def register(chain, cpu):
    """按生产装配把真实授权者、准入端口和共享进程表注入任务进程。"""
    chain.cpu = cpu
    chain.service = make_task_process_service(
        chain.bus,
        cpu=cpu,
        access_gateway=chain.access.gateway,
        operation_authorizer=chain.access.authorizer,
        workspace_read_view=chain.runtime,
        process_table=chain.table,
    )
    return await chain.service.register_process(
        adapter="local",
        principal=chain.access.principal,
        actor=_EXECUTOR,
        workspace=chain.access.default_workspace,
        process_id="thread-lifetime",
        message="测试线程寿命",
    )


class DerivingCPU(ScriptedCPU):
    """CPU 运行期间派生两个子线程，覆盖所有终态的统一结束。"""

    def __init__(self, chain, **kwargs):
        super().__init__(**kwargs)
        self.chain = chain

    async def execute(self, manifest, *, thread, **kwargs):
        for agent_id in ("worker", "reviewer"):
            child = self.chain.threads.derive_child(
                parent=thread.handle, profile=AgentProfile(agent_id=agent_id)
            )
            self.chain.children.append(child.handle)
        return await super().execute(manifest, thread=thread, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cpu_outcome, expected_submitted",
    [
        pytest.param(
            {"result": make_cpu_result(status="completed")}, ["thread-lifetime"], id="completed"
        ),
        pytest.param({"error": RuntimeError("CPU failed")}, [], id="exception"),
    ],
)
async def test_cpu_end_ends_threads_before_terminal_and_submission_runs_as_system(
    process_chain, cpu_outcome, expected_submitted
):
    """有终态与无终态两条路径都先于交付结束线程；有输出时交互提交仍由系统执行。

    三种终态共用取得结果后的统一结束，异常与缺失终态共用无结果时的提前结束；
    各取一个代表。CPU 运行期间主线程解析为执行者在本进程的主线程。
    """
    chain = process_chain
    observed = []

    async def inspect_thread(_submit):
        resolved = chain.threads.resolve(chain.cpu.calls[0].thread.handle)
        observed.append((resolved.process_id, resolved.thread_id, resolved.actor.actor))

    cpu = DerivingCPU(chain, **cpu_outcome, operation_script=inspect_thread)
    handle = await register(chain, cpu)
    stream = chain.service.run_process(handle)
    try:
        async with asyncio.timeout(2):
            while (await anext(stream))["event"] not in {"done", "error", "cancelled"}:
                pass
        with pytest.raises(ThreadEndedError):
            chain.threads.resolve(cpu.calls[0].thread.handle)
        # 终态交付后继续耗尽，确保注册入口的关闭也执行完成。
        assert [event async for event in stream] == []
    finally:
        await stream.aclose()

    assert observed == [("thread-lifetime", "main", _EXECUTOR)]
    assert chain.submitted == expected_submitted


@pytest.mark.asyncio
async def test_user_stop_ends_threads_before_cpu_finishes_its_cancellation(process_chain):
    """用户停止使线程立即视为结束：CPU 收尾前持有的句柄已被拒绝，没有输出的一轮不提交。"""
    chain = process_chain
    cpu = DerivingCPU(chain, hang_before_result=True)
    handle = await register(chain, cpu)
    task = asyncio.create_task(collect_events(chain.service, handle))
    try:
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
        requester = await chain.access.authenticate(user_id="u1", agent_id=SYSTEM_AGENT_ID)
        assert chain.service.cancel_process("thread-lifetime", actor=requester).cancelled
        for thread in (cpu.calls[0].thread.handle, *chain.children):
            with pytest.raises(ThreadEndedError):
                await cpu.operation_entry.execute(
                    ResolveReferencesRequest(("missing",)), thread=thread
                )
        events = await asyncio.wait_for(task, timeout=2)

        assert done_data(events)["status"] == "cancelled"
        assert chain.submitted == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
