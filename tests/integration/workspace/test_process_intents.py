"""任务进程、真实能力层与意图登记的生命周期协作测试。

Gateway 与 Patchouli 的公开路由位于本边界之外，由替身提供；CPU 经真实操作
入口提交请求，验证写入在授权与基础冷读之后才为这一轮确定话题、实时派发、
终态不撤回、意图的线程归因与线程结束后的拒绝。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import OperationDeniedError, ResourceNotFoundError, ScopeRequiredError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    ResolvedAgentProfile,
)
from hivememory.core.models.pending import PendingAtomStatus, WriteFocus
from hivememory.core.protocol.gateway import GatewayDecision, GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.errors import TopicBusyError
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.contracts import (
    DeriveCallThreadRequest,
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
    ThreadEndedError,
)
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.read_view import WorkspaceReadView
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.intent_dispatch import IntentAdmissionStub, install_intent_admission
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import (
    PatchouliTurnStub,
    collect_events,
    done_data,
    error_code,
    make_task_process_service,
)
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
)

pytestmark = pytest.mark.integration

ACTOR = ActorIdentity(user_id="u1", agent_id="omni_doll")
#: Gateway 为这一轮给出的新话题决定：话题在第一次写入或交互应用时才建立。
NEW_TOPIC_DECISION = make_gateway_decision(
    target_topic_id="NEW_TOPIC", new_topic_title="新话题", new_topic_summary="新话题摘要"
)


@dataclass
class IntentChain:
    """一条进程链路：真实认证授权、读取视图与意图登记，Gateway 与 Patchouli 是替身。

    ``patchouli`` 记录话题准备与交互提交，新话题建立为 ``topic-new``；``admission``
    记录生成队列收到的意图。
    """

    access: AccessTestComposition
    bus: GlobalSystemBus
    runtime: WorkspaceReadView
    patchouli: PatchouliTurnStub
    admission: IntentAdmissionStub

    async def start(
        self, cpu, *, process_id="current", authorizer=None
    ) -> tuple[TaskProcessService, ProcessHandle]:
        """装配注册入口并以执行者注册进程；``authorizer`` 缺省为访问组合的授权者。"""
        service = make_task_process_service(
            self.bus,
            cpu=cpu,
            access_gateway=self.access.gateway,
            operation_authorizer=authorizer or self.access.authorizer,
            workspace_read_view=self.runtime,
        )
        handle = await service.register_process(
            adapter="local",
            principal=self.access.principal,
            actor=ACTOR,
            workspace=self.access.default_workspace,
            process_id=process_id,
            message="登记",
        )
        return service, handle

    async def run(self, cpu, **kwargs) -> list[dict]:
        """注册并把进程运行到关闭，返回全部流式事件。"""
        service, handle = await self.start(cpu, **kwargs)
        return await collect_events(service, handle)

    def override(self, route, handler) -> None:
        """替换一条已登记的替身路由。"""
        self.bus.unregister(route)
        self.bus.register(route, handler)


@pytest.fixture
def make_chain():
    """按 Gateway 决定与执行者的操作许可构造链路，测试结束时关闭读取视图与网关。"""
    chains: list[IntentChain] = []

    def build(
        *,
        decision: GatewayDecision = NEW_TOPIC_DECISION,
        executor_operations: Iterable[WorkspaceOperation] | None = None,
    ) -> IntentChain:
        access = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1", agent_id="omni_doll", allowed_operations=executor_operations
                ),
                make_actor_access_record(owner_user_id="u1", agent_id="worker"),
                make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
            ]
        )
        bus = GlobalSystemBus()
        admission = install_intent_admission(bus)
        patchouli = PatchouliTurnStub().install(bus)

        async def gateway(**_kwargs):
            return GatewayDecisionOutcome(decision=decision)

        async def profile(agent_alias, **_kwargs):
            if agent_alias == "worker":
                return ResolvedAgentProfile(profile=AgentProfile(agent_id="worker"))
            return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

        async def no_aliases(*_args, **_kwargs):
            return []

        bus.register(GatewayPublicRoutes.PROCESS, gateway)
        bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile)
        bus.register(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, no_aliases)
        runtime = WorkspaceReadView(
            backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
        )
        chain = IntentChain(
            access=access, bus=bus, runtime=runtime, patchouli=patchouli, admission=admission
        )
        chains.append(chain)
        return chain

    yield build
    for chain in chains:
        chain.runtime.close()
        chain.access.gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
async def test_process_terminal_status_preserves_dispatched_intents(make_chain, status):
    """完成、取消与失败都不撤回已经派发的意图。

    Gateway 指向新话题：第一次写入为这一轮建立话题，意图以建立的真实话题派发，
    交互也提交到这个话题，不沿用 Gateway 的目标标记。
    """
    chain = make_chain()
    workspace = chain.access.default_workspace
    other = chain.runtime.intents.register_write(
        WriteFocus(content="另一个进程"),
        belong_to=workspace,
        from_actor=ACTOR,
        process_id="other",
        thread_id="main",
    )
    submitted = []

    async def write(submit):
        submitted.append(await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="本进程"))))

    cpu = ScriptedCPU(result=make_cpu_result(status=status), operation_script=write)
    events = await chain.run(cpu)

    assert done_data(events)["status"] == status
    ((intent, scope),) = chain.admission.submissions
    assert (intent.topic_id, scope.actor_identity) == ("topic-new", ACTOR)
    assert [
        (submission["requested_topic_id"], submission["payload"].outcome)
        for submission in chain.patchouli.submissions
    ] == [("topic-new", status)]
    current = chain.runtime.intents.get(submitted[0].pending_alias, workspace)
    assert current.status == PendingAtomStatus.MATERIALIZING
    assert (
        chain.runtime.intents.get(other.pending_alias, workspace).status
        == PendingAtomStatus.MATERIALIZING
    )
    # completed 同样结束主线程，全部请求类型均拒绝继续执行。
    for request in (
        SubmitWriteIntentRequest(focus=WriteFocus(content="关闭后")),
        SubmitUpdateIntentRequest(base_alias="base", instruction="关闭后"),
        ResolveReferencesRequest(aliases=(other.pending_alias,)),
    ):
        with pytest.raises(ThreadEndedError):
            await cpu.operation_entry.execute(request, thread=cpu.calls[0].thread.handle)
    assert chain.runtime.intents.size == 2


@pytest.mark.asyncio
async def test_concurrent_main_and_child_writes_share_one_topic_and_record_their_own_threads(
    make_chain,
):
    """主线程与 CALL 子线程同时写入：这一轮只建立一个话题，两份意图都以它派发。

    两份意图各自记录线程标识与发起者，共用进程关联。
    """
    chain = make_chain(
        executor_operations=[WorkspaceOperation.MEMORY_WRITE, WorkspaceOperation.AGENT_CALL]
    )
    submitted = []

    async def write_from_both_threads(submit):
        callee = await submit.derive_child(DeriveCallThreadRequest("worker"))
        submitted.extend(
            await asyncio.gather(
                submit(SubmitWriteIntentRequest(focus=WriteFocus(content="主线程"))),
                callee(SubmitWriteIntentRequest(focus=WriteFocus(content="子线程"))),
            )
        )

    cpu = ScriptedCPU(result=make_cpu_result(), operation_script=write_from_both_threads)
    await chain.run(cpu, process_id="attributed")

    stored = [
        chain.runtime.intents.get(pending.pending_alias, chain.access.default_workspace)
        for pending in submitted
    ]
    assert [(item.process_id, item.thread_id, item.from_actor.agent_id) for item in stored] == [
        ("attributed", "main", "omni_doll"),
        ("attributed", "child-1", "worker"),
    ]
    assert [intent.topic_id for intent, _scope in chain.admission.submissions] == [
        "topic-new",
        "topic-new",
    ]
    assert len(chain.patchouli.prepared_topics) == 1


@pytest.mark.asyncio
async def test_searches_reads_runs_and_calls_do_not_determine_the_topic(make_chain):
    """预检索、SEARCH、READ、RUN 与 CALL 都不为这一轮确定话题。

    Gateway 指向新话题、这一轮没有写入时，话题池里没有为它建立的话题，交互按
    Gateway 的新话题决定提交。
    """
    chain = make_chain()

    async def operate(submit):
        await submit(RetrieveRequest(semantic_query="检索"))
        await submit(ResolveReferencesRequest(("missing",)))
        await submit(ResolveRunTargetRequest("missing"))
        callee = await submit.derive_child(DeriveCallThreadRequest("worker"))
        await callee(ResolveReferencesRequest(("missing",)))

    events = await chain.run(ScriptedCPU(result=make_cpu_result(), operation_script=operate))

    assert done_data(events)["status"] == "completed"
    # 预检索与 SEARCH 都到达了检索路由，操作确实执行过。
    assert len(chain.patchouli.retrievals) == 2
    assert chain.patchouli.prepared_topics == []
    assert [s["requested_topic_id"] for s in chain.patchouli.submissions] == ["NEW_TOPIC"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_, error",
    [
        pytest.param(
            SubmitWriteIntentRequest(focus=WriteFocus(content="无权写入")),
            OperationDeniedError,
            id="write-not-allowed",
        ),
        pytest.param(
            SubmitUpdateIntentRequest(base_alias="missing", instruction="改写"),
            ResourceNotFoundError,
            id="update-base-missing",
        ),
    ],
)
async def test_rejected_write_does_not_determine_the_topic(make_chain, request_, error):
    """写入先授权、UPDATE 先冷读基础原子，之后才为这一轮确定话题。

    执行者没有 ``memory.write`` 时写入在授权处被拒，UPDATE 的基础原子不存在时在
    冷读处被拒：两者都不建立话题，也不登记意图。
    """
    chain = make_chain(
        executor_operations=set(WorkspaceOperation) - {WorkspaceOperation.MEMORY_WRITE}
    )
    rejections = []

    async def submit_rejected(submit):
        try:
            await submit(request_)
        except error as rejected:
            rejections.append(rejected)

    events = await chain.run(
        ScriptedCPU(result=make_cpu_result(), operation_script=submit_rejected)
    )

    assert done_data(events)["status"] == "completed"
    assert len(rejections) == 1
    assert chain.patchouli.prepared_topics == []
    assert chain.runtime.intents.size == 0


@pytest.mark.asyncio
async def test_failed_topic_determination_fails_that_write_and_the_next_write_retries(make_chain):
    """话题确定失败时 Patchouli 的错误交回这次写入，不登记意图；下一次写入重新确定。

    重新确定成功后，后一次写入以建立的话题派发。
    """
    chain = make_chain()
    failures = [TopicBusyError("话题被占用")]
    attempts = []

    async def prepare_topic(*, target_topic_id, **_kwargs):
        attempts.append(target_topic_id)
        if failures:
            raise failures.pop()
        return "topic-new"

    chain.override(PatchouliRoutes.PREPARE_TOPIC, prepare_topic)
    outcomes = []

    async def write_twice(submit):
        for content in ("第一次", "第二次"):
            try:
                outcomes.append(
                    await submit(SubmitWriteIntentRequest(focus=WriteFocus(content=content)))
                )
            except TopicBusyError as busy:
                outcomes.append(busy)

    await chain.run(ScriptedCPU(result=make_cpu_result(), operation_script=write_twice))

    first, second = outcomes
    assert isinstance(first, TopicBusyError)
    workspace = chain.access.default_workspace
    assert chain.runtime.intents.get(second.pending_alias, workspace).focus.content == "第二次"
    assert chain.runtime.intents.size == 1
    assert [intent.topic_id for intent, _scope in chain.admission.submissions] == ["topic-new"]
    assert attempts == ["NEW_TOPIC", "NEW_TOPIC"]


@pytest.mark.asyncio
async def test_stop_during_topic_determination_rejects_the_waiting_write_before_registration(
    make_chain,
):
    """写入等待话题确定期间进程收到停止：确定照常完成，登记前的同步段拒绝这次写入。

    写入不登记意图、不派发；进程按取消收口。
    """
    chain = make_chain()
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare_topic(**_kwargs):
        entered.set()
        await release.wait()
        return "topic-new"

    chain.override(PatchouliRoutes.PREPARE_TOPIC, prepare_topic)
    writes = []

    async def write_then_hang(submit):
        # 写入任务独立于 CPU 拉取任务，停止不会取消它。
        writes.append(
            asyncio.create_task(
                submit(SubmitWriteIntentRequest(focus=WriteFocus(content="等待话题")))
            )
        )
        await entered.wait()

    cpu = ScriptedCPU(hang_before_result=True, operation_script=write_then_hang)
    service, handle = await chain.start(cpu)
    run = asyncio.create_task(collect_events(service, handle))
    try:
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
        requester = await chain.access.authenticate(agent_id=SYSTEM_AGENT_ID)
        assert service.cancel_process(handle.process_id, actor=requester).cancelled
        release.set()

        (write,) = writes
        with pytest.raises(ThreadEndedError):
            await asyncio.wait_for(write, timeout=1)
        events = await asyncio.wait_for(run, timeout=2)
        assert done_data(events)["status"] == "cancelled"
        assert chain.runtime.intents.size == 0
        assert chain.admission.submissions == []
    finally:
        release.set()
        for task in (run, *writes):
            task.cancel()
        await asyncio.gather(run, *writes, return_exceptions=True)


class _RevokeAdmissionAtSubmission:
    """Actor 登记意图后切换到已失效的真实准入登记，交互提交的 root 授权必须即时拒绝。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._revoked = make_access_composition([]).authorizer
        self.revoked = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def authorize_root(self, actor, target_workspace):
        authorizer = self._revoked if self.revoked else self._inner
        return authorizer.authorize_root(actor, target_workspace)


@pytest.mark.asyncio
async def test_submission_authorization_failure_preserves_dispatched_intents(make_chain):
    """交互提交的授权失败只影响提交，不撤回已经派发的意图。"""
    chain = make_chain()
    authorizer = _RevokeAdmissionAtSubmission(chain.access.authorizer)
    submitted = []

    async def write(submit):
        submitted.append(await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="本进程"))))
        authorizer.revoked = True

    events = await chain.run(
        ScriptedCPU(result=make_cpu_result(), operation_script=write), authorizer=authorizer
    )

    status = chain.runtime.intents.get(
        submitted[0].pending_alias, chain.access.default_workspace
    ).status
    assert error_code(events) == ScopeRequiredError.code
    assert (status, chain.patchouli.submissions) == (PendingAtomStatus.MATERIALIZING, [])


@pytest.mark.asyncio
async def test_process_close_rejects_waiting_update_without_cancelling_request_task(make_chain):
    """进程在 UPDATE 冷读期间关闭：拒绝登记、不为这一轮确定话题，调用方任务不因线程结束被取消。"""
    chain = make_chain()
    entered = asyncio.Event()
    release = asyncio.Event()
    atom = MemoryAtom(
        meta=make_memory_metadata(source_agent_id="omni_doll", user_id="u1"),
        index=IndexLayer(
            title="基础原子", summary="基础摘要", alias="base", memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="原正文"),
    )

    async def read(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return [atom]

    chain.override(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, read)
    update_tasks = []

    async def update(submit):
        # 外部适配器任务独立于 CPU 拉取任务，线程结束不得取消它。
        update_tasks.append(
            asyncio.create_task(
                submit(SubmitUpdateIntentRequest(base_alias="base", instruction="新内容"))
            )
        )
        await entered.wait()

    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "一"}}],
        hang_before_result=True,
        operation_script=update,
    )
    service, handle = await chain.start(cpu, process_id="waiting")
    stream = service.run_process(handle)
    try:
        async with asyncio.timeout(1):
            while (await anext(stream))["event"] != "token":
                pass
        await stream.aclose()
        # 交付方离开只请求停止；等待进程在自己的 task 中结束线程并关闭。
        await service.close_process(handle)
        (task,) = update_tasks
        assert task.cancelling() == 0
        assert chain.runtime.intents.size == 0
        release.set()
        with pytest.raises(ThreadEndedError):
            await asyncio.wait_for(task, timeout=1)
        assert task.cancelled() is False
        assert chain.runtime.intents.size == 0
        assert chain.patchouli.prepared_topics == []
        with pytest.raises(ThreadEndedError):
            await cpu.operation_entry.execute(
                ResolveReferencesRequest(aliases=("base",)), thread=cpu.calls[0].thread.handle
            )
    finally:
        await stream.aclose()
        for task in update_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*update_tasks, return_exceptions=True)
