"""真实进程与操作入口协作，验证 CPU 边界上的线程寿命与阶段收尾。"""

import asyncio
from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    ResolvedAgentProfile,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.contracts import ResolveReferencesRequest, ThreadEndedError
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.process.threads import ProcessThreads
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_runtime,
)


@pytest.fixture
def process_chain():
    """仅替换阶段 RPC 的资源 owner，认证、授权、执行器和入口均为真实对象。"""
    access = make_access_composition([make_actor_access_record(owner_user_id="u1", agent_id=None)])
    bus = GlobalSystemBus()
    runtime = make_workspace_runtime(bus)
    table = ProcessTable()
    threads = ProcessThreads(table, lifecycle=access.gateway, authorizer=access.authorizer)
    chain = SimpleNamespace(
        access=access,
        bus=bus,
        runtime=runtime,
        stages=[],
        table=table,
        threads=threads,
        children=[],
    )

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(*_args, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, interaction_id, identity_scope, **_kwargs):
        return make_prepared_run(interaction_id=interaction_id, identity_scope=identity_scope)

    async def after_cpu(stage, identity_scope):
        # 这些真实阶段调用能到达 owner，说明进程 context 仍可授权；CPU 令牌已不可用。
        assert identity_scope.actor_identity == ActorIdentity(user_id="u1", agent_id="omni_doll")
        assert identity_scope.workspace_identity == access.default_workspace
        with pytest.raises(ThreadEndedError):
            await chain.cpu.operation_entry.execute(
                ResolveReferencesRequest(("missing",)), access=chain.cpu.calls[0].access
            )
        for child in chain.children:
            with pytest.raises(ThreadEndedError):
                await chain.cpu.operation_entry.execute(
                    ResolveReferencesRequest(("missing",)), access=child
                )
        chain.stages.append(stage)

    async def finalize(*, identity_scope, **_kwargs):
        assert chain.cpu.finished
        await after_cpu("finalize", identity_scope)
        return []

    async def topics(*, identity_scope, **_kwargs):
        await after_cpu("topics", identity_scope)
        return []

    async def cleanup(*, identity_scope, **_kwargs):
        await after_cpu("cleanup", identity_scope)
        return True

    for route, handler in (
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.FINALIZE_AGENT_RUN, finalize),
        (PatchouliRoutes.TOPIC_LIST_ACTIVE, topics),
        (PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup),
    ):
        bus.register(route, handler)
    yield chain
    runtime.close()
    access.gateway.close()
    access.gateway.revoke_all_contexts()


async def register(chain, cpu):
    """按生产装配把真实授权者、生命周期端口和共享登记注入任务进程。"""
    chain.cpu = cpu
    chain.service = make_task_process_service(
        chain.bus,
        cpu=cpu,
        access_gateway=chain.access.gateway,
        operation_authorizer=chain.access.authorizer,
        workspace_runtime=chain.runtime,
        process_table=chain.table,
    )
    return await chain.service.register_process(
        adapter="local",
        principal=chain.access.principal,
        actor=ActorIdentity(user_id="u1", agent_id="omni_doll"),
        workspace=chain.access.default_workspace,
        process_id="thread-lifetime",
        message="测试线程寿命",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cpu_outcome, expected_stages",
    [
        pytest.param(
            {"result": make_cpu_result(status="completed")},
            ["finalize"],
            id="completed",
        ),
        pytest.param({"error": RuntimeError("CPU failed")}, ["cleanup"], id="exception"),
    ],
)
async def test_cpu_end_revokes_thread_before_terminal_and_keeps_stage_authorization(
    process_chain, cpu_outcome, expected_stages
):
    """有终态与无终态两条撤销路径都先于交付撤销线程；finalize 或 cleanup 仍以进程身份调用。

    三种终态共用取得结果后的统一撤销，异常与缺失终态共用无结果时的提前撤销；
    各取一个代表。无输出终态与缺失终态的进程结局由 CPU 端口单测覆盖。
    """
    chain = process_chain
    observed = []

    async def inspect_thread(_submit):
        summary = chain.access.gateway.describe_context(chain.cpu.calls[0].access)
        observed.append((summary.run_type, summary.run_id, summary.thread_id))

    cpu = DerivingCPU(chain, **cpu_outcome, operation_script=inspect_thread)
    handle = await register(chain, cpu)
    stream = chain.service.run_process(handle, stream=True)
    try:
        async with asyncio.timeout(2):
            while (await anext(stream))["event"] not in {"done", "error", "cancelled"}:
                pass
        assert chain.access.gateway.describe_context(cpu.calls[0].access) is None
        # 终态交付后继续耗尽，确保注册入口的关闭与 cleanup 也执行完成。
        assert [event async for event in stream] == []
    finally:
        await stream.aclose()

    assert observed == [("execution_thread", "thread-lifetime", "main")]
    assert chain.stages == expected_stages


class DerivingCPU(ScriptedCPU):
    """CPU 运行期间派生两个子线程，覆盖所有终态的统一撤销。"""

    def __init__(self, chain, **kwargs):
        super().__init__(**kwargs)
        self.chain = chain

    async def execute(self, manifest, *, access, **kwargs):
        for agent_id in ("worker", "reviewer"):
            child = self.chain.threads.derive_child(
                parent=access, profile=AgentProfile(agent_id=agent_id)
            )
            self.chain.children.append(child.access)
        return await super().execute(manifest, access=access, **kwargs)


async def start_hanging_process(chain):
    """CPU 挂起在终态之前，使停止与取消都落在 Actor 阶段内。"""
    cpu = DerivingCPU(chain, hang_before_result=True)
    handle = await register(chain, cpu)
    return cpu, asyncio.create_task(chain.service.run_process(handle, stream=False))


@pytest.mark.asyncio
async def test_user_stop_revokes_thread_and_keeps_cleanup_authorized(process_chain):
    """用户停止撤销线程，cleanup 仍以尚有效的进程 context 执行。"""
    chain = process_chain
    cpu, task = await start_hanging_process(chain)
    try:
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
        requestor = await chain.access.authenticate(agent_id="omni_doll")
        assert chain.service.cancel_process("thread-lifetime", access=requestor).cancelled
        terminal = await asyncio.wait_for(task, timeout=2)

        assert terminal.execution_result.status == "cancelled"
        with pytest.raises(ThreadEndedError):
            await cpu.operation_entry.execute(
                ResolveReferencesRequest(("missing",)), access=cpu.calls[0].access
            )
        assert chain.stages == ["cleanup"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
