"""真实进程与操作入口协作，验证 CPU 边界上的线程寿命与阶段收尾。"""

import asyncio
from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, ResolvedAgentProfile
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.workspace.contracts import ResolveReferencesRequest, ThreadContextRevokedError
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
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    bus = GlobalSystemBus()
    runtime = make_workspace_runtime(bus)
    chain = SimpleNamespace(access=access, bus=bus, runtime=runtime, stages=[])

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
        with pytest.raises(ThreadContextRevokedError):
            await chain.cpu.operation_entry.execute(
                ResolveReferencesRequest(("missing",)), access=chain.cpu.calls[0].access
            )
        chain.stages.append(stage)

    async def finalize(*, identity_scope, **_kwargs):
        assert chain.cpu.closed
        await after_cpu("finalize", identity_scope)
        return []

    async def topics(*, identity_scope, **_kwargs):
        await after_cpu("topics", identity_scope)
        return []

    async def cleanup(*, identity_scope, **_kwargs):
        await after_cpu("cleanup", identity_scope)
        return True

    for route, handler in (
        (GlobalRoutes.GATEWAY_PROCESS, gateway),
        (GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile),
        (GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare),
        (GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize),
        (GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, topics),
        (GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup),
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
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "exception", "no-result"])
async def test_cpu_end_revokes_thread_before_terminal_and_keeps_stage_authorization(
    process_chain, status
):
    """正常终态、异常和缺失终态均立即撤销；finalize/话题池或 cleanup 仍以进程身份调用。"""
    chain = process_chain
    observed = []

    async def inspect_thread(_submit):
        summary = chain.access.gateway.describe_context(chain.cpu.calls[0].access)
        observed.append((summary.run_type, summary.run_id, summary.thread_id))

    cpu = ScriptedCPU(
        result=(
            make_cpu_result(status=status)
            if status in {"completed", "failed", "cancelled"}
            else None
        ),
        error=RuntimeError("CPU failed") if status == "exception" else None,
        operation_script=inspect_thread,
    )
    handle = await register(chain, cpu)
    stream = chain.service.run_process(handle, stream=True)
    try:
        async with asyncio.timeout(2):
            while (await anext(stream))["event"] not in {"done", "error", "cancelled"}:
                pass
        with pytest.raises(ThreadContextRevokedError):
            chain.access.authorizer.validate_thread_context(cpu.calls[0].access)
        # 终态交付后继续耗尽，确保注册入口的关闭与 cleanup 也执行完成。
        assert [event async for event in stream] == []
    finally:
        await stream.aclose()

    assert observed == [("execution_thread", "thread-lifetime", "main")]
    assert chain.stages == (["finalize", "topics"] if status == "completed" else ["cleanup"])


class ClosingCPU(ScriptedCPU):
    """让输出流的异步关闭挂起，以便观察关闭流程的第一个 await 之前的状态。"""

    def __init__(self):
        super().__init__(events=[{"event": "token", "data": {"content": "一"}}])
        self.close_entered = asyncio.Event()
        self.release_close = asyncio.Event()

    def execute(self, manifest, *, access, generation_options, stream):
        source = super().execute(
            manifest, access=access, generation_options=generation_options, stream=stream
        )
        return self.output(source)

    async def output(self, source):
        try:
            async for item in source:
                yield item
        finally:
            self.close_entered.set()
            await self.release_close.wait()
            await source.aclose()


@pytest.mark.asyncio
async def test_disconnect_revokes_thread_before_awaiting_cpu_close(process_chain):
    """交付方断流时先同步撤销，CPU 输出关闭挂起也不能继续提交 actor 请求。"""
    chain = process_chain
    cpu = ClosingCPU()
    handle = await register(chain, cpu)
    stream = chain.service.run_process(handle, stream=True)
    close_task = None
    try:
        async with asyncio.timeout(2):
            while (await anext(stream))["event"] != "token":
                pass
        close_task = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(cpu.close_entered.wait(), timeout=2)
        reads_before_request = chain.runtime.stats()["cold_reads"]
        with pytest.raises(ThreadContextRevokedError):
            await cpu.operation_entry.execute(
                ResolveReferencesRequest(("missing",)), access=cpu.calls[0].access
            )
        assert chain.runtime.stats()["cold_reads"] == reads_before_request
        cpu.release_close.set()
        await asyncio.wait_for(close_task, timeout=2)
        assert chain.stages == ["cleanup"]
    finally:
        cpu.release_close.set()
        if close_task is not None:
            await asyncio.wait_for(close_task, timeout=2)
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["stop", "owner-task"])
async def test_interrupted_actor_revokes_thread_and_keeps_cleanup_authorized(
    process_chain, cancellation
):
    """用户停止与调用方取消都撤销线程，cleanup 仍以尚有效的进程 context 执行。"""
    chain = process_chain
    cpu = ScriptedCPU(hang_before_result=True)
    handle = await register(chain, cpu)
    task = asyncio.create_task(chain.service.run_process(handle, stream=False))
    try:
        await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
        if cancellation == "stop":
            requestor = await chain.access.authenticate(agent_id="omni_doll")
            result = chain.service.cancel_process("thread-lifetime", access=requestor)
            assert result.cancelled
            terminal = await asyncio.wait_for(task, timeout=2)
            assert terminal.execution_result.status == "cancelled"
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        with pytest.raises(ThreadContextRevokedError):
            await cpu.operation_entry.execute(
                ResolveReferencesRequest(("missing",)), access=cpu.calls[0].access
            )
        assert chain.stages == ["cleanup"]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
