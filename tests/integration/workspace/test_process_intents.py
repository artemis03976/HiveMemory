"""任务进程、真实能力层与意图登记的生命周期协作测试。

Gateway/prepare/finalize 位于本边界之外，由公开路由替身提供；CPU 经真实
操作通道提交意图，验证 completed 认领、取消隔离与关闭后的调用拒绝。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, ResolvedAgentProfile
from hivememory.core.models.pending import PendingAtomStatus, WriteFocus
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.contracts import ProcessOperationsClosedError
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import make_access_composition, make_actor_access_record

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
async def test_process_claims_or_cancels_only_its_own_intents(status):
    """完成认领本进程，取消/失败只取消未认领记录，均不影响其他进程。"""
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    workspace = access.default_workspace
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    bus = GlobalSystemBus()
    runtime = WorkspaceRuntime(
        backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
    )
    other = runtime.intents.register_write(
        WriteFocus(content="另一个进程"),
        belong_to=workspace,
        from_actor=actor,
        process_id="other",
    )
    submitted = []
    finalized = []

    async def write(operations):
        submitted.append(await operations.submit_write_intent(WriteFocus(content="本进程")))

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(*_args, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, interaction_id, identity_scope, **_kwargs):
        return make_prepared_run(interaction_id=interaction_id, identity_scope=identity_scope)

    async def finalize(*, payload, **_kwargs):
        finalized.append(payload)
        return []

    async def cleanup(**_kwargs):
        return True

    for route, handler in (
        (GlobalRoutes.GATEWAY_PROCESS, gateway),
        (GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile),
        (GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare),
        (GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize),
        (GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup),
    ):
        bus.register(route, handler)
    cpu = ScriptedCPU(result=make_cpu_result(status=status), operation_script=write)
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        workspace_runtime=runtime,
    )
    handle = await service.register_process(
        adapter="local",
        principal=access.principal,
        actor=actor,
        workspace=workspace,
        process_id="current",
        message="登记",
    )
    try:
        result = await service.run_process(handle, stream=False)
        assert result.execution_result.status == status
        current = runtime.intents.get(submitted[0].pending_alias, workspace)
        expected = (
            PendingAtomStatus.MATERIALIZING
            if status == "completed"
            else PendingAtomStatus.CANCELLED
        )
        assert current.status == expected
        assert (
            runtime.intents.get(other.pending_alias, workspace).status == PendingAtomStatus.PENDING
        )
        if status == "completed":
            (payload,) = finalized
            (task,) = payload.materialize_tasks
            assert task.pending_alias == submitted[0].pending_alias
            assert task.intent_id == submitted[0].intent_id
            assert task.belong_to == workspace
            assert task.from_actor == actor
            assert task.focus.content == "本进程"
        else:
            assert finalized == []
        with pytest.raises(ProcessOperationsClosedError, match="closed"):
            await cpu.calls[0].operations.submit_write_intent(WriteFocus(content="关闭后"))
        with pytest.raises(ProcessOperationsClosedError, match="closed"):
            await cpu.calls[0].operations.submit_update_intent("base", "关闭后")
        with pytest.raises(ProcessOperationsClosedError, match="closed"):
            await cpu.calls[0].operations.resolve_references([other.pending_alias])
        assert runtime.intents.size == 2
    finally:
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_channel_close_cancels_update_waiting_for_cold_read():
    """关闭发生在 UPDATE 冷读期间时，不能在清理之后登记出新的 PENDING。"""
    from hivememory.workspace.capability.memory import MemoryApplicationService
    from hivememory.workspace.process.operations import ProcessOperationChannel

    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    context = await access.authenticate(agent_id="omni_doll")
    bus = GlobalSystemBus()
    entered = asyncio.Event()

    async def read(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES, read)
    runtime = WorkspaceRuntime(
        backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
    )
    memory = MemoryApplicationService(
        bus,
        operation_authorizer=access.authorizer,
        memory_reader=runtime.aliases,
        intent_registry=runtime.intents,
    )
    channel = ProcessOperationChannel(
        memory, access=context, target_workspace=access.default_workspace, process_id="waiting"
    )
    task = asyncio.create_task(channel.submit_update_intent("base", "新内容"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        channel.close()
        runtime.intents.cancel_process("waiting")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.intents.size == 0
        with pytest.raises(ProcessOperationsClosedError, match="closed"):
            await channel.resolve_references(["base"])
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
