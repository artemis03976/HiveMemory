"""任务进程、真实能力层与意图登记的生命周期协作测试。

Gateway/prepare/finalize 位于本边界之外，由公开路由替身提供；CPU 经真实
操作入口提交请求，验证 completed 认领、取消隔离与关闭后的凭据拒绝。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    ResolvedAgentProfile,
)
from hivememory.core.models.pending import PendingAtomStatus, WriteFocus
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.contracts import (
    CancelIntentsRequest,
    ExecutionCredentialRevokedError,
    ResolveReferencesRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.memory import make_memory_metadata
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

    async def write(submit):
        submitted.append(await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="本进程"))))

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
        # completed 同样关闭主线程凭据，全部请求类型均拒绝继续执行。
        for request in (
            SubmitWriteIntentRequest(focus=WriteFocus(content="关闭后")),
            SubmitUpdateIntentRequest(base_alias="base", instruction="关闭后"),
            ResolveReferencesRequest(aliases=(other.pending_alias,)),
            CancelIntentsRequest(aliases=(other.pending_alias,)),
        ):
            with pytest.raises(ExecutionCredentialRevokedError):
                await cpu.operation_entry.execute(request, credential=cpu.calls[0].credential)
        assert runtime.intents.size == 2
    finally:
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


class _DenySubmitAtFinalize:
    """真实授权者的包装：进入 Actor 前的 interaction.submit 预检照常通过，
    finalize 时的同一检查被拒绝，模拟两次检查之间权限失效。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._submit_checks = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def authorize_operation(self, access, operation, target_workspace):
        if operation == WorkspaceOperation.INTERACTION_SUBMIT:
            self._submit_checks += 1
            if self._submit_checks > 1:
                raise OperationDeniedError(details={"reason": "revoked_for_test"})
        return self._inner.authorize_operation(access, operation, target_workspace)


@pytest.mark.asyncio
async def test_finalize_authorization_failure_cancels_instead_of_stranding_intents():
    """finalize 授权在认领之前：被拒绝时意图未被认领，随关闭取消，不停在 MATERIALIZING。"""
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    workspace = access.default_workspace
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    bus = GlobalSystemBus()
    runtime = WorkspaceRuntime(
        backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
    )
    submitted = []
    finalized = []

    async def write(submit):
        submitted.append(await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="本进程"))))

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
    service = make_task_process_service(
        bus,
        cpu=ScriptedCPU(result=make_cpu_result(status="completed"), operation_script=write),
        access_gateway=access.gateway,
        operation_authorizer=_DenySubmitAtFinalize(access.authorizer),
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
        with pytest.raises(OperationDeniedError) as exc_info:
            await service.run_process(handle, stream=False)

        status = runtime.intents.get(submitted[0].pending_alias, workspace).status
        assert exc_info.value.details["reason"] == "revoked_for_test"
        assert (status, finalized) == (PendingAtomStatus.CANCELLED, [])
    finally:
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_process_close_revokes_waiting_update_without_cancelling_request_task():
    """进程在 UPDATE 冷读期间关闭：入口补偿登记，调用方任务不被凭据吊销取消。"""
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    bus = GlobalSystemBus()
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

    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES, read)
    runtime = WorkspaceRuntime(
        backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
    )
    update_tasks = []

    async def update(submit):
        # 外部适配器任务独立于 CPU 拉取任务，吊销凭据不得取消它。
        update_tasks.append(
            asyncio.create_task(
                submit(SubmitUpdateIntentRequest(base_alias="base", instruction="新内容"))
            )
        )
        await entered.wait()

    async def profile(*_args, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def prepare(*, interaction_id, identity_scope, **_kwargs):
        return make_prepared_run(interaction_id=interaction_id, identity_scope=identity_scope)

    async def cleanup(**_kwargs):
        return True

    for route, handler in (
        (GlobalRoutes.GATEWAY_PROCESS, gateway),
        (GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile),
        (GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare),
        (GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup),
    ):
        bus.register(route, handler)
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "一"}}],
        hang_before_result=True,
        operation_script=update,
    )
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
        actor=ActorIdentity(user_id="u1", agent_id="omni_doll"),
        workspace=access.default_workspace,
        process_id="waiting",
        message="冷读 UPDATE",
    )
    stream = service.run_process(handle, stream=True)
    try:
        async with asyncio.timeout(1):
            while (await anext(stream))["event"] != "token":
                pass
        await stream.aclose()
        (task,) = update_tasks
        assert task.cancelling() == 0
        assert runtime.intents.size == 0
        release.set()
        with pytest.raises(ExecutionCredentialRevokedError):
            await asyncio.wait_for(task, timeout=1)
        assert task.cancelled() is False
        assert runtime.intents.size == 1
        assert runtime.intents.claim_process("waiting") == []
        with pytest.raises(ExecutionCredentialRevokedError):
            await cpu.operation_entry.execute(
                ResolveReferencesRequest(aliases=("base",)), credential=cpu.calls[0].credential
            )
    finally:
        await stream.aclose()
        for task in update_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*update_tasks, return_exceptions=True)
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
