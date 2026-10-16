"""任务进程、真实能力层与意图登记的生命周期协作测试。

Gateway/prepare/finalize 位于本边界之外，由公开路由替身提供；CPU 经真实
操作入口提交请求，验证实时派发、终态不撤回与 CPU 结束后的线程 context 拒绝。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.errors import ScopeRequiredError
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
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.contracts import (
    ResolveReferencesRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
    ThreadContextRevokedError,
)
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import make_access_composition, make_actor_access_record

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
@pytest.mark.parametrize("new_topic", [False, True])
async def test_process_terminal_status_preserves_dispatched_intents(status, new_topic):
    """完成、取消与失败都不撤回已经派发的意图。"""
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    workspace = access.default_workspace
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    bus = GlobalSystemBus()
    admission = install_intent_admission(bus)
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
        return GatewayDecisionOutcome(
            decision=make_gateway_decision(target_topic_id="NEW_TOPIC" if new_topic else "topic-1")
        )

    async def profile(*_args, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, interaction_id, identity_scope, **_kwargs):
        return make_prepared_run(
            interaction_id=interaction_id,
            identity_scope=identity_scope,
            topic_id="prebuilt-topic" if new_topic else "topic-1",
            is_new_topic=new_topic,
        )

    async def finalize(*, payload, **_kwargs):
        finalized.append(payload)
        return []

    async def cleanup(**_kwargs):
        return True

    for route, handler in (
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.FINALIZE_AGENT_RUN, finalize),
        (PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup),
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
        intent, scope = admission.submissions[0]
        assert (intent.topic_id, scope.actor_identity) == (cpu.calls[0].manifest.topic_id, actor)
        assert intent.topic_id == ("prebuilt-topic" if new_topic else "topic-1")
        current = runtime.intents.get(submitted[0].pending_alias, workspace)
        assert current.status == PendingAtomStatus.MATERIALIZING
        assert (
            runtime.intents.get(other.pending_alias, workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
        assert [payload.outcome for payload in finalized] == [status]
        # completed 同样关闭主线程 context，全部请求类型均拒绝继续执行。
        for request in (
            SubmitWriteIntentRequest(focus=WriteFocus(content="关闭后")),
            SubmitUpdateIntentRequest(base_alias="base", instruction="关闭后"),
            ResolveReferencesRequest(aliases=(other.pending_alias,)),
        ):
            with pytest.raises(ThreadContextRevokedError):
                await cpu.operation_entry.execute(request, access=cpu.calls[0].access)
        assert runtime.intents.size == 2
    finally:
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


class _RevokeAdmissionAtFinalize:
    """Actor 登记意图后切换到已撤销的真实准入登记，阶段调用必须即时拒绝。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._revoked = make_access_composition([]).authorizer
        self.revoked = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def authorize_stage(self, access, target_workspace):
        authorizer = self._revoked if self.revoked else self._inner
        return authorizer.authorize_stage(access, target_workspace)


@pytest.mark.asyncio
async def test_finalize_authorization_failure_preserves_dispatched_intents():
    """finalize 授权失败只影响交互提交，不撤回已经派发的意图。"""
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")]
    )
    workspace = access.default_workspace
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    bus = GlobalSystemBus()
    runtime = WorkspaceRuntime(
        backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
    )
    authorizer = _RevokeAdmissionAtFinalize(access.authorizer)
    submitted = []
    finalized = []

    async def write(submit):
        submitted.append(await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="本进程"))))
        authorizer.revoked = True

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
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.FINALIZE_AGENT_RUN, finalize),
        (PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup),
    ):
        bus.register(route, handler)
    service = make_task_process_service(
        bus,
        cpu=ScriptedCPU(result=make_cpu_result(status="completed"), operation_script=write),
        access_gateway=access.gateway,
        operation_authorizer=authorizer,
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
        with pytest.raises(ScopeRequiredError) as exc_info:
            await service.run_process(handle, stream=False)

        status = runtime.intents.get(submitted[0].pending_alias, workspace).status
        assert exc_info.value.details["reason"] == "actor_not_admitted"
        assert (status, finalized) == (PendingAtomStatus.MATERIALIZING, [])
    finally:
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_process_close_revokes_waiting_update_without_cancelling_request_task():
    """进程在 UPDATE 冷读期间关闭：拒绝登记，调用方任务不被线程 context 撤销取消。"""
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

    bus.register(PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES, read)
    runtime = WorkspaceRuntime(
        backing=BusCanonicalReadBackend(bus), atom_capacity=8, profile_capacity=8
    )
    update_tasks = []

    async def update(submit):
        # 外部适配器任务独立于 CPU 拉取任务，撤销线程 context 不得取消它。
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
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup),
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
        with pytest.raises(ThreadContextRevokedError):
            await asyncio.wait_for(task, timeout=1)
        assert task.cancelled() is False
        assert runtime.intents.size == 0
        with pytest.raises(ThreadContextRevokedError):
            await cpu.operation_entry.execute(
                ResolveReferencesRequest(aliases=("base",)), access=cpu.calls[0].access
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
