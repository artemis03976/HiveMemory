"""任务进程编排（chat）经共享总线并发传播 IdentityScope 的集成测试。

访问边界：每个进程的 context 在
``register_process`` 经真实网关签发并绑定本进程；进程控制（状态查询与
取消）由请求级 context 经操作授权者的进程控制授权比对驻留坐标——跨
workspace 请求与不存在统一按 ``not_found`` 呈现。
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.models import ActorIdentity, ResolvedAgentProfile, TurnEvent
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    RetrievalPlan,
)
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUExecutionResult
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.chat_handoff import make_prepared_run
from tests.helpers.process import collect_events, done_data, make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_workspace_identity,
)


def _decision() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id="topic-shared-name",
            rewritten_query="question",
            retrieval_plan=RetrievalPlan(),
            intent_type=IntentType.RAG,
        )
    )


def _profile_route():
    """PATCHOULI_GET_AGENT_PROFILE 替身：返回 builtin omni_doll Profile。"""
    from hivememory.core.models import OMNI_DOLL_PROFILE

    async def route(agent_id, *, identity_scope):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    return route


def _prepared(identity_scope) -> PreparedAgentRun:
    return make_prepared_run(
        identity_scope=identity_scope,
        interaction_id="interaction-test",
        topic_id="topic-shared-name",
    )


def _composition():
    """两个 workspace（同 owner）的用户级登记 + 真实网关组合。"""
    return make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", workspace_id="main_workspace", agent_id=None
            ),
            make_actor_access_record(
                owner_user_id="u1", workspace_id="isolation_workspace", agent_id=None
            ),
        ],
        default_workspace=make_workspace_identity(
            owner_user_id="u1", workspace_id="main_workspace"
        ),
    )


def _service(bus: GlobalSystemBus, composition, cpu) -> TaskProcessService:
    return make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
    )


async def _register(service, composition, *, workspace, process_id: str, message: str):
    """经注册入口完成两阶段认证并登记进程（认证失败不创建进程）。"""
    return await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=ActorIdentity(user_id="u1", agent_id="a1"),
        workspace=workspace,
        process_id=process_id,
        message=message,
    )


class _WorkspaceEchoCPU:
    """按清单回显 workspace_id 的最小 CPU 实现：验证并发 run 的上下文隔离。

    轮次事件以本轮用户消息开头，回复正文由封口补成事件。
    """

    async def execute(self, manifest, *, thread, generation_options=None, events):
        return CPUExecutionResult(
            final_text=manifest.labels.workspace_id,
            turn_events=[TurnEvent(kind="user_message", sequence=0, content=manifest.user_message)],
        )


@pytest.mark.asyncio
async def test_concurrent_scoped_runs_keep_independent_contexts_on_shared_service() -> None:
    """防止共享 Chat/Gateway/Patchouli 单例保存并覆盖 current workspace。"""
    composition = _composition()
    bus = GlobalSystemBus()
    service = _service(bus, composition, _WorkspaceEchoCPU())
    main_ws = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
    isolation_ws = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")
    both_gateway_calls_started = asyncio.Event()
    release_gateway = asyncio.Event()
    gateway_contexts = []
    finalized_contexts = []

    async def gateway(*, identity_scope, **_kwargs):
        gateway_contexts.append(identity_scope)
        if len(gateway_contexts) == 2:
            both_gateway_calls_started.set()
        await release_gateway.wait()
        return _decision()

    async def prepare(*, identity_scope, **_kwargs):
        return _prepared(identity_scope)

    async def finalize(*, prepared_run, identity_scope, **_kwargs):
        finalized_contexts.append(identity_scope)
        return []

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route())
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

    main_task = asyncio.create_task(
        collect_events(
            service,
            await _register(
                service,
                composition,
                workspace=main_ws,
                process_id="process-main",
                message="question",
            ),
        )
    )
    isolation_task = asyncio.create_task(
        collect_events(
            service,
            await _register(
                service,
                composition,
                workspace=isolation_ws,
                process_id="process-isolation",
                message="question",
            ),
        )
    )

    await asyncio.wait_for(both_gateway_calls_started.wait(), timeout=1)

    release_gateway.set()
    main_result, isolation_result = await asyncio.wait_for(
        asyncio.gather(main_task, isolation_task),
        timeout=1,
    )

    assert done_data(main_result)["final_text"] == "main_workspace"
    assert done_data(isolation_result)["final_text"] == "isolation_workspace"
    assert {context for context in gateway_contexts} == {
        make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace"),
        make_identity_scope(user_id="u1", agent_id="a1", workspace_id="isolation_workspace"),
    }
    assert {context for context in finalized_contexts} == {
        make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace"),
        make_identity_scope(user_id="u1", agent_id="a1", workspace_id="isolation_workspace"),
    }


@pytest.mark.asyncio
async def test_cross_workspace_cancel_cannot_stop_the_other_run() -> None:
    """捕获共享进程表以裸 process_id 取消异域进程的缺陷。"""
    composition = _composition()
    bus = GlobalSystemBus()
    service = _service(bus, composition, _WorkspaceEchoCPU())
    main_ws = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
    isolation_ws = make_workspace_identity(owner_user_id="u1", workspace_id="isolation_workspace")
    both_gateway_calls_started = asyncio.Event()
    release_gateway = asyncio.Event()
    gateway_calls = 0

    async def gateway(*, identity_scope, **_kwargs):
        nonlocal gateway_calls
        gateway_calls += 1
        if gateway_calls == 2:
            both_gateway_calls_started.set()
        await release_gateway.wait()
        return _decision()

    async def prepare(*, identity_scope, **_kwargs):
        return _prepared(identity_scope)

    async def finalize(*, prepared_run, **_kwargs):
        return []

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_route())
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)

    main_task = asyncio.create_task(
        collect_events(
            service,
            await _register(
                service,
                composition,
                workspace=main_ws,
                process_id="process-run-main",
                message="main",
            ),
        )
    )
    isolated_task = asyncio.create_task(
        collect_events(
            service,
            await _register(
                service,
                composition,
                workspace=isolation_ws,
                process_id="process-run-isolated",
                message="isolated",
            ),
        )
    )

    main_request_access = await composition.authenticate(agent_id="a1", workspace=main_ws)
    try:
        await asyncio.wait_for(both_gateway_calls_started.wait(), timeout=1)
        cross_scope_cancel = service.cancel_process(
            "process-run-isolated",
            access=main_request_access,
        )

        assert cross_scope_cancel.cancelled is False
        assert cross_scope_cancel.status == "not_found"
    finally:
        release_gateway.set()
        await asyncio.wait_for(asyncio.gather(main_task, isolated_task), timeout=1)

    main_result, isolated_result = main_task.result(), isolated_task.result()
    assert done_data(main_result)["final_text"] == "main_workspace"
    assert done_data(isolated_result)["final_text"] == "isolation_workspace"
