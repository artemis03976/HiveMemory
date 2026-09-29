"""任务进程编排（chat）经共享总线并发传播 IdentityScope 的集成测试。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import ResolvedAgentProfile
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import AgentRunResult
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.workspace import make_identity_scope


def _decision() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id="topic-shared-name",
            rewritten_query="question",
            memory_write_signal=MemoryWriteSignal.WRITE,
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
        user_message="question",
        gateway_decision=make_gateway_decision(target_topic_id="topic-shared-name"),
        topic_id="topic-shared-name",
    )


@pytest.mark.asyncio
async def test_concurrent_scoped_runs_keep_independent_contexts_on_shared_service() -> None:
    """防止共享 Chat/Gateway/Patchouli 单例保存并覆盖 current workspace。"""
    bus = GlobalSystemBus()
    service = TaskProcessService(bus)
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

    async def alice(*, input_manifest, **_kwargs):
        return AgentRunResult(
            final_text=input_manifest.identity_scope.workspace_identity.workspace_id
        )

    async def finalize(*, prepared_run, **_kwargs):
        finalized_contexts.append(prepared_run.identity_scope)
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    main_context = make_identity_scope(
        user_id="u1",
        workspace_id="main_workspace",
    )
    isolation_context = make_identity_scope(
        user_id="u1",
        workspace_id="isolation_workspace",
    )
    main_task = asyncio.create_task(
        service.run_process(
            "question",
            stream=False,
            identity_scope=main_context,
            process_id="process-main",
        )
    )
    isolation_task = asyncio.create_task(
        service.run_process(
            "question",
            stream=False,
            identity_scope=isolation_context,
            process_id="process-isolation",
        )
    )

    await asyncio.wait_for(both_gateway_calls_started.wait(), timeout=1)
    assert (
        service.process_status(
            "process-main",
            identity_scope=isolation_context,
        )
        is None
    )
    assert (
        service.process_status(
            "process-isolation",
            identity_scope=main_context,
        )
        is None
    )

    release_gateway.set()
    main_result, isolation_result = await asyncio.wait_for(
        asyncio.gather(main_task, isolation_task),
        timeout=1,
    )

    assert main_result.agent_run_result.final_text == "main_workspace"
    assert isolation_result.agent_run_result.final_text == "isolation_workspace"
    assert {context for context in gateway_contexts} == {
        main_context,
        isolation_context,
    }
    assert {context for context in finalized_contexts} == {
        main_context,
        isolation_context,
    }


@pytest.mark.asyncio
async def test_chat_rejects_prepared_run_from_different_workspace_before_alice() -> None:
    """防止 prepare 返回漂移 scope 后继续执行 Alice 或 finalize 写入。"""
    bus = GlobalSystemBus()
    requested = make_identity_scope(
        user_id="u1",
        workspace_id="main_workspace",
    )
    drifted = make_identity_scope(
        user_id="u1",
        workspace_id="isolation_workspace",
    )
    cleaned = []

    async def gateway(**_kwargs):
        return _decision()

    async def prepare(**_kwargs):
        return _prepared(drifted)

    async def cleanup(*, prepared_run):
        cleaned.append(prepared_run.identity_scope)

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    with pytest.raises(WorkspaceMismatchError, match="身份作用域不一致"):
        await TaskProcessService(bus).run_process(
            "question",
            stream=False,
            identity_scope=requested,
            process_id="process-drifted",
        )

    assert cleaned == [drifted]


@pytest.mark.asyncio
async def test_cross_workspace_cancel_cannot_stop_the_other_run() -> None:
    """捕获共享进程表以裸 process_id 取消异域进程的缺陷。"""
    bus = GlobalSystemBus()
    service = TaskProcessService(bus)
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

    async def alice(*, input_manifest, **_kwargs):
        return AgentRunResult(
            final_text=input_manifest.identity_scope.workspace_identity.workspace_id,
        )

    async def finalize(*, prepared_run, **_kwargs):
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    main = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="main_workspace",
    )
    isolated = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolation_workspace",
    )
    main_task = asyncio.create_task(
        service.run_process(
            "main",
            stream=False,
            identity_scope=main,
            process_id="process-run-main",
        )
    )
    isolated_task = asyncio.create_task(
        service.run_process(
            "isolated",
            stream=False,
            identity_scope=isolated,
            process_id="process-run-isolated",
        )
    )

    try:
        await asyncio.wait_for(both_gateway_calls_started.wait(), timeout=1)
        cross_scope_cancel = service.cancel_process(
            "process-run-isolated",
            identity_scope=main,
        )

        assert cross_scope_cancel.cancelled is False
        assert cross_scope_cancel.status == "not_found"
        isolated_status = service.process_status(
            "process-run-isolated",
            identity_scope=isolated,
        )
        assert isolated_status.status == "running"
        assert (
            service.process_status(
                "process-run-isolated",
                identity_scope=main,
            )
            is None
        )
    finally:
        release_gateway.set()
        await asyncio.wait_for(asyncio.gather(main_task, isolated_task), timeout=1)

    main_result, isolated_result = main_task.result(), isolated_task.result()
    assert main_result.agent_run_result.final_text == "main_workspace"
    assert isolated_result.agent_run_result.final_text == "isolation_workspace"
