"""Chat application 经共享总线并发传播 WorkspaceAccessContext 的集成测试。"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    RetrievalResponse,
)
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.workspace import make_access_context


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


def _prepared(identity_scope) -> PreparedAgentRun:
    return PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity_scope=identity_scope,
            interaction_id="interaction-test",
            topic_id="topic-shared-name",
            user_message="question",
            retrieval_result=RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
        ),
        gateway_decision=_decision().decision,
        stream_prelude=StreamPrelude(
            topic_id="topic-shared-name",
            is_new_topic=False,
            pool_topics=[],
            memory_refs=[],
        ),
    )


@pytest.mark.asyncio
async def test_concurrent_scoped_runs_keep_independent_contexts_on_shared_service() -> None:
    """防止共享 Chat/Gateway/Patchouli 单例保存并覆盖 current workspace。"""
    bus = GlobalSystemBus()
    service = ChatApplicationService(bus)
    both_gateway_calls_started = asyncio.Event()
    release_gateway = asyncio.Event()
    gateway_contexts = []
    finalized_contexts = []

    async def gateway(*, access_context, **_kwargs):
        gateway_contexts.append(access_context)
        if len(gateway_contexts) == 2:
            both_gateway_calls_started.set()
        await release_gateway.wait()
        return _decision()

    async def prepare(*, identity_scope, **_kwargs):
        return _prepared(identity_scope)

    async def alice(*, agent_run_context, **_kwargs):
        return AgentRunResult(final_text=agent_run_context.identity_scope.workspace_identity.workspace_id)

    async def finalize(*, prepared_run, **_kwargs):
        finalized_contexts.append(prepared_run.identity_scope)
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    main_context = make_access_context(
        user_id="u1",
        workspace_id="main_workspace",
    )
    isolation_context = make_access_context(
        user_id="u1",
        workspace_id="isolation_workspace",
    )
    main_task = asyncio.create_task(
        service.chat_scoped(
            "question",
            identity_scope=main_context,
            interaction_id="generation-main",
        )
    )
    isolation_task = asyncio.create_task(
        service.chat_scoped(
            "question",
            identity_scope=isolation_context,
            interaction_id="generation-isolation",
        )
    )

    await asyncio.wait_for(both_gateway_calls_started.wait(), timeout=1)
    assert service.generation_status_scoped(
        "generation-main",
        identity_scope=isolation_context,
    ) is None
    assert service.generation_status_scoped(
        "generation-isolation",
        identity_scope=main_context,
    ) is None

    release_gateway.set()
    main_result, isolation_result = await asyncio.wait_for(
        asyncio.gather(main_task, isolation_task),
        timeout=1,
    )

    assert main_result.agent_run_result.final_text == "main_workspace"
    assert isolation_result.agent_run_result.final_text == "isolation_workspace"
    assert {context.scope_fingerprint for context in gateway_contexts} == {
        main_context.scope_fingerprint,
        isolation_context.scope_fingerprint,
    }
    assert {context.scope_fingerprint for context in finalized_contexts} == {
        main_context.scope_fingerprint,
        isolation_context.scope_fingerprint,
    }


@pytest.mark.asyncio
async def test_chat_rejects_prepared_run_from_different_workspace_before_alice() -> None:
    """防止 prepare 返回漂移 scope 后继续执行 Alice 或 finalize 写入。"""
    bus = GlobalSystemBus()
    requested = make_access_context(
        user_id="u1",
        workspace_id="main_workspace",
    )
    drifted = make_access_context(
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
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    with pytest.raises(WorkspaceMismatchError, match="身份作用域不一致"):
        await ChatApplicationService(bus).chat_scoped(
            "question",
            identity_scope=requested,
            interaction_id="generation-drifted",
        )

    assert cleaned == [drifted]
