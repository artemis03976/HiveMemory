"""Phase 3F 被动入口 GatewayDecision 契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    GatewayCommandOutcome,
    GatewayDecision,
    GatewayDecisionOutcome,
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.system.application.passive import (
    PassiveIngressEvent,
    PassiveMessageIngressor,
)
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


def _decision_outcome(
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id="topic-passive",
            rewritten_query="被动原问题",
            search_keywords=("被动",),
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(
                mode=mode,
                top_k=5 if mode != RetrievalMode.SKIP else 0,
            ),
            intent_type=IntentType.RAG,
        )
    )


@pytest.mark.asyncio
async def test_passive_user_requests_gateway_then_patchouli_retrieval() -> None:
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_decision_outcome())
    retrieve = AsyncMock(return_value=RetrievalResponse())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        gateway_request_timeout_ms=123,
    )
    identity = Identity(user_id="u1", session_id="s1")

    outcome = await ingressor.route_event(
        PassiveIngressEvent(role="user", content="被动原问题"),
        identity,
    )

    assert outcome.gateway_decision == _decision_outcome().decision
    assert gateway.await_args.kwargs["ingress_mode"] == (
        GatewayIngressMode.PASSIVE_MEMORY
    )
    assert gateway.await_args.kwargs["request_timeout_ms"] == 123
    request = retrieve.await_args.kwargs["request"]
    assert request.semantic_query == "被动原问题"
    assert request.top_k == 5

    flushed = ingressor.flush_session(identity)
    assert flushed is not None
    payload, target_topic = flushed
    assert target_topic == "topic-passive"
    assert payload.rewritten_query == "被动原问题"
    assert payload.worth_saving is True


@pytest.mark.asyncio
async def test_passive_simple_chat_skips_retrieval() -> None:
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome(RetrievalMode.SKIP)),
    )
    ingressor = PassiveMessageIngressor(bus)

    outcome = await ingressor.route_event(
        PassiveIngressEvent(role="user", content="你好"),
        Identity(user_id="u1"),
    )

    assert outcome.retrieval_result is not None
    assert outcome.retrieval_result.is_empty()
    assert GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE not in bus.list_routes()


@pytest.mark.asyncio
async def test_passive_rejects_impossible_command_outcome() -> None:
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(
            return_value=GatewayCommandOutcome(
                command_execution_result=CommandExecutionResult(
                    command_id="system.clear",
                    status=CommandExecutionStatus.COMPLETED,
                    message="clear",
                )
            )
        ),
    )

    with pytest.raises(RuntimeError, match="不得返回 command"):
        await PassiveMessageIngressor(bus).route_event(
            PassiveIngressEvent(role="user", content="/clear"),
            Identity(user_id="u1"),
        )
