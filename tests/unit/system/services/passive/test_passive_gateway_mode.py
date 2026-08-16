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
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressContractError,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)

SOURCE = "unit_test"
CONVERSATION = "conv-1"


class _SubmissionQueueRecorder:
    def __init__(self, submitted: list | None = None) -> None:
        self.submitted = submitted if submitted is not None else []

    async def submit(self, submission) -> None:
        self.submitted.append(submission)


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


def _event(role: str, content: str, **kwargs) -> PassiveIngressEvent:
    return PassiveIngressEvent(
        source=SOURCE,
        external_conversation_id=CONVERSATION,
        role=role,
        content=content,
        **kwargs,
    )


def _key(identity: Identity) -> PassiveConversationKey:
    return PassiveConversationKey.build(
        source=SOURCE,
        external_conversation_id=CONVERSATION,
        identity=identity,
    )


@pytest.mark.asyncio
async def test_passive_user_requests_gateway_then_patchouli_retrieval() -> None:
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_decision_outcome())
    retrieve = AsyncMock(return_value=RetrievalResponse())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, retrieve)
    submitted = []
    ingressor = PassiveMessageIngressor(
        bus,
        interaction_queue=_SubmissionQueueRecorder(submitted),
        gateway_request_timeout_ms=123,
    )
    identity = Identity(user_id="u1", session_id="s1")

    outcome = await ingressor.route_event(_event("user", "被动原问题"), identity)

    assert outcome.gateway_decision == _decision_outcome().decision
    assert gateway.await_args.kwargs["ingress_mode"] == (GatewayIngressMode.PASSIVE_MEMORY)
    assert gateway.await_args.kwargs["request_timeout_ms"] == 123
    request = retrieve.await_args.kwargs["request"]
    assert request.semantic_query == "被动原问题"
    assert request.top_k == 5

    assert await ingressor.flush_conversation(_key(identity), identity) == 1
    assert len(submitted) == 1
    submission = submitted[0]
    assert submission.requested_topic_id == "topic-passive"
    assert submission.correlation["seal_reason"] == "manual_flush"
    assert submission.payload.rewritten_query == "被动原问题"
    assert submission.payload.worth_saving is True


@pytest.mark.asyncio
async def test_passive_simple_chat_skips_retrieval() -> None:
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome(RetrievalMode.SKIP)),
    )
    ingressor = PassiveMessageIngressor(
        bus,
        interaction_queue=_SubmissionQueueRecorder(),
    )

    outcome = await ingressor.route_event(
        _event("user", "你好"),
        Identity(user_id="u1"),
    )

    assert outcome.retrieval_result is not None
    assert outcome.retrieval_result.is_empty()


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

    ingressor = PassiveMessageIngressor(
        bus,
        interaction_queue=_SubmissionQueueRecorder(),
    )
    # 契约违约不走 §6 降级路径，必须向上抛出
    with pytest.raises(PassiveIngressContractError, match="不得返回 command"):
        await ingressor.route_event(_event("user", "/clear"), Identity(user_id="u1"))
