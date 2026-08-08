"""Passive seal 到 Interaction Submission Queue 的 Q2 适配测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.services.passive import PassiveIngressEvent, PassiveMessageIngressor


class _SubmissionQueueRecorder:
    def __init__(self) -> None:
        self.submissions = []

    async def submit(self, submission):
        self.submissions.append(submission)


@pytest.mark.asyncio
async def test_passive_final_seal_enters_common_submission_queue() -> None:
    async def gateway(**kwargs):
        return GatewayDecisionOutcome(
            decision=GatewayDecision(
                target_topic_id="topic-1",
                rewritten_query="hello",
                memory_write_signal=MemoryWriteSignal.WRITE,
                retrieval_plan=RetrievalPlan(mode=RetrievalMode.SKIP, top_k=0),
                intent_type=IntentType.RAG,
            )
        )

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    queue = _SubmissionQueueRecorder()
    ingressor = PassiveMessageIngressor(bus, interaction_queue=queue)
    identity = Identity(user_id="u1", agent_id="a1")

    outcome = await ingressor.route_event(
        PassiveIngressEvent(
            source="codex",
            external_conversation_id="conversation-1",
            external_event_id="event-1",
            turn_id="turn-1",
            role="user",
            content="hello",
            is_final=True,
        ),
        identity,
    )

    assert outcome.kind == "user"
    assert ingressor.outbox.pending_count() == 0
    assert len(queue.submissions) == 1
    submission = queue.submissions[0]
    assert submission.interaction_id.startswith("interaction_")
    assert submission.origin == "passive_memory"
    assert submission.requested_topic_id == "topic-1"
    assert submission.ordering_key == "codex/conversation-1@u1:a1:<no-team>"
    assert submission.correlation == {
        "source": "codex",
        "external_conversation_id": "conversation-1",
        "turn_id": "turn-1",
    }


@pytest.mark.asyncio
async def test_shutdown_waits_for_accepted_submission_work() -> None:
    queue = MagicMock()
    queue.drain_all = AsyncMock(return_value=1)
    queue.pending_count = AsyncMock(return_value=0)
    ingressor = MagicMock()
    ingressor.shutdown_drain = AsyncMock(
        return_value={
            "sealed_turns": 1,
            "submitted_turns": 1,
            "outbox_pending": 0,
        }
    )
    service = PassiveIngressService.__new__(PassiveIngressService)
    service._interaction_queue = queue
    service._ingressor = ingressor
    service._maintenance_registered = False

    result = await service.shutdown_drain()

    queue.drain_all.assert_awaited_once()
    queue.pending_count.assert_awaited_once()
    assert result == {
        "success": True,
        "observer_payloads_submitted": 1,
        "observer_payloads_pending": 0,
    }
