"""Passive accumulator 到 Interaction Submission Queue 的 Q2 适配测试。"""

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
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.work_queue import QueuePolicy, WorkQueueCapacityError
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)

IDENTITY = Identity(user_id="u1", agent_id="a1")


class _SubmissionQueueRecorder:
    def __init__(self) -> None:
        self.submissions = []
        self.fail_submit = False

    async def submit(self, submission):
        if self.fail_submit:
            raise WorkQueueCapacityError("patchouli.interaction_submission", 1)
        self.submissions.append(submission)


def _build() -> tuple[PassiveMessageIngressor, _SubmissionQueueRecorder]:
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
    return PassiveMessageIngressor(bus, interaction_queue=queue), queue


def _event(
    role: str,
    content: str,
    *,
    external_event_id: str,
    is_final: bool = False,
) -> PassiveIngressEvent:
    return PassiveIngressEvent(
        source="codex",
        external_conversation_id="conversation-1",
        external_event_id=external_event_id,
        turn_id="turn-1",
        role=role,
        content=content,
        is_final=is_final,
    )


def _key() -> PassiveConversationKey:
    return PassiveConversationKey.build(
        source="codex",
        external_conversation_id="conversation-1",
        identity=IDENTITY,
    )


@pytest.mark.asyncio
async def test_passive_final_enters_common_submission_queue() -> None:
    ingressor, queue = _build()

    outcome = await ingressor.route_event(
        _event("user", "hello", external_event_id="event-1", is_final=True),
        IDENTITY,
    )

    assert outcome.kind == "user"
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
        "seal_reason": "explicit_final",
    }
    assert not ingressor.buffers.peek_buffer(_key()).has_pending_round


@pytest.mark.asyncio
async def test_admission_failure_keeps_payload_and_interaction_id_for_retry() -> None:
    ingressor, queue = _build()
    queue.fail_submit = True

    with pytest.raises(WorkQueueCapacityError):
        await ingressor.route_event(
            _event("user", "hello", external_event_id="event-1", is_final=True),
            IDENTITY,
        )

    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer is not None
    interaction_id = buffer.interaction_id
    prepared = buffer.prepare_flush()
    assert prepared is not None
    assert prepared[0].user_message == "hello"
    assert buffer.has_pending_round

    queue.fail_submit = False
    assert await ingressor.flush_conversation(_key(), IDENTITY) == 1
    assert queue.submissions[0].interaction_id == interaction_id
    assert not buffer.has_pending_round
    assert buffer.interaction_id is None


@pytest.mark.asyncio
async def test_next_user_admission_failure_does_not_overwrite_previous_turn() -> None:
    ingressor, queue = _build()
    await ingressor.route_event(
        _event("user", "u1", external_event_id="event-1"),
        IDENTITY,
    )
    await ingressor.route_event(
        _event("assistant", "a1", external_event_id="event-2"),
        IDENTITY,
    )

    buffer = ingressor.buffers.peek_buffer(_key())
    previous_interaction_id = buffer.interaction_id
    next_user = _event("user", "u2", external_event_id="event-3")
    queue.fail_submit = True
    with pytest.raises(WorkQueueCapacityError):
        await ingressor.route_event(next_user, IDENTITY)

    prepared = buffer.prepare_flush()
    assert prepared is not None
    assert prepared[0].user_message == "u1"
    assert prepared[0].assistant_final_text == "a1"
    assert buffer.interaction_id == previous_interaction_id

    # admission 失败发生在新 user 写入前，原事件可用同一 external_event_id 重试。
    queue.fail_submit = False
    outcome = await ingressor.route_event(next_user, IDENTITY)
    assert outcome.kind == "user"
    assert queue.submissions[0].interaction_id == previous_interaction_id
    current = buffer.prepare_flush()
    assert current is not None
    assert current[0].user_message == "u2"


@pytest.mark.asyncio
async def test_apply_retry_does_not_block_next_accumulator() -> None:
    attempts: list[str] = []

    async def submit_interaction(payload, **kwargs) -> str:
        attempts.append(payload.user_message)
        if payload.user_message == "u1" and attempts.count("u1") == 1:
            raise ConnectionError("temporary perception failure")
        return f"topic-{payload.user_message}"

    queue = InteractionSubmissionQueue(
        submit_interaction,
        policy=QueuePolicy(
            capacity=8,
            max_concurrency=1,
            ordered_by_key=True,
            max_attempts=2,
        ),
    )
    await queue.start()
    try:

        async def gateway(**kwargs):
            return GatewayDecisionOutcome(
                decision=GatewayDecision(
                    target_topic_id="topic-1",
                    rewritten_query="hello",
                    memory_write_signal=MemoryWriteSignal.WRITE,
                    retrieval_plan=RetrievalPlan(
                        mode=RetrievalMode.SKIP,
                        top_k=0,
                    ),
                    intent_type=IntentType.RAG,
                )
            )

        bus = GlobalSystemBus()
        bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
        ingressor = PassiveMessageIngressor(bus, interaction_queue=queue)

        await ingressor.route_event(
            _event("user", "u1", external_event_id="event-1", is_final=True),
            IDENTITY,
        )
        await ingressor.route_event(
            _event("user", "u2", external_event_id="event-2", is_final=True),
            IDENTITY,
        )

        assert not ingressor.buffers.peek_buffer(_key()).has_pending_round
        assert await queue.drain_all(timeout=2.0) == 2
        assert attempts == ["u1", "u1", "u2"]
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_shutdown_waits_for_accepted_submission_work() -> None:
    queue = MagicMock()
    queue.drain_all = AsyncMock(return_value=1)
    queue.pending_count = AsyncMock(return_value=0)
    ingressor = MagicMock()
    ingressor.shutdown_drain = AsyncMock(
        return_value={
            "finalized_turns": 1,
            "accepted_submissions": 1,
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
