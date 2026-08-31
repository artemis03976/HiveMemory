"""Passive accumulator 到 Interaction Submission Queue 的 Q2 适配测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hivememory.core.errors import WorkspaceMismatchError
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
from hivememory.system.config.passive import PassiveIngressConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler
from hivememory.system.runtime.work_queue import QueuePolicy, WorkQueueCapacityError
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)
from tests.helpers.workspace import make_identity_scope

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
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
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
    assert submission.interaction_id.startswith("passive_")
    assert submission.identity_scope.actor_identity.user_id == "u1"
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


def test_conversation_key_does_not_partition_by_workspace() -> None:
    """捕获相同 passive 领域会话因 IdentityScope 不同而被隐式分区的缺陷。"""
    main = PassiveConversationKey.build(
        source="codex",
        external_conversation_id="conversation-shared",
        identity_scope=make_identity_scope(
            user_id="u1",
            agent_id="a1",
            workspace_id="main_workspace",
        ),
    )
    isolated = PassiveConversationKey.build(
        source="codex",
        external_conversation_id="conversation-shared",
        identity_scope=make_identity_scope(
            user_id="u1",
            agent_id="a1",
            workspace_id="isolation_workspace",
        ),
    )

    assert main == isolated
    assert main.ordering_key == isolated.ordering_key


@pytest.mark.asyncio
async def test_pending_turn_rejects_workspace_scope_drift() -> None:
    """捕获共享 conversation key 把另一 Workspace 事件混入原 turn 的缺陷。"""
    ingressor, queue = _build()
    main_scope = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="main_workspace",
    )
    isolated_scope = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolation_workspace",
    )

    await ingressor.route_event_scoped(
        _event("user", "main user", external_event_id="event-main"),
        main_scope,
        "interaction-main",
    )

    with pytest.raises(WorkspaceMismatchError) as exc_info:
        await ingressor.route_event_scoped(
            _event(
                "assistant",
                "isolated assistant",
                external_event_id="event-isolated",
                is_final=True,
            ),
            isolated_scope,
            "interaction-isolated",
        )

    buffer = ingressor.buffers.peek_buffer(_key())
    prepared = buffer.prepare_flush()
    assert exc_info.value.code == "workspace.mismatch"
    assert prepared is not None
    assert prepared[0].user_message == "main user"
    assert prepared[0].assistant_final_text is None
    assert queue.submissions == []


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
    assert await ingressor.flush_conversation(_key()) == 1
    assert queue.submissions[0].interaction_id == interaction_id
    assert not buffer.has_pending_round
    assert buffer.interaction_id is None


@pytest.mark.asyncio
async def test_same_explicit_final_event_retries_admission_without_reappend() -> None:
    ingressor, queue = _build()
    event = _event(
        "user",
        "hello",
        external_event_id="event-final-retry",
        is_final=True,
    )
    queue.fail_submit = True

    with pytest.raises(WorkQueueCapacityError):
        await ingressor.route_event(event, IDENTITY)

    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer is not None
    interaction_id = buffer.interaction_id
    assert buffer.event_count == 1
    assert buffer.pending_final_event_key == event.dedup_key

    queue.fail_submit = False
    outcome = await ingressor.route_event(event, IDENTITY)

    assert outcome.kind == "user"
    assert len(queue.submissions) == 1
    assert queue.submissions[0].interaction_id == interaction_id
    assert queue.submissions[0].payload.user_message == "hello"
    assert len(queue.submissions[0].payload.turn_events) == 1
    assert not buffer.has_pending_round

    duplicate = await ingressor.route_event(event, IDENTITY)
    assert duplicate.kind == "duplicate"
    assert len(queue.submissions) == 1


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
    attempts: list[str] = []

    async def submit_interaction(payload, **kwargs) -> str:
        attempts.append(payload.user_message)
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
        config = MagicMock()
        config.gateway.workflow.default_request_timeout_ms = 100
        config.passive_ingress = PassiveIngressConfig()
        config.scheduler.enabled = False
        service = PassiveIngressService(
            bus=bus,
            config=config,
            scheduler=AsyncMaintenanceScheduler(),
            interaction_queue=queue,
        )

        # 先提交一个已接受的 user turn 进入 queue
        await service.ingressor.route_event(
            _event("user", "u1", external_event_id="event-1", is_final=True),
            IDENTITY,
        )

        result = await service.shutdown_drain()

        # drain_all 等待并执行了已接受的 submission work
        assert attempts == ["u1"]
        assert result["observer_payloads_pending"] == 0
        assert result["success"] is True
    finally:
        await queue.stop()
