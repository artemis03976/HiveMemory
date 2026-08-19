"""Active interaction submission 的 Q4 applied gate 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, WriteFocus
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    InteractionPayload,
    RetrievalResponse,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
)
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import (
    ActiveInteractionFinalizationError,
    PatchouliService,
)
from hivememory.system.runtime.work_queue import QueuePolicy, WorkState
from tests.helpers.workspace import make_access_context
from tests.helpers.memory import make_memory_creation_context, make_memory_metadata


def _queue_policy(*, capacity: int = 8) -> QueuePolicy:
    return QueuePolicy(
        capacity=capacity,
        max_concurrency=1,
        ordered_by_key=True,
        cancellable=False,
        timeout_seconds=1,
        max_attempts=1,
        terminal_retention=16,
    )


def _decision() -> GatewayDecision:
    return GatewayDecision(
        target_topic_id="topic-1",
        rewritten_query="normalized question",
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )


def _memory() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="title",
            summary="memory summary",
            tags=[],
            memory_type=MemoryType.FACT,
            alias="memory_alias",
        ),
        payload=PayloadLayer(content="content"),
    )


def _prepared(
    *,
    interaction_id: str = "active-interaction-1",
    is_new: bool = False,
    memories: list[MemoryAtom] | None = None,
) -> PreparedAgentRun:
    identity = Identity(user_id="u1", agent_id="a1", session_id="session-1")
    access_context = make_access_context(
        actor_identity=identity,
        interaction_id=interaction_id,
    )
    return PreparedAgentRun(
        agent_run_context=AgentRunContext(
            access_context=access_context,
            topic_id="topic-1",
            user_message="question",
            topic_context=None,
            retrieval_result=RetrievalResponse(memories=memories or []),
            agent_profile=OMNI_DOLL_PROFILE,
        ),
        gateway_decision=_decision(),
        stream_prelude=StreamPrelude(
            topic_id="topic-1",
            is_new_topic=is_new,
            pool_topics=[],
            memory_refs=[],
        ),
    )


def _write_task() -> PendingAtomMaterializeTask:
    return PendingAtomMaterializeTask(
        pending_alias="draft_active",
        intent_id="intent_active",
        source_verb="WRITE",
        creation_context=make_memory_creation_context(user_id="u1", agent_id="a1"),
        focus=WriteFocus(content="remember this"),
    )


@pytest.mark.asyncio
async def test_active_finalize_waits_for_apply_before_follow_up_side_effects() -> None:
    calls: list[str] = []
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def apply(payload, *, target_topic_id, interaction_id):
        calls.append("apply_started")
        apply_started.set()
        await release_apply.wait()
        calls.append("apply")
        return target_topic_id

    async def materialize(*_args, **_kwargs):
        calls.append("materialize")
        return []

    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, materialize)
    queue = InteractionSubmissionQueue(apply, policy=_queue_policy())
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = _prepared()
    loop_result = AgentRunResult(
        final_text="answer",
        materialize_tasks=[_write_task()],
    )

    try:
        submit_spy = AsyncMock(wraps=queue.submit)
        queue.submit = submit_spy
        await queue.start()
        finalize_task = asyncio.create_task(service.finalize_agent_run(prepared, loop_result))
        await asyncio.wait_for(apply_started.wait(), timeout=1)
        assert calls == ["apply_started"]

        release_apply.set()
        await finalize_task

        submission = submit_spy.await_args.args[0]
        assert isinstance(submission, InteractionSubmission)
        assert submission.origin == "active_chat"
        assert submission.payload.access_context == prepared.access_context
        assert submission.requested_topic_id == prepared.topic_id
        assert submission.ordering_key == f"topic:{prepared.topic_id}"
        assert calls == ["apply_started", "apply", "materialize"]

        # 重复 finalize 复用原 work，interaction apply 不会再次执行。
        await service.finalize_agent_run(prepared, loop_result)
        assert calls.count("apply_started") == 1
        # finalize 可重新 dispatch；真正的幂等复用由下游 intent_id 边界保证。
        assert calls.count("materialize") == 2
    finally:
        release_apply.set()
        await queue.stop()


@pytest.mark.asyncio
async def test_active_finalize_keeps_retrieval_hit_in_owned_continuation() -> None:
    hit_started = asyncio.Event()
    release_hit = asyncio.Event()

    async def record_hit(*_args, **_kwargs) -> None:
        hit_started.set()
        await release_hit.wait()

    memory = _memory()
    record_hit_mock = AsyncMock(side_effect=record_hit)
    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit_mock)
    queue = InteractionSubmissionQueue(
        AsyncMock(return_value="topic-1"),
        policy=_queue_policy(),
    )
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = _prepared(memories=[memory, memory])
    finalize_task: asyncio.Task[list] | None = None

    try:
        await queue.start()
        finalize_task = asyncio.create_task(
            service.finalize_agent_run(
                prepared,
                AgentRunResult(final_text="answer"),
            )
        )

        await asyncio.wait_for(hit_started.wait(), timeout=1)
        assert not finalize_task.done()

        release_hit.set()
        assert await finalize_task == []
    finally:
        release_hit.set()
        if finalize_task is not None and not finalize_task.done():
            await asyncio.gather(finalize_task, return_exceptions=True)
        await queue.stop()

    record_hit_mock.assert_awaited_once_with(
        memory.id,
        access_context=prepared.access_context,
        source="retrieval.finalize",
    )


@pytest.mark.asyncio
async def test_terminal_apply_failure_stops_materialization_and_hit_record() -> None:
    apply = AsyncMock(side_effect=ConnectionError("temporary"))
    materialize = AsyncMock()
    record_hit = AsyncMock()
    discard = AsyncMock(return_value=True)
    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, materialize)
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    queue = InteractionSubmissionQueue(apply, policy=_queue_policy())
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = _prepared(is_new=True, memories=[_memory()])

    try:
        await queue.start()
        with pytest.raises(ActiveInteractionFinalizationError) as exc_info:
                await service.finalize_agent_run(
                    prepared,
                AgentRunResult(
                    final_text="answer",
                    materialize_tasks=[_write_task()],
                ),
            )
    finally:
        await queue.stop()

    assert exc_info.value.stage == "interaction_apply"
    assert exc_info.value.work_state == WorkState.DEAD_LETTER
    assert exc_info.value.error_class == "ConnectionError"
    materialize.assert_not_awaited()
    record_hit.assert_not_awaited()
    discard.assert_awaited_once_with(
        "topic-1",
        access_context=prepared.access_context,
    )


@pytest.mark.asyncio
async def test_queue_shutdown_returns_explicit_active_finalize_error() -> None:
    apply = AsyncMock(return_value="topic-1")
    queue = InteractionSubmissionQueue(apply, policy=_queue_policy())
    service = PatchouliService(PatchouliBus(), interaction_queue=queue)
    prepared = _prepared()

    finalize_task = asyncio.create_task(
        service.finalize_agent_run(prepared, AgentRunResult(final_text="answer"))
    )
    for _ in range(20):
        if await queue.is_accepted(prepared.interaction_id):
            break
        await asyncio.sleep(0)
    await queue.stop()

    with pytest.raises(ActiveInteractionFinalizationError) as exc_info:
        await finalize_task

    assert exc_info.value.stage == "interaction_apply"
    assert exc_info.value.reason == "queue_stopped"
    assert exc_info.value.work_state == WorkState.QUEUED
    apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_passive_backlog_capacity_rejects_active_before_side_effects() -> None:
    apply = AsyncMock(return_value="topic-1")
    materialize = AsyncMock()
    discard = AsyncMock(return_value=True)
    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, materialize)
    bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    queue = InteractionSubmissionQueue(apply, policy=_queue_policy(capacity=1))
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = _prepared(interaction_id="active-rejected", is_new=True)

    await queue.submit(
        InteractionSubmission(
            interaction_id="passive-pending",
            payload=InteractionPayload(
                access_context=make_access_context(
                    user_id="u1",
                    agent_id="a1",
                    interaction_id="passive-pending",
                ),
                user_message="passive question",
                assistant_final_text="passive answer",
            ),
            requested_topic_id="topic-1",
            ordering_key="topic:topic-1",
            origin="passive_memory",
            correlation={"turn_id": "passive-turn"},
        )
    )

    try:
        with pytest.raises(ActiveInteractionFinalizationError) as exc_info:
            await service.finalize_agent_run(
                prepared,
                AgentRunResult(
                    final_text="answer",
                    materialize_tasks=[_write_task()],
                ),
            )

        assert exc_info.value.stage == "interaction_admission"
        assert exc_info.value.reason == "capacity_rejected"
        materialize.assert_not_awaited()
        apply.assert_not_awaited()

        assert await service.cleanup_prepared_agent_run(prepared) is True
        discard.assert_awaited_once_with(
            prepared.topic_id,
            access_context=prepared.access_context,
        )
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_cancelled_wait_does_not_cancel_work_or_cleanup_topic() -> None:
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def apply(payload, *, target_topic_id, interaction_id):
        apply_started.set()
        await release_apply.wait()
        return target_topic_id

    discard = AsyncMock(return_value=True)
    materialize = AsyncMock(return_value=[])
    record_hit = AsyncMock()
    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, materialize)
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    queue = InteractionSubmissionQueue(apply, policy=_queue_policy())
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = _prepared(is_new=True, memories=[_memory()])

    try:
        await queue.start()
        finalize_task = asyncio.create_task(
            service.finalize_agent_run(
                prepared,
                AgentRunResult(
                    final_text="answer",
                    materialize_tasks=[_write_task()],
                ),
            )
        )
        await asyncio.wait_for(apply_started.wait(), timeout=1)
        finalize_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finalize_task

        assert await service.cleanup_prepared_agent_run(prepared) is False
        discard.assert_not_awaited()

        release_apply.set()
        await service.drain_active_finalizations()
        outcome = await queue.wait(prepared.interaction_id, timeout=1)
        assert outcome is not None
        assert outcome.state == WorkState.SUCCEEDED
        materialize.assert_awaited_once()
        record_hit.assert_awaited_once()
    finally:
        release_apply.set()
        await queue.stop()


@pytest.mark.asyncio
async def test_detached_apply_failure_cleans_new_empty_topic() -> None:
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()

    async def apply(payload, *, target_topic_id, interaction_id):
        apply_started.set()
        await release_apply.wait()
        raise ConnectionError("interaction store unavailable")

    discard = AsyncMock(return_value=True)
    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    queue = InteractionSubmissionQueue(apply, policy=_queue_policy())
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = _prepared(is_new=True)

    try:
        await queue.start()
        finalize_task = asyncio.create_task(
            service.finalize_agent_run(
                prepared,
                AgentRunResult(final_text="answer"),
            )
        )
        await asyncio.wait_for(apply_started.wait(), timeout=1)
        finalize_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finalize_task

        assert await service.cleanup_prepared_agent_run(prepared) is False

        release_apply.set()
        await service.drain_active_finalizations()
    finally:
        release_apply.set()
        await queue.stop()

    discard.assert_awaited_once_with(
        prepared.topic_id,
        access_context=prepared.access_context,
    )


@pytest.mark.asyncio
async def test_post_apply_materialization_failure_isolated_from_chat() -> None:
    queue = InteractionSubmissionQueue(
        AsyncMock(return_value="topic-1"),
        policy=_queue_policy(),
    )
    bus = PatchouliBus()
    bus.register(
        PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
        AsyncMock(side_effect=RuntimeError("generation unavailable")),
    )
    failed_aliases: list[str] = []

    async def capture_failure(*, pending_alias: str) -> None:
        failed_aliases.append(pending_alias)

    bus.subscribe(PatchouliLocalEvents.PENDING_ATOM_FAILED, capture_failure)
    service = PatchouliService(bus, interaction_queue=queue)

    try:
        await queue.start()
        result = await service.finalize_agent_run(
            _prepared(),
            AgentRunResult(
                final_text="answer",
                materialize_tasks=[_write_task()],
            ),
        )
    finally:
        await queue.stop()

    assert result == []
    assert failed_aliases == []


@pytest.mark.asyncio
async def test_pending_atom_failure_publish_does_not_reopen_chat_outcome() -> None:
    queue = InteractionSubmissionQueue(
        AsyncMock(return_value="topic-1"),
        policy=_queue_policy(),
    )
    bus = PatchouliBus()
    bus.register(
        PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
        AsyncMock(side_effect=RuntimeError("generation unavailable")),
    )
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_FAILED,
        AsyncMock(side_effect=ConnectionError("event subscriber unavailable")),
    )
    service = PatchouliService(bus, interaction_queue=queue)

    try:
        await queue.start()
        result = await service.finalize_agent_run(
            _prepared(),
            AgentRunResult(
                final_text="answer",
                materialize_tasks=[_write_task()],
            ),
        )
    finally:
        await queue.stop()

    assert result == []


@pytest.mark.asyncio
async def test_post_apply_retrieval_hit_failure_is_best_effort() -> None:
    queue = InteractionSubmissionQueue(
        AsyncMock(return_value="topic-1"),
        policy=_queue_policy(),
    )
    bus = PatchouliBus()
    record_hit = AsyncMock(side_effect=ConnectionError("hit store unavailable"))
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    service = PatchouliService(bus, interaction_queue=queue)

    try:
        await queue.start()
        result = await service.finalize_agent_run(
            _prepared(memories=[_memory()]),
            AgentRunResult(final_text="answer"),
        )
        await service.drain_active_finalizations()
    finally:
        await queue.stop()

    assert result == []
    record_hit.assert_awaited_once()


def test_default_interaction_policy_has_finite_attempt_timeout() -> None:
    queue = InteractionSubmissionQueue(AsyncMock(return_value="topic-1"))

    policy = queue.runtime.lanes[0].policy

    assert policy.timeout_seconds == 30.0
    assert policy.max_attempts == 3
    assert policy.cancellable is False
