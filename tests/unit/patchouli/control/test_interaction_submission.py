"""Interaction Submission Queue 的 Q2 契约测试。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.engines.perception.semantic_flow_perception_layer import (
    NullPerceptionLayer,
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
    InteractionApplyStage,
)
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionCodec,
    InteractionSubmissionQueue,
    TransientInteractionSubmissionError,
)
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.system.runtime.work_queue import (
    QueuePolicy,
    WorkPayloadCodecRegistry,
    WorkPayloadDecodeError,
    WorkState,
    encode_canonical_json,
)
from tests.helpers.memory import make_memory_identity_scope
from tests.helpers.workspace import make_identity_scope


def _payload(message: str = "hello") -> InteractionPayload:
    return InteractionPayload(
        user_message=message,
        assistant_final_text=f"answer:{message}",
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=f"answer:{message}",
            )
        ],
    )


def _submission(
    interaction_id: str,
    *,
    message: str | None = None,
    ordering_key: str = "conversation-1",
    payload: InteractionPayload | None = None,
) -> InteractionSubmission:
    return InteractionSubmission(
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        interaction_id=interaction_id,
        payload=payload or _payload(message or interaction_id),
        requested_topic_id="NEW_TOPIC",
        ordering_key=ordering_key,
        origin="passive_memory",
        correlation={"turn_id": interaction_id},
    )


@pytest.mark.asyncio
async def test_enqueue_uses_payload_snapshot_and_each_retry_gets_fresh_dto() -> None:
    attempts: list[InteractionPayload] = []
    attempt_scopes = []

    async def submit(payload, *, identity_scope, target_topic_id, interaction_id):
        attempts.append(payload)
        attempt_scopes.append(identity_scope)
        if len(attempts) == 1:
            payload.user_message = "attempt-local-mutation"
            payload.turn_events.clear()
            raise TransientInteractionSubmissionError("temporary failure")
        return "topic-real"

    queue = InteractionSubmissionQueue(submit)
    original = _payload("original")
    submission = _submission("interaction-1", payload=original)
    assert submission.payload is original

    try:
        await queue.start()
        receipt = await queue.submit(submission)

        # 入队后的调用方修改不得影响 canonical bytes 中的工作快照。
        original.user_message = "external-mutation"
        original.turn_events.clear()
        submission.payload.user_message = "submission-mutation"
        submission.payload.turn_events.clear()

        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    assert outcome.topic_id == "topic-real"
    assert len(attempts) == 2
    assert attempts[0] is not attempts[1]
    assert attempts[1].user_message == "original"
    assert len(attempts[1].turn_events) == 1
    assert attempt_scopes == [submission.identity_scope, submission.identity_scope]
    assert attempt_scopes[0] is not attempt_scopes[1]


@pytest.mark.asyncio
async def test_same_ordering_key_keeps_fifo_during_retry() -> None:
    calls: list[str] = []
    first_attempt = 0

    async def submit(payload, *, identity_scope, target_topic_id, interaction_id):
        nonlocal first_attempt
        calls.append(payload.user_message)
        if payload.user_message == "first":
            first_attempt += 1
            if first_attempt == 1:
                raise TransientInteractionSubmissionError("retry first")
        return f"topic:{payload.user_message}"

    queue = InteractionSubmissionQueue(submit)
    try:
        await queue.submit(_submission("interaction-first", message="first"))
        second = await queue.submit(_submission("interaction-second", message="second"))
        await queue.start()
        second_outcome = await queue.wait(second, timeout=2)
    finally:
        await queue.stop()

    assert second_outcome is not None
    assert second_outcome.state == WorkState.SUCCEEDED
    assert calls == ["first", "first", "second"]


@pytest.mark.asyncio
async def test_different_ordering_keys_can_execute_concurrently() -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()

    async def submit(payload, *, identity_scope, target_topic_id, interaction_id):
        if payload.user_message == "first":
            first_started.set()
        else:
            second_started.set()
        await release.wait()
        return f"topic:{payload.user_message}"

    queue = InteractionSubmissionQueue(submit)
    try:
        await queue.submit(
            _submission("interaction-a", message="first", ordering_key="conversation-a")
        )
        await queue.submit(
            _submission("interaction-b", message="second", ordering_key="conversation-b")
        )
        await queue.start()

        await asyncio.wait_for(
            asyncio.gather(first_started.wait(), second_started.wait()),
            timeout=1,
        )
        release.set()
        assert await queue.drain_all(timeout=2) == 2
    finally:
        release.set()
        await queue.stop()


@pytest.mark.asyncio
async def test_ambiguous_failure_after_add_block_does_not_duplicate_block() -> None:
    # 容量设为 1，确保 retry 的幂等快路径发生在 LRU 检查之前。
    store = ShortTermMemoryStore(max_resident_topics=1)
    interaction_journal = InMemoryInteractionApplyJournal()
    relay = Mock()
    relay.should_relay.return_value = None
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
        relay_controller=relay,
        short_term_store=store,
        interaction_journal=interaction_journal,
    )
    original_fold = layer._maybe_fold_pages
    fold_calls = 0

    async def fail_once_after_add(topic_key):
        nonlocal fold_calls
        fold_calls += 1
        if fold_calls == 1:
            raise TransientInteractionSubmissionError("caller missed apply result")
        return await original_fold(topic_key)

    layer._maybe_fold_pages = fail_once_after_add
    bus = Mock()
    bus.request = AsyncMock(return_value=None)
    familiar = PerceptionFamiliar(
        perception_layer=layer,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=30),
        memory_library=MemoryLibrary(
            short_term=store,
            mid_term=Mock(),
            long_term=Mock(),
        ),
        interaction_journal=interaction_journal,
    )
    queue = InteractionSubmissionQueue(familiar.apply_interaction)

    try:
        await queue.start()
        receipt = await queue.submit(_submission("interaction-ambiguous"))
        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    assert outcome.topic_id is not None
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic = store.get_topic_data(identity_scope, outcome.topic_id, touch=False)
    assert topic is not None
    assert topic.block_count == 1
    assert store.get_last_active_topic(identity_scope) == outcome.topic_id

    replayed_topic = await asyncio.wait_for(
        familiar.apply_interaction(
            _payload("interaction-ambiguous"),
            identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
            target_topic_id=outcome.topic_id,
            interaction_id="interaction-ambiguous",
        ),
        timeout=1,
    )
    assert replayed_topic == outcome.topic_id
    assert topic.block_count == 1


@pytest.mark.asyncio
async def test_unclassified_failure_is_not_retried() -> None:
    submit = AsyncMock(side_effect=RuntimeError("invalid apply state"))
    queue = InteractionSubmissionQueue(submit)

    try:
        await queue.start()
        receipt = await queue.submit(_submission("interaction-failed"))
        outcome = await queue.wait(receipt, timeout=1)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.FAILED
    assert outcome.error_class == "RuntimeError"
    submit.assert_awaited_once()


@pytest.mark.asyncio
async def test_handler_timeout_is_not_retried() -> None:
    attempts = 0

    async def submit(payload, *, identity_scope, target_topic_id, interaction_id):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    queue = InteractionSubmissionQueue(
        submit,
        policy=QueuePolicy(
            capacity=4,
            max_concurrency=1,
            timeout_seconds=0.01,
            max_attempts=3,
        ),
    )

    try:
        await queue.start()
        receipt = await queue.submit(_submission("interaction-timeout"))
        outcome = await queue.wait(receipt, timeout=1)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.FAILED
    assert outcome.error_class == "TimeoutError"
    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_resubmits_pending_settlement_without_duplicating_block() -> None:
    store = ShortTermMemoryStore(max_resident_topics=1)
    interaction_journal = InMemoryInteractionApplyJournal()
    relay = Mock()
    relay.should_relay.return_value = None
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
        relay_controller=relay,
        short_term_store=store,
        interaction_journal=interaction_journal,
    )
    settlement = TopicMaterializeTask(
        topic_id="topic-settlement",
        identity_scope=make_memory_identity_scope(user_id="u1", agent_id="a1"),
    )
    layer._maybe_fold_pages = AsyncMock(return_value=settlement)
    bus = Mock()
    bus.request = AsyncMock(
        side_effect=[ConnectionError("queue admission failed"), None]
    )
    familiar = PerceptionFamiliar(
        perception_layer=layer,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=30),
        memory_library=MemoryLibrary(
            short_term=store,
            mid_term=Mock(),
            long_term=Mock(),
        ),
        interaction_journal=interaction_journal,
    )
    queue = InteractionSubmissionQueue(familiar.apply_interaction)

    try:
        await queue.start()
        receipt = await queue.submit(_submission("interaction-settlement"))
        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    topic = store.get_topic_data(
        make_identity_scope(user_id="u1", agent_id="a1"),
        outcome.topic_id,
        touch=False,
    )
    assert topic is not None
    assert topic.block_count == 1
    assert bus.request.await_count == 2
    assert all(call.args[1] is settlement for call in bus.request.await_args_list)
    record = interaction_journal.get("interaction-settlement")
    assert record is not None
    assert record.stage is InteractionApplyStage.COMPLETED
    assert record.settlement_to_submit is None


@pytest.mark.asyncio
async def test_disabled_perception_does_not_require_apply_journal_entry() -> None:
    store = ShortTermMemoryStore()
    interaction_journal = InMemoryInteractionApplyJournal()
    familiar = PerceptionFamiliar(
        perception_layer=NullPerceptionLayer(),
        bus=Mock(request=AsyncMock()),
        config=SimpleNamespace(idle_timeout_seconds=30),
        memory_library=MemoryLibrary(
            short_term=store,
            mid_term=Mock(),
            long_term=Mock(),
        ),
        interaction_journal=interaction_journal,
    )

    topic_id = await familiar.apply_interaction(
        _payload(),
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        interaction_id="interaction-disabled",
    )
    replayed_topic_id = await asyncio.wait_for(
        familiar.apply_interaction(
            _payload(),
            identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
            interaction_id="interaction-disabled",
        ),
        timeout=1,
    )

    assert topic_id == "NEW_TOPIC"
    assert replayed_topic_id == "NEW_TOPIC"
    assert interaction_journal.get("interaction-disabled") is None


@pytest.mark.asyncio
async def test_submission_lookup_evicts_old_terminal_work() -> None:
    submit = AsyncMock(return_value="topic-real")
    queue = InteractionSubmissionQueue(
        submit,
        policy=QueuePolicy(
            capacity=1,
            max_concurrency=1,
            terminal_retention=1,
        ),
    )

    try:
        await queue.start()
        for index in range(3):
            receipt = await queue.submit(_submission(f"interaction-{index}"))
            outcome = await queue.wait(receipt, timeout=2)
            assert outcome is not None
            assert outcome.state == WorkState.SUCCEEDED
    finally:
        await queue.stop()

    assert await queue.wait("interaction-0") is None
    latest = await queue.wait("interaction-2")
    assert latest is not None
    assert latest.state == WorkState.SUCCEEDED
    assert await queue.pending_count() == 0


@pytest.mark.asyncio
async def test_duplicate_interaction_id_is_idempotent_but_rejects_another_payload() -> None:
    submit = AsyncMock(return_value="topic-real")
    queue = InteractionSubmissionQueue(submit)
    first = _submission("interaction-same", message="first")

    try:
        first_receipt = await queue.submit(first)
        same_receipt = await queue.submit(first)
        with pytest.raises(ValueError, match="already belongs"):
            await queue.submit(_submission("interaction-same", message="changed"))

        await queue.start()
        outcome = await queue.wait(first_receipt, timeout=2)
    finally:
        await queue.stop()

    assert same_receipt == first_receipt
    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    submit.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_interaction_id_different_scope_is_one_conflicting_work() -> None:
    """捕获 payload scope 把同一 interaction work 隐式拆成两个分区的缺陷。"""
    queue = InteractionSubmissionQueue(AsyncMock(return_value="topic-real"))
    first = _submission("interaction-shared")
    different_scope = replace(
        first,
        identity_scope=make_identity_scope(
            user_id="u1",
            agent_id="a1",
            workspace_id="isolation_workspace",
        ),
    )

    try:
        receipt = await queue.submit(first)
        with pytest.raises(ValueError, match="already belongs"):
            await queue.submit(different_scope)
    finally:
        await queue.stop()

    assert receipt.work_id == "interaction:interaction-shared"
    assert await queue.pending_count() == 1


def test_codec_rejects_flattened_identity_projection() -> None:
    """捕获 work payload 同时接受嵌套 scope 与平铺 workspace_id 的缺陷。"""
    codec = InteractionSubmissionCodec()
    encoded = codec.encode(_submission("interaction-tampered"))
    encoded["workspace_id"] = "isolation_workspace"
    codecs = WorkPayloadCodecRegistry()
    codecs.register(codec)

    with pytest.raises(WorkPayloadDecodeError):
        codecs.decode(
            codec.kind,
            codec.schema_version,
            encode_canonical_json(encoded),
        )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_codec_rejects_nested_noncanonical_payload(mutation: str) -> None:
    """捕获 InteractionPayload 嵌套字段被 Pydantic 静默忽略或补默认值的缺陷。"""
    codec = InteractionSubmissionCodec()
    encoded = codec.encode(_submission("interaction-nested-tamper"))
    if mutation == "extra":
        encoded["payload"]["workspace_id"] = "isolation_workspace"
    else:
        del encoded["payload"]["model_used"]
    codecs = WorkPayloadCodecRegistry()
    codecs.register(codec)

    with pytest.raises(WorkPayloadDecodeError):
        codecs.decode(
            codec.kind,
            codec.schema_version,
            encode_canonical_json(encoded),
        )
