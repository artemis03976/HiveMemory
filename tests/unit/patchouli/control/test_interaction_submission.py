"""Interaction Submission Queue 的 Q2 契约测试。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
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
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.system.runtime.work_queue import (
    QueuePolicy,
    WorkPayloadCodecRegistry,
    WorkPayloadDecodeError,
    WorkState,
    encode_canonical_json,
)
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


def _build_familiar(
    store: ShortTermMemoryStore,
    interaction_journal: InMemoryInteractionApplyJournal,
    relay,
    bus,
    *,
    max_resident_topics: int = 5,
    engine="default",
) -> PerceptionFamiliar:
    """按当前装配形态构造 Familiar：Engine + Store + WorkingSet 编排。

    ``engine="default"`` 构造默认阈值引擎；显式传 ``None`` 表示感知关闭。
    """
    if engine == "default":
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=999999)
        )
    return PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=TopicWorkingSet(max_resident=max_resident_topics),
        relay_controller=relay,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=30),
        interaction_journal=interaction_journal,
    )


@pytest.mark.asyncio
async def test_ambiguous_failure_after_add_block_does_not_duplicate_block() -> None:
    store = ShortTermMemoryStore()
    interaction_journal = InMemoryInteractionApplyJournal()
    relay = Mock()
    relay.should_relay.return_value = None
    familiar = _build_familiar(
        store, interaction_journal, relay, Mock(request=AsyncMock(return_value=None))
    )
    original_fold = familiar._compact_topic_if_needed
    fold_calls = 0

    async def fail_once_after_add(identity_scope, topic_id):
        nonlocal fold_calls
        fold_calls += 1
        if fold_calls == 1:
            raise TransientInteractionSubmissionError("caller missed apply result")
        return await original_fold(identity_scope, topic_id)

    familiar._compact_topic_if_needed = fail_once_after_add
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
    topic = store.get(identity_scope, outcome.topic_id)
    assert topic is not None
    assert topic.block_count == 1
    scope_key = (
        identity_scope.workspace_identity.owner_user_id,
        identity_scope.workspace_identity.workspace_id,
    )
    # 行为断言：缺省 manual settle 命中的正是本次写入的最近活跃话题
    settle_result = await familiar.manual_settle_topic(identity_scope)
    assert settle_result.topic_id == outcome.topic_id
    assert store.get(identity_scope, outcome.topic_id) is None

    # retry 已全部完成（COMPLETED），幂等返回同一话题且不重复写块
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
async def test_retry_resumes_pending_compact_without_duplicating_block() -> None:
    """INTERACTION_APPLIED 阶段的 retry 补跑本地 compact 义务，不重复写块。"""
    store = ShortTermMemoryStore()
    interaction_journal = InMemoryInteractionApplyJournal()
    relay = Mock()
    relay.generate_summary.side_effect = [
        TransientInteractionSubmissionError("relay missed"),
        "folded-summary",
    ]
    engine = MemoryPerceptionEngine(
        config=SemanticFlowPerceptionConfig(fold_token_threshold=1, fold_retain_recent_blocks=1)
    )
    familiar = _build_familiar(
        store,
        interaction_journal,
        relay,
        Mock(request=AsyncMock(return_value=None)),
        engine=engine,
    )
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _payload("first"), identity_scope=identity_scope, interaction_id="i-first"
    )
    queue = InteractionSubmissionQueue(familiar.apply_interaction)
    submission = InteractionSubmission(
        identity_scope=identity_scope,
        interaction_id="interaction-settlement",
        payload=_payload("second"),
        requested_topic_id=topic_id,
        ordering_key="conversation-1",
        origin="passive_memory",
        correlation={"turn_id": "interaction-settlement"},
    )

    try:
        await queue.start()
        receipt = await queue.submit(submission)
        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    # 首次 attempt 写入 block 后 compact 失败（relay 抛瞬态错误）；retry 补跑
    # compact 完成，不重复写块。
    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    topic = store.get(identity_scope, topic_id)
    assert topic is not None
    assert topic.block_count == 1  # 折叠保留最近 1 块
    assert topic.state_summary == "folded-summary"
    record = interaction_journal.get("interaction-settlement")
    assert record is not None
    assert record.stage is InteractionApplyStage.COMPLETED
    assert record.settlement_to_submit is None
    # 首次 attempt 失败一次 + retry 补跑成功一次
    assert relay.generate_summary.call_count == 2


@pytest.mark.asyncio
async def test_disabled_perception_does_not_require_apply_journal_entry() -> None:
    store = ShortTermMemoryStore()
    interaction_journal = InMemoryInteractionApplyJournal()
    familiar = _build_familiar(
        store,
        interaction_journal,
        relay=Mock(),
        bus=Mock(request=AsyncMock()),
        engine=None,  # 感知关闭（原 NullPerceptionLayer 语义）
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
