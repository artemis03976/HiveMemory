"""Interaction Submission Queue 的契约测试。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from hivememory.components.work_queue import (
    QueuePolicy,
    WorkItem,
    WorkPayloadCodecRegistry,
    WorkPayloadDecodeError,
    WorkState,
    encode_canonical_json,
)
from hivememory.core.models import TurnEvent, WorkspaceAssetRef
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionCodec,
    InteractionSubmissionQueue,
    TransientInteractionSubmissionError,
    topic_ordering_key,
)
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import make_identity_scope


def _payload(message: str = "hello") -> InteractionPayload:
    return make_interaction_payload(message, f"answer:{message}")


def _submission(
    interaction_id: str,
    *,
    message: str | None = None,
    ordering_key: str = "conversation-1",
    payload: InteractionPayload | None = None,
    new_topic_title: str | None = None,
    new_topic_summary: str | None = None,
) -> InteractionSubmission:
    return InteractionSubmission(
        belong_to=make_identity_scope(user_id="u1", agent_id="a1").workspace_identity,
        from_actor=make_identity_scope(user_id="u1", agent_id="a1").actor_identity,
        interaction_id=interaction_id,
        payload=payload or _payload(message or interaction_id),
        requested_topic_id="NEW_TOPIC",
        ordering_key=ordering_key,
        origin="import_bus",
        correlation={"turn_id": interaction_id},
        new_topic_title=new_topic_title,
        new_topic_summary=new_topic_summary,
    )


@pytest.mark.asyncio
async def test_enqueue_uses_payload_snapshot_and_each_retry_gets_fresh_dto() -> None:
    attempts: list[InteractionPayload] = []
    attempt_scopes = []

    async def submit(
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
    ):
        attempts.append(payload)
        attempt_scopes.append(belong_to)
        if len(attempts) == 1:
            payload.model_used = "attempt-local-mutation"
            payload.turn_events.clear()
            raise TransientInteractionSubmissionError("temporary failure")
        return "topic-real"

    queue = InteractionSubmissionQueue(submit)
    original = _payload("original")
    original_events = list(original.turn_events)
    submission = _submission("interaction-1", payload=original)
    assert submission.payload is original

    try:
        await queue.start()
        receipt = await queue.submit(submission)

        # 入队后的调用方修改不得影响 canonical bytes 中的工作快照。
        original.model_used = "external-mutation"
        original.turn_events.clear()
        submission.payload.model_used = "submission-mutation"
        submission.payload.turn_events.clear()

        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    assert outcome.topic_id == "topic-real"
    assert len(attempts) == 2
    assert attempts[0] is not attempts[1]
    assert attempts[1].model_used == ""
    assert attempts[1].turn_events == original_events
    assert attempt_scopes == [submission.belong_to, submission.belong_to]
    assert attempt_scopes[0] is not attempt_scopes[1]


@pytest.mark.asyncio
async def test_same_ordering_key_keeps_fifo_during_retry() -> None:
    calls: list[str] = []
    first_attempt = 0

    async def submit(
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
    ):
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

    async def submit(
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
    ):
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

    async def submit(
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
    ):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    queue = InteractionSubmissionQueue(submit, apply_timeout_seconds=0.01)

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
        belong_to=make_identity_scope(
            user_id="u1",
            agent_id="a1",
            workspace_id="isolation_workspace",
        ).workspace_identity,
    )

    try:
        receipt = await queue.submit(first)
        with pytest.raises(ValueError, match="already belongs"):
            await queue.submit(different_scope)
    finally:
        await queue.stop()

    assert receipt.work_id == "interaction:interaction-shared"
    assert await queue.pending_count() == 1


def test_codec_roundtrip_preserves_split_identity_and_attachment_order() -> None:
    """编码或解码丢失发起者、归属、结局、附件顺序或新话题标题与摘要时必须失败。"""
    attachments = [
        WorkspaceAssetRef(asset_id="asset-a", token="ref-a"),
        WorkspaceAssetRef(asset_id="asset-b", token="ref-b"),
    ]
    submission = _submission(
        "interaction-attachments",
        payload=make_interaction_payload(
            "带附件的消息",
            outcome="cancelled",
            used_attachments=attachments,
            turn_events=[
                TurnEvent(kind="user_message", sequence=0, content="带附件的消息"),
                TurnEvent(
                    kind="tool_call",
                    sequence=1,
                    content="",
                    action_id="read",
                    verb="READ",
                    targets=("fact_a", "fact_b"),
                    status="success",
                    result="正文",
                ),
            ],
        ),
        new_topic_title="附件整理",
        new_topic_summary="整理两份附件",
    )
    codec = InteractionSubmissionCodec()

    decoded = codec.decode(codec.encode(submission))

    assert decoded.payload.outcome == "cancelled"
    assert decoded.payload.used_attachments == attachments
    assert decoded.belong_to == submission.belong_to
    assert decoded.from_actor == submission.from_actor
    assert (decoded.new_topic_title, decoded.new_topic_summary) == ("附件整理", "整理两份附件")


def test_codec_rejects_previous_schema_key_set() -> None:
    """缺少新话题标题与摘要的旧版本键集不能被当作当前版本静默补默认值。"""
    codec = InteractionSubmissionCodec()
    encoded = codec.encode(_submission("interaction-previous-schema"))
    del encoded["new_topic_title"]
    del encoded["new_topic_summary"]

    with pytest.raises(ValueError, match="schema mismatch"):
        codec.decode(encoded)


def test_new_topic_title_and_summary_require_new_topic_target() -> None:
    """已有话题的标题与摘要由话题持有，提交不能借新话题字段改写它们。"""
    with pytest.raises(ValueError, match="NEW_TOPIC"):
        replace(
            _submission("interaction-existing"),
            requested_topic_id="topic-1",
            new_topic_title="改写标题",
        )


@pytest.mark.asyncio
async def test_new_topic_title_and_summary_reach_apply() -> None:
    """handler 把提交携带的新话题标题与摘要原样交给感知应用。"""
    received: dict[str, object] = {}

    async def submit(payload, *, new_topic_title=None, new_topic_summary=None, **_kwargs):
        received.update(title=new_topic_title, summary=new_topic_summary)
        return "topic-new"

    queue = InteractionSubmissionQueue(submit)
    try:
        await queue.start()
        receipt = await queue.submit(
            _submission(
                "interaction-titled",
                new_topic_title="部署贪吃蛇",
                new_topic_summary="讨论部署步骤",
            )
        )
        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    assert outcome is not None
    assert outcome.state == WorkState.SUCCEEDED
    assert received == {"title": "部署贪吃蛇", "summary": "讨论部署步骤"}


@pytest.mark.asyncio
async def test_lookup_reads_owner_from_index_and_from_work_record() -> None:
    """记录投影返回同名提交的归属；旁路索引缺失时从工作记录的规范载荷解码。"""
    queue = InteractionSubmissionQueue(AsyncMock(return_value="topic-real"))
    indexed = _submission("interaction-indexed")
    record_only = _submission("interaction-record-only")
    codecs = WorkPayloadCodecRegistry()
    codecs.register(InteractionSubmissionCodec())

    try:
        await queue.submit(indexed)
        # 调用方在 store 已接纳、旁路索引尚未写入时被取消，只留下工作记录。
        await queue.runtime.enqueue(
            WorkItem(
                work_id="interaction:interaction-record-only",
                lane=InteractionSubmissionQueue.LANE,
                kind=InteractionSubmissionCodec.kind,
                schema_version=InteractionSubmissionCodec.schema_version,
                payload=codecs.encode(
                    InteractionSubmissionCodec.kind,
                    InteractionSubmissionCodec.schema_version,
                    record_only,
                ),
                ordering_key=record_only.ordering_key,
                correlation_id=record_only.interaction_id,
                idempotency_key=record_only.interaction_id,
            )
        )

        views = [
            await queue.lookup("interaction-indexed"),
            await queue.lookup("interaction-record-only"),
        ]
        missing = await queue.lookup("interaction-missing")
    finally:
        await queue.stop()

    assert [(view.belong_to, view.state, view.topic_id) for view in views] == [
        (indexed.belong_to, WorkState.QUEUED, None),
        (record_only.belong_to, WorkState.QUEUED, None),
    ]
    assert missing is None


@pytest.mark.asyncio
async def test_lookup_reports_real_topic_on_success_and_none_on_failure() -> None:
    """成功的交互给出真实落点（NEW_TOPIC 已解析）；应用失败的交互不给落点。"""

    async def apply(payload, *, interaction_id, **_kwargs):
        if interaction_id == "interaction-failing":
            raise RuntimeError("apply failed")
        return "topic-real"

    queue = InteractionSubmissionQueue(apply)
    try:
        await queue.start()
        succeeded = await queue.submit(_submission("interaction-ok"))
        failed = await queue.submit(_submission("interaction-failing"))
        await queue.wait(succeeded, timeout=2)
        await queue.wait(failed, timeout=2)
        views = [
            await queue.lookup("interaction-ok"),
            await queue.lookup("interaction-failing"),
        ]
    finally:
        await queue.stop()

    assert [(view.state, view.topic_id) for view in views] == [
        (WorkState.SUCCEEDED, "topic-real"),
        (WorkState.FAILED, None),
    ]


@pytest.mark.asyncio
async def test_failed_interaction_can_be_resubmitted_with_a_new_payload() -> None:
    """应用失败的交互以同一标识重新提交（载荷可以不同），成功后报告新落点；
    等待中或已成功的交互仍按原规则拒绝不同的载荷。"""
    attempts: list[str] = []

    async def apply(payload, *, target_topic_id, **_kwargs):
        attempts.append(target_topic_id)
        if len(attempts) == 1:
            raise RuntimeError("target topic evicted")
        return "topic-rerouted"

    queue = InteractionSubmissionQueue(apply)
    try:
        await queue.start()
        routed_to_old = replace(_submission("interaction-retry"), requested_topic_id="topic-old")
        first = await queue.submit(routed_to_old)
        await queue.wait(first, timeout=2)
        failed = await queue.lookup("interaction-retry")
        retry = await queue.submit(_submission("interaction-retry"))
        await queue.wait(retry, timeout=2)
        succeeded = await queue.lookup("interaction-retry")
        with pytest.raises(ValueError, match="another payload"):
            await queue.submit(routed_to_old)
    finally:
        await queue.stop()

    assert failed.state == WorkState.FAILED
    assert retry.work_id != first.work_id
    assert attempts == ["topic-old", "NEW_TOPIC"]
    assert (succeeded.state, succeeded.topic_id) == (WorkState.SUCCEEDED, "topic-rerouted")


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


# ========== 话题折叠：排在同一话题提交之间的独立工作项 ==========


def _topic_submission(interaction_id: str, message: str, topic_id: str) -> InteractionSubmission:
    """写入已有话题的提交：排序键按话题约定生成，与该话题的折叠项一致。"""
    return InteractionSubmission(
        belong_to=make_identity_scope(user_id="u1", agent_id="a1").workspace_identity,
        from_actor=make_identity_scope(user_id="u1", agent_id="a1").actor_identity,
        interaction_id=interaction_id,
        payload=_payload(message),
        requested_topic_id=topic_id,
        ordering_key=topic_ordering_key(topic_id),
        origin="import_bus",
        correlation={"turn_id": interaction_id},
    )


class _FakeTopics:
    """感知一侧的替身：写入使话题需要折叠，折叠完成后不再需要。

    ``on_compact`` 在折叠时执行，可阻塞或失败；失败的折叠不改变话题的状态。
    """

    def __init__(self, events: list[str], *, on_compact=None, fail_needs_check: bool = False):
        self._events = events
        self._on_compact = on_compact
        self._fail_needs_check = fail_needs_check
        self._over_threshold: set[str] = set()

    async def apply(self, payload, *, target_topic_id, **kwargs) -> str:
        self._events.append(f"apply:{payload.user_message}")
        self._over_threshold.add(target_topic_id)
        return target_topic_id

    def needs_compact(self, belong_to, topic_id) -> bool:
        if self._fail_needs_check:
            raise RuntimeError("topic store unavailable")
        return topic_id in self._over_threshold

    async def compact_topic(self, belong_to, topic_id) -> None:
        self._events.append(f"compact:{topic_id}")
        if self._on_compact is not None:
            await self._on_compact(topic_id)
        self._over_threshold.discard(topic_id)


def _events_until(events: list[str], last: str) -> list[str]:
    """截取到指定事件为止的序列；其后由新写入触发的折叠不影响断言。"""
    return events[: events.index(last) + 1]


@pytest.mark.asyncio
async def test_same_topic_submission_waits_for_the_compaction() -> None:
    """折叠执行期间，同话题的提交不开始；其他话题的提交照常执行。"""
    events: list[str] = []
    release = asyncio.Event()

    async def hold_t1(topic_id):
        if topic_id == "t1":
            await release.wait()

    topics = _FakeTopics(events, on_compact=hold_t1)
    queue = InteractionSubmissionQueue(topics.apply, compaction=topics)
    try:
        await queue.start()
        first = await queue.submit(_topic_submission("i-1", "first", "t1"))
        await queue.wait(first, timeout=2)
        second = await queue.submit(_topic_submission("i-2", "second", "t1"))
        other = await queue.submit(_topic_submission("i-3", "other", "t2"))
        other_outcome = await queue.wait(other, timeout=2)
        applied_while_compacting = list(events)
        release.set()
        second_outcome = await queue.wait(second, timeout=2)
    finally:
        release.set()
        await queue.stop()

    assert other_outcome.state is WorkState.SUCCEEDED
    assert "apply:second" not in applied_while_compacting
    assert second_outcome.state is WorkState.SUCCEEDED
    t1_events = [event for event in events if event not in {"apply:other", "compact:t2"}]
    assert _events_until(t1_events, "apply:second") == ["apply:first", "compact:t1", "apply:second"]


@pytest.mark.asyncio
async def test_failed_compaction_neither_fails_the_interaction_nor_holds_the_topic() -> None:
    events: list[str] = []

    async def relay_down(topic_id):
        raise RuntimeError("relay down")

    topics = _FakeTopics(events, on_compact=relay_down)
    queue = InteractionSubmissionQueue(topics.apply, compaction=topics)
    try:
        await queue.start()
        first = await queue.submit(_topic_submission("i-1", "first", "t1"))
        first_outcome = await queue.wait(first, timeout=2)
        second = await queue.submit(_topic_submission("i-2", "second", "t1"))
        second_outcome = await queue.wait(second, timeout=2)
    finally:
        await queue.stop()

    assert first_outcome.state is WorkState.SUCCEEDED
    assert second_outcome.state is WorkState.SUCCEEDED
    assert _events_until(events, "apply:second") == ["apply:first", "compact:t1", "apply:second"]


@pytest.mark.asyncio
async def test_topic_with_a_pending_compaction_is_not_scheduled_again() -> None:
    """折叠项排在已入队的同话题提交之后；那次提交不再追加第二个折叠项。"""
    events: list[str] = []
    topics = _FakeTopics(events)
    queue = InteractionSubmissionQueue(topics.apply, compaction=topics)
    try:
        await queue.submit(_topic_submission("i-1", "first", "t1"))
        second = await queue.submit(_topic_submission("i-2", "second", "t1"))
        await queue.start()
        await queue.wait(second, timeout=2)
        third = await queue.submit(_topic_submission("i-3", "third", "t1"))
        third_outcome = await queue.wait(third, timeout=2)
    finally:
        await queue.stop()

    assert third_outcome.state is WorkState.SUCCEEDED
    assert _events_until(events, "apply:third") == [
        "apply:first",
        "apply:second",
        "compact:t1",
        "apply:third",
    ]


@pytest.mark.asyncio
async def test_closing_compactions_cancels_the_running_one_and_releases_the_topic() -> None:
    events: list[str] = []
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_for_summary(topic_id):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    topics = _FakeTopics(events, on_compact=wait_for_summary)
    queue = InteractionSubmissionQueue(topics.apply, compaction=topics)
    try:
        await queue.start()
        first = await queue.submit(_topic_submission("i-1", "first", "t1"))
        await queue.wait(first, timeout=2)
        await asyncio.wait_for(started.wait(), timeout=2)
        second = await queue.submit(_topic_submission("i-2", "second", "t1"))

        queue.close_compactions()
        second_outcome = await queue.wait(second, timeout=2)
    finally:
        await queue.stop()

    assert cancelled.is_set()
    assert second_outcome.state is WorkState.SUCCEEDED
    assert events == ["apply:first", "compact:t1", "apply:second"]


@pytest.mark.asyncio
async def test_drain_all_returns_after_closed_compactions_finish_their_cleanup() -> None:
    """停止时先关闭折叠再排空：排空要等进行中的折叠收尾，之后的结算才不会遇到占用。"""
    events: list[str] = []
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def slow_cleanup_on_cancel(topic_id):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleaned_up.set()
            raise

    topics = _FakeTopics(events, on_compact=slow_cleanup_on_cancel)
    queue = InteractionSubmissionQueue(topics.apply, compaction=topics)
    try:
        await queue.start()
        first = await queue.submit(_topic_submission("i-1", "first", "t1"))
        await queue.wait(first, timeout=2)

        queue.close_compactions()
        drain = asyncio.create_task(queue.drain_all(timeout=None))
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        drained_before_cleanup = drain.done()
        allow_cleanup.set()
        drained = await asyncio.wait_for(drain, timeout=2)
    finally:
        allow_cleanup.set()
        await queue.stop()

    assert drained_before_cleanup is False
    assert cleaned_up.is_set()
    assert drained == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["queue_full", "needs_check_error"])
async def test_failure_while_scheduling_compaction_keeps_the_interaction_succeeded(
    failure: str,
) -> None:
    """交互已经写入：队列已满或判断出错时放弃这次折叠，交互结果仍是成功。"""
    events: list[str] = []
    topics = _FakeTopics(events, fail_needs_check=failure == "needs_check_error")
    policy = (
        QueuePolicy(capacity=1, max_concurrency=1, ordered_by_key=True, cancellable=False)
        if failure == "queue_full"
        else None
    )
    queue = InteractionSubmissionQueue(topics.apply, compaction=topics, policy=policy)
    try:
        await queue.start()
        first = await queue.submit(_topic_submission("i-1", "first", "t1"))
        first_outcome = await queue.wait(first, timeout=2)
        second = await queue.submit(_topic_submission("i-2", "second", "t1"))
        second_outcome = await queue.wait(second, timeout=2)
    finally:
        await queue.stop()

    assert first_outcome.state is WorkState.SUCCEEDED
    assert second_outcome.state is WorkState.SUCCEEDED
    assert events == ["apply:first", "apply:second"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"apply_timeout_seconds": 0}, "apply_timeout_seconds"),
        (
            {"compaction": _FakeTopics([]), "policy": QueuePolicy(capacity=4, max_concurrency=1)},
            "ordered by key",
        ),
    ],
)
def test_queue_rejects_configuration_that_breaks_its_guarantees(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        InteractionSubmissionQueue(AsyncMock(return_value="t1"), **kwargs)
