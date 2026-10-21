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
        payload=InteractionPayload(
            user_message="带附件的消息", outcome="cancelled", used_attachments=attachments
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
