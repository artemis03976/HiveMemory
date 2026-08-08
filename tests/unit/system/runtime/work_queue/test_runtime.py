"""WorkQueueRuntime 的 Q1 执行与隔离测试。"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import pytest

from hivememory.infrastructure.work_queue import InMemoryWorkStore
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.work_queue import (
    FailureAction,
    FailureDecision,
    QueuePolicy,
    UnknownWorkPayloadCodecError,
    UnsupportedWorkQueueFeatureError,
    WorkExecutionContext,
    WorkItem,
    WorkPayloadCodecRegistry,
    WorkQueueCapacityError,
    WorkQueueRuntime,
    WorkState,
    encode_canonical_json,
)

_TEST_WORK_KIND = "test.work.v1"
_TEST_SCHEMA_VERSION = 1


class _TestPayloadCodec:
    kind = _TEST_WORK_KIND
    schema_version = _TEST_SCHEMA_VERSION

    def encode(self, payload: Any) -> object:
        return payload

    def decode(self, payload: Any) -> Any:
        return payload


_TEST_PAYLOAD_CODECS = WorkPayloadCodecRegistry()
_TEST_PAYLOAD_CODECS.register(_TestPayloadCodec())


def _runtime(**kwargs: Any) -> WorkQueueRuntime:
    return WorkQueueRuntime(
        store=InMemoryWorkStore(),
        payload_codecs=_TEST_PAYLOAD_CODECS,
        **kwargs,
    )


def _item(
    work_id: str,
    *,
    lane: str,
    payload: Any | None = None,
    key: str | None = None,
) -> WorkItem:
    return WorkItem(
        work_id=work_id,
        lane=lane,
        kind=_TEST_WORK_KIND,
        schema_version=_TEST_SCHEMA_VERSION,
        payload=_TEST_PAYLOAD_CODECS.encode(
            _TEST_WORK_KIND,
            _TEST_SCHEMA_VERSION,
            work_id if payload is None else payload,
        ),
        ordering_key=key,
    )


class _ImmediateHandler:
    async def execute(self, payload: Any, context: WorkExecutionContext) -> str | None:
        return f"result:{payload}"

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision:
        return FailureDecision(action=FailureAction.FAIL, reason="non_retryable")


class _NoResultHandler(_ImmediateHandler):
    async def execute(self, payload: Any, context: WorkExecutionContext) -> None:
        return None


class _ControlledHandler(_ImmediateHandler):
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.releases: dict[str, asyncio.Event] = {}

    async def execute(self, payload: str, context: WorkExecutionContext) -> str:
        self.releases.setdefault(payload, asyncio.Event())
        await self.started.put(payload)
        await self.releases[payload].wait()
        return f"result:{payload}"

    def release(self, payload: str) -> None:
        self.releases[payload].set()


class _RetryOnceHandler(_ImmediateHandler):
    def __init__(self) -> None:
        self.attempts: Counter[str] = Counter()
        self.execution_order: list[str] = []

    async def execute(self, payload: str, context: WorkExecutionContext) -> None:
        self.attempts[payload] += 1
        self.execution_order.append(payload)
        if payload == "first" and self.attempts[payload] == 1:
            raise RuntimeError("transient detail must not be recorded")

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision:
        return FailureDecision(
            action=FailureAction.RETRY,
            retry_after_seconds=0,
            reason="transient",
        )


@pytest.mark.asyncio
async def test_different_lanes_do_not_block_each_other() -> None:
    slow = _ControlledHandler()
    runtime = _runtime()
    runtime.register_lane(
        "slow",
        handler=slow,
        policy=QueuePolicy(capacity=4, max_concurrency=1),
    )
    runtime.register_lane(
        "fast",
        handler=_ImmediateHandler(),
        policy=QueuePolicy(capacity=4, max_concurrency=1),
    )
    await runtime.enqueue(_item("slow-1", lane="slow"))
    await runtime.enqueue(_item("slow-2", lane="slow"))
    await runtime.enqueue(_item("fast-1", lane="fast"))
    await runtime.start()

    assert await asyncio.wait_for(slow.started.get(), timeout=1) == "slow-1"
    fast = await runtime.wait("fast-1", timeout=1)
    assert fast is not None and fast.state == WorkState.SUCCEEDED
    assert (await runtime.get("slow-2")).state == WorkState.QUEUED  # type: ignore[union-attr]

    slow.release("slow-1")
    assert await asyncio.wait_for(slow.started.get(), timeout=1) == "slow-2"
    slow.release("slow-2")
    await runtime.stop()


@pytest.mark.asyncio
async def test_same_key_is_fifo_while_different_keys_run_concurrently() -> None:
    handler = _ControlledHandler()
    runtime = _runtime()
    runtime.register_lane(
        "ordered",
        handler=handler,
        policy=QueuePolicy(
            capacity=4,
            max_concurrency=2,
            ordered_by_key=True,
        ),
    )
    await runtime.enqueue(_item("first", lane="ordered", key="topic-1"))
    await runtime.enqueue(_item("second", lane="ordered", key="topic-1"))
    await runtime.enqueue(_item("other", lane="ordered", key="topic-2"))
    await runtime.start()

    first_wave = {
        await asyncio.wait_for(handler.started.get(), timeout=1),
        await asyncio.wait_for(handler.started.get(), timeout=1),
    }
    assert first_wave == {"first", "other"}

    handler.release("first")
    assert await asyncio.wait_for(handler.started.get(), timeout=1) == "second"
    handler.release("second")
    handler.release("other")
    await runtime.stop()


@pytest.mark.asyncio
async def test_retry_preserves_work_id_and_same_key_order() -> None:
    handler = _RetryOnceHandler()
    runtime = _runtime(worker_poll_interval_seconds=0.01)
    runtime.register_lane(
        "ordered",
        handler=handler,
        policy=QueuePolicy(
            capacity=4,
            max_concurrency=2,
            ordered_by_key=True,
            max_attempts=2,
        ),
    )
    await runtime.enqueue(_item("first", lane="ordered", payload="first", key="topic-1"))
    await runtime.enqueue(_item("second", lane="ordered", payload="second", key="topic-1"))
    await runtime.start()

    second = await runtime.wait("second", timeout=1)

    assert second is not None and second.state == WorkState.SUCCEEDED
    assert handler.execution_order == ["first", "first", "second"]
    assert (await runtime.get("first")).attempt_count == 2  # type: ignore[union-attr]
    await runtime.stop()


@pytest.mark.asyncio
async def test_retry_decodes_a_fresh_payload_snapshot_for_each_attempt() -> None:
    class MutatingRetryHandler(_ImmediateHandler):
        def __init__(self) -> None:
            self.observed_events: list[list[str]] = []

        async def execute(self, payload: Any, context: WorkExecutionContext) -> None:
            events = payload["events"]
            self.observed_events.append(list(events))
            events.append("handler-changed")
            if context.attempt_count == 1:
                raise RuntimeError("retry once")

        def classify_failure(
            self,
            error: Exception,
            context: WorkExecutionContext,
        ) -> FailureDecision:
            return FailureDecision(
                action=FailureAction.RETRY,
                retry_after_seconds=0,
                reason="transient",
            )

    handler = MutatingRetryHandler()
    runtime = _runtime(worker_poll_interval_seconds=0.01)
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(capacity=1, max_concurrency=1, max_attempts=2),
    )
    source_payload = {"events": ["created"]}
    await runtime.enqueue(_item("work-1", lane="lane", payload=source_payload))
    source_payload["events"].append("source-changed")
    await runtime.start()

    terminal = await runtime.wait("work-1", timeout=1)

    assert terminal is not None and terminal.state == WorkState.SUCCEEDED
    assert handler.observed_events == [["created"], ["created"]]
    await runtime.stop()


@pytest.mark.asyncio
async def test_retry_exhaustion_moves_work_to_dead_letter() -> None:
    class AlwaysRetry(_RetryOnceHandler):
        async def execute(self, payload: str, context: WorkExecutionContext) -> None:
            self.attempts[payload] += 1
            raise RuntimeError("transient")

    handler = AlwaysRetry()
    runtime = _runtime(worker_poll_interval_seconds=0.01)
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(capacity=2, max_concurrency=1, max_attempts=2),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))
    await runtime.start()

    terminal = await runtime.wait("work-1", timeout=1)

    assert terminal is not None and terminal.state == WorkState.DEAD_LETTER
    assert terminal.attempt_count == 2
    await runtime.stop()


@pytest.mark.asyncio
async def test_non_retryable_failure_is_failed_instead_of_dead_lettered() -> None:
    class FailingHandler(_ImmediateHandler):
        async def execute(self, payload: Any, context: WorkExecutionContext) -> str | None:
            if payload == "failed":
                raise ValueError("raw failure detail")
            return await super().execute(payload, context)

    runtime = _runtime()
    runtime.register_lane(
        "lane",
        handler=FailingHandler(),
        policy=QueuePolicy(capacity=2, max_concurrency=1),
    )
    await runtime.enqueue(_item("failed", lane="lane"))
    await runtime.enqueue(_item("next", lane="lane"))
    await runtime.start()

    failed = await runtime.wait("failed", timeout=1)
    next_record = await runtime.wait("next", timeout=1)

    assert failed is not None and failed.state == WorkState.FAILED
    assert failed.last_error is not None
    assert failed.last_error.error_class == "ValueError"
    assert failed.last_error.message == "non_retryable"
    assert next_record is not None and next_record.state == WorkState.SUCCEEDED
    await runtime.stop()


@pytest.mark.asyncio
async def test_timeout_is_classified_by_handler() -> None:
    class TimeoutHandler(_ImmediateHandler):
        async def execute(self, payload: Any, context: WorkExecutionContext) -> None:
            await asyncio.sleep(10)

    runtime = _runtime()
    runtime.register_lane(
        "lane",
        handler=TimeoutHandler(),
        policy=QueuePolicy(
            capacity=1,
            max_concurrency=1,
            timeout_seconds=0.01,
        ),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))
    await runtime.start()

    terminal = await runtime.wait("work-1", timeout=1)

    assert terminal is not None and terminal.state == WorkState.FAILED
    assert terminal.last_error is not None
    assert terminal.last_error.error_class == "TimeoutError"
    await runtime.stop()


@pytest.mark.asyncio
async def test_capacity_rejection_is_explicit_and_observable() -> None:
    sink = RecordingRuntimeEventSink()
    runtime = _runtime(runtime_events=sink)
    runtime.register_lane(
        "lane",
        handler=_ImmediateHandler(),
        policy=QueuePolicy(capacity=1, max_concurrency=1),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))

    with pytest.raises(WorkQueueCapacityError):
        await runtime.enqueue(_item("work-2", lane="lane"))

    assert (await runtime.get("work-1")).state == WorkState.QUEUED  # type: ignore[union-attr]
    assert sink.events[-1].event_type == RuntimeEventType.WORK_REJECTED


@pytest.mark.asyncio
async def test_enqueue_rejects_unknown_payload_codec_before_store_acceptance() -> None:
    sink = RecordingRuntimeEventSink()
    runtime = _runtime(runtime_events=sink)
    runtime.register_lane(
        "lane",
        handler=_ImmediateHandler(),
        policy=QueuePolicy(capacity=1, max_concurrency=1),
    )
    item = WorkItem(
        work_id="work-1",
        lane="lane",
        kind="unknown.work",
        schema_version=1,
        payload=encode_canonical_json({"value": "safe"}),
    )

    with pytest.raises(UnknownWorkPayloadCodecError):
        await runtime.enqueue(item)

    assert await runtime.get("work-1") is None
    assert sink.events[-1].event_type == RuntimeEventType.WORK_REJECTED
    assert sink.events[-1].reason == "unknown_payload_codec"


@pytest.mark.asyncio
async def test_payload_decode_failure_is_dead_lettered_without_calling_handler() -> None:
    class BrokenDecodeCodec:
        kind = "broken.work"
        schema_version = 1

        def encode(self, payload: Any) -> object:
            return payload

        def decode(self, payload: Any) -> Any:
            raise ValueError("invalid business schema")

    class CountingHandler(_ImmediateHandler):
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, payload: Any, context: WorkExecutionContext) -> None:
            self.calls += 1

    codecs = WorkPayloadCodecRegistry()
    codecs.register(BrokenDecodeCodec())
    handler = CountingHandler()
    runtime = WorkQueueRuntime(store=InMemoryWorkStore(), payload_codecs=codecs)
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(capacity=1, max_concurrency=1),
    )
    item = WorkItem(
        work_id="work-1",
        lane="lane",
        kind="broken.work",
        schema_version=1,
        payload=codecs.encode("broken.work", 1, {"value": "invalid"}),
    )
    await runtime.enqueue(item)
    await runtime.start()

    terminal = await runtime.wait("work-1", timeout=1)

    assert terminal is not None and terminal.state == WorkState.DEAD_LETTER
    assert terminal.last_error is not None
    assert terminal.last_error.error_class == "WorkPayloadDecodeError"
    assert terminal.last_error.message == "invalid_work_payload"
    assert handler.calls == 0
    await runtime.stop()


@pytest.mark.asyncio
async def test_running_cancellation_is_idempotent_and_releases_ordering_key() -> None:
    handler = _ControlledHandler()
    sink = RecordingRuntimeEventSink()
    runtime = _runtime(runtime_events=sink)
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(
            capacity=3,
            max_concurrency=1,
            ordered_by_key=True,
            cancellable=True,
        ),
    )
    await runtime.enqueue(_item("first", lane="lane", key="topic-1"))
    await runtime.enqueue(_item("second", lane="lane", key="topic-1"))
    await runtime.start()
    assert await asyncio.wait_for(handler.started.get(), timeout=1) == "first"

    assert await runtime.cancel("first")
    assert await runtime.cancel("first")
    assert await asyncio.wait_for(handler.started.get(), timeout=1) == "second"
    handler.release("second")
    cancelled = await runtime.wait("first", timeout=1)

    assert cancelled is not None and cancelled.state == WorkState.CANCELLED
    event_types = [event.event_type for event in sink.events if event.task_id == "first"]
    assert event_types.count(RuntimeEventType.WORK_CANCEL_REQUESTED) == 1
    assert event_types.count(RuntimeEventType.WORK_CANCELLED) == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_queued_cancellation_never_invokes_handler() -> None:
    handler = _ControlledHandler()
    runtime = _runtime()
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(capacity=1, max_concurrency=1, cancellable=True),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))

    assert await runtime.cancel("work-1")
    await runtime.start()
    await asyncio.sleep(0.02)
    record = await runtime.get("work-1")

    assert record is not None and record.state == WorkState.CANCELLED
    assert handler.started.empty()
    await runtime.stop()


@pytest.mark.asyncio
async def test_shutdown_summary_reports_in_memory_loss_risk_per_lane() -> None:
    handler = _ControlledHandler()
    runtime = _runtime(shutdown_wait_seconds=0)
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(
            capacity=3,
            max_concurrency=1,
            cancellable=False,
        ),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))
    await runtime.enqueue(_item("work-2", lane="lane"))
    await runtime.enqueue(_item("work-3", lane="lane"))
    await runtime.start()
    assert await asyncio.wait_for(handler.started.get(), timeout=1) == "work-1"

    summary = await runtime.stop()

    assert summary.pending == 2
    assert summary.running == 1
    assert summary.in_memory_loss_risk == 3
    assert summary.lanes[0].drain_timed_out
    repeated = await runtime.stop()
    assert repeated.already_stopped
    handler.release("work-1")
    await runtime.wait("work-1", timeout=1)


@pytest.mark.asyncio
async def test_shutdown_cancels_running_work_when_lane_policy_allows_it() -> None:
    handler = _ControlledHandler()
    runtime = _runtime(shutdown_wait_seconds=0)
    runtime.register_lane(
        "lane",
        handler=handler,
        policy=QueuePolicy(
            capacity=1,
            max_concurrency=1,
            cancellable=True,
        ),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))
    await runtime.start()
    assert await asyncio.wait_for(handler.started.get(), timeout=1) == "work-1"

    summary = await runtime.stop()
    terminal = await runtime.get("work-1")

    assert summary.running == 0
    assert summary.cancelled_during_shutdown == 1
    assert summary.in_memory_loss_risk == 0
    assert terminal is not None and terminal.state == WorkState.CANCELLED


@pytest.mark.asyncio
async def test_work_events_emit_safe_identifiers_without_business_payload() -> None:
    sink = RecordingRuntimeEventSink()
    runtime = _runtime(runtime_events=sink)
    runtime.register_lane(
        "lane",
        handler=_NoResultHandler(),
        policy=QueuePolicy(capacity=1, max_concurrency=1, ordered_by_key=True),
    )
    item = WorkItem(
        work_id="work-1",
        lane="lane",
        kind="test.work.v1",
        schema_version=1,
        payload=_TEST_PAYLOAD_CODECS.encode(
            _TEST_WORK_KIND,
            _TEST_SCHEMA_VERSION,
            {"private_text": "payload-secret"},
        ),
        ordering_key="ordering-secret",
        correlation_id="correlation-secret",
        idempotency_key="idempotency-secret",
    )
    await runtime.enqueue(item)
    await runtime.start()
    await runtime.wait("work-1", timeout=1)

    event_types = [event.event_type for event in sink.events]
    serialized_data = repr([event.data for event in sink.events])

    assert event_types == [
        RuntimeEventType.WORK_QUEUED,
        RuntimeEventType.WORK_STARTED,
        RuntimeEventType.WORK_SUCCEEDED,
    ]
    assert "payload-secret" not in serialized_data
    assert "ordering-secret" not in serialized_data
    assert "correlation-secret" not in serialized_data
    assert "idempotency-secret" not in serialized_data
    assert sink.events[0].data["ordering_key_digest"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_event_failure_does_not_change_work_result() -> None:
    class FailingSink:
        def emit(self, event) -> None:
            raise RuntimeError("sink unavailable")

        def scoped(self, *_args, **_kwargs):
            return self

    runtime = _runtime(runtime_events=FailingSink())
    runtime.register_lane(
        "lane",
        handler=_ImmediateHandler(),
        policy=QueuePolicy(capacity=1, max_concurrency=1),
    )
    await runtime.enqueue(_item("work-1", lane="lane"))
    await runtime.start()

    terminal = await runtime.wait("work-1", timeout=1)

    assert terminal is not None and terminal.state == WorkState.SUCCEEDED
    await runtime.stop()


def test_q1_rejects_priority_instead_of_silently_ignoring_it() -> None:
    runtime = _runtime()

    with pytest.raises(UnsupportedWorkQueueFeatureError):
        runtime.register_lane(
            "priority",
            handler=_ImmediateHandler(),
            policy=QueuePolicy(
                capacity=1,
                max_concurrency=1,
                priority_enabled=True,
            ),
        )
