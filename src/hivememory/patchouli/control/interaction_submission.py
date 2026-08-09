"""Patchouli Interaction Submission 的通用队列适配。

这里的队列只保存 versioned codec 生成的 JSON bytes；Perception DTO 只在
handler 的单次 attempt 中重新构造。这样 passive seal 后，外部继续修改原始
``InteractionPayload`` 也不会影响已经接受的 work。
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from hivememory.core.protocol.models import InteractionPayload
from hivememory.infrastructure.work_queue import InMemoryWorkStore
from hivememory.system.runtime.events import RuntimeEventSink
from hivememory.system.runtime.work_queue import (
    FailureAction,
    FailureDecision,
    QueuePolicy,
    WorkExecutionContext,
    WorkHandlerPort,
    WorkItem,
    WorkPayloadCodecRegistry,
    WorkQueueRuntime,
    WorkRecord,
    WorkState,
)

InteractionOrigin = Literal["active_chat", "passive_memory"]


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True)
class InteractionSubmission:
    """进入 Patchouli 摄入队列的一次交互提交快照。"""

    interaction_id: str
    payload: InteractionPayload
    requested_topic_id: str
    ordering_key: str
    origin: InteractionOrigin
    correlation: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_text(self.interaction_id, field_name="interaction_id")
        _require_text(self.requested_topic_id, field_name="requested_topic_id")
        _require_text(self.ordering_key, field_name="ordering_key")
        if self.origin not in {"active_chat", "passive_memory"}:
            raise ValueError("origin must be active_chat or passive_memory")
        if not isinstance(self.correlation, Mapping):
            raise TypeError("correlation must be a mapping")

        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.correlation.items()
        ):
            raise TypeError("correlation keys and values must be strings")


class InteractionSubmissionCodec:
    """InteractionSubmission 的 v1 canonical JSON codec。"""

    kind = "patchouli.interaction_submission"
    schema_version = 1

    def encode(self, submission: InteractionSubmission) -> object:
        if not isinstance(submission, InteractionSubmission):
            raise TypeError("interaction submission payload has an unexpected type")
        return {
            "interaction_id": submission.interaction_id,
            "payload": submission.payload.model_dump(mode="json"),
            "requested_topic_id": submission.requested_topic_id,
            "ordering_key": submission.ordering_key,
            "origin": submission.origin,
            "correlation": dict(submission.correlation),
        }

    def decode(self, payload: object) -> InteractionSubmission:
        if not isinstance(payload, dict):
            raise TypeError("interaction submission payload must be an object")
        raw_payload = payload.get("payload")
        correlation = payload.get("correlation", {})
        if not isinstance(raw_payload, dict):
            raise TypeError("interaction submission payload.payload must be an object")
        if not isinstance(correlation, dict):
            raise TypeError("interaction submission payload.correlation must be an object")
        return InteractionSubmission(
            interaction_id=payload["interaction_id"],
            payload=InteractionPayload.model_validate(raw_payload),
            requested_topic_id=payload["requested_topic_id"],
            ordering_key=payload["ordering_key"],
            origin=payload["origin"],
            correlation=correlation,
        )


@dataclass(frozen=True)
class InteractionSubmissionResult:
    """handler 成功后的业务结果；``topic_id`` 是 NEW_TOPIC 的真实落点。"""

    interaction_id: str
    topic_id: str

    @property
    def result_ref(self) -> str:
        return self.topic_id


@dataclass(frozen=True)
class InteractionSubmissionReceipt:
    """队列接受 submission 后返回的领域收据。"""

    interaction_id: str
    work_id: str
    ordering_key: str
    state: WorkState
    enqueued_at: datetime


@dataclass(frozen=True)
class InteractionSubmissionOutcome:
    """查询 submission work 的终态投影。"""

    interaction_id: str
    work_id: str
    state: WorkState
    topic_id: str | None = None
    error_class: str | None = None


SubmitInteraction = Callable[..., Awaitable[str]]


class InteractionSubmissionHandler(
    WorkHandlerPort[InteractionSubmission, InteractionSubmissionResult]
):
    """把通用 work attempt 适配到 PerceptionFamiliar。"""

    def __init__(self, submit_interaction: SubmitInteraction) -> None:
        self._submit_interaction = submit_interaction

    async def execute(
        self,
        payload: InteractionSubmission,
        context: WorkExecutionContext,
    ) -> InteractionSubmissionResult:
        topic_id = await self._submit_interaction(
            payload.payload,
            target_topic_id=payload.requested_topic_id,
            interaction_id=payload.interaction_id,
        )
        return InteractionSubmissionResult(
            interaction_id=payload.interaction_id,
            topic_id=topic_id,
        )

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision:
        # 感知写入失败通常是暂时性存储/总线故障，交由 queue policy 截断重试次数。
        return FailureDecision(
            action=FailureAction.RETRY,
            retry_after_seconds=0.05,
            reason="interaction_submission_failed",
        )


@dataclass
class _StoredSubmission:
    receipt: InteractionSubmissionReceipt
    payload_bytes: bytes


class InteractionSubmissionQueue:
    """Patchouli 交互摄入 lane 的轻量业务队列。"""

    LANE = "patchouli.interaction_submission"

    def __init__(
        self,
        submit_interaction: SubmitInteraction,
        *,
        store: InMemoryWorkStore | None = None,
        runtime_events: RuntimeEventSink | None = None,
        policy: QueuePolicy | None = None,
    ) -> None:
        self._codecs = WorkPayloadCodecRegistry()
        self._codecs.register(InteractionSubmissionCodec())
        lane_policy = policy or QueuePolicy(
            capacity=256,
            max_concurrency=4,
            ordered_by_key=True,
            max_attempts=3,
            terminal_retention=512,
        )
        self._runtime = WorkQueueRuntime(
            store=store or InMemoryWorkStore(),
            payload_codecs=self._codecs,
            runtime_events=runtime_events,
            worker_poll_interval_seconds=0.02,
            shutdown_wait_seconds=2.0,
        )
        self._runtime.register_lane(
            self.LANE,
            handler=InteractionSubmissionHandler(submit_interaction),
            policy=lane_policy,
        )
        self._max_submission_entries = (
            lane_policy.capacity + lane_policy.terminal_retention
        )
        self._submissions: OrderedDict[str, _StoredSubmission] = OrderedDict()
        self._submit_lock = asyncio.Lock()

    @property
    def runtime(self) -> WorkQueueRuntime:
        return self._runtime

    @property
    def started(self) -> bool:
        return self._runtime.started

    async def start(self) -> None:
        await self._runtime.start()

    async def stop(self):
        return await self._runtime.stop()

    async def submit(self, submission: InteractionSubmission) -> InteractionSubmissionReceipt:
        """规范化并接受一次 submission；重复 interaction_id 只返回原收据。"""
        if not isinstance(submission, InteractionSubmission):
            raise TypeError("submission must be an InteractionSubmission")

        payload_bytes = self._codecs.encode(
            InteractionSubmissionCodec.kind,
            InteractionSubmissionCodec.schema_version,
            submission,
        )
        work_id = f"interaction:{submission.interaction_id}"
        item = WorkItem(
            work_id=work_id,
            lane=self.LANE,
            kind=InteractionSubmissionCodec.kind,
            schema_version=InteractionSubmissionCodec.schema_version,
            payload=payload_bytes,
            ordering_key=submission.ordering_key,
            correlation_id=submission.correlation.get("turn_id") or submission.interaction_id,
            idempotency_key=submission.interaction_id,
        )

        async with self._submit_lock:
            existing = self._submissions.get(submission.interaction_id)
            if existing is not None:
                if existing.payload_bytes != payload_bytes:
                    raise ValueError(
                        f"interaction_id '{submission.interaction_id}' already belongs to another payload"
                    )
                return existing.receipt

            receipt = await self._runtime.enqueue(item)
            domain_receipt = InteractionSubmissionReceipt(
                interaction_id=submission.interaction_id,
                work_id=receipt.work_id,
                ordering_key=submission.ordering_key,
                state=receipt.state,
                enqueued_at=receipt.enqueued_at,
            )
            self._submissions[submission.interaction_id] = _StoredSubmission(
                receipt=domain_receipt,
                payload_bytes=payload_bytes,
            )
            await self._trim_submissions_locked()
            return domain_receipt

    async def _trim_submissions_locked(self) -> None:
        """保留全部活跃 submission，只淘汰超出窗口的终态旁路索引。"""
        if len(self._submissions) <= self._max_submission_entries:
            return
        for interaction_id, stored in list(self._submissions.items()):
            if len(self._submissions) <= self._max_submission_entries:
                break
            record = await self._runtime.get(stored.receipt.work_id)
            if record is None:
                self._submissions.pop(interaction_id, None)

    async def wait(
        self,
        receipt_or_interaction_id: InteractionSubmissionReceipt | str,
        timeout: float | None = None,
    ) -> InteractionSubmissionOutcome | None:
        interaction_id = (
            receipt_or_interaction_id.interaction_id
            if isinstance(receipt_or_interaction_id, InteractionSubmissionReceipt)
            else receipt_or_interaction_id
        )
        stored = self._submissions.get(interaction_id)
        if stored is None:
            return None
        record = await self._runtime.wait(stored.receipt.work_id, timeout=timeout)
        return self._to_outcome(interaction_id, stored.receipt.work_id, record)

    async def drain(
        self,
        ordering_key: str | None = None,
        *,
        timeout: float | None = 5.0,
    ) -> int:
        """等待指定会话已知 work 进入终态，返回成功数量。"""
        entries = [
            stored
            for stored in self._submissions.values()
            if ordering_key is None or stored.receipt.ordering_key == ordering_key
        ]
        if not entries:
            return 0
        results = await asyncio.gather(
            *(self._runtime.wait(stored.receipt.work_id, timeout=timeout) for stored in entries)
        )
        return sum(
            1 for record in results if record is not None and record.state == WorkState.SUCCEEDED
        )

    async def drain_all(self, *, timeout: float | None = 5.0) -> int:
        return await self.drain(timeout=timeout)

    async def pending_count(self) -> int:
        pending = 0
        for stored in self._submissions.values():
            record = await self._runtime.get(stored.receipt.work_id)
            if record is not None and record.state not in {
                WorkState.SUCCEEDED,
                WorkState.FAILED,
                WorkState.DEAD_LETTER,
                WorkState.CANCELLED,
            }:
                pending += 1
        return pending

    @staticmethod
    def _to_outcome(
        interaction_id: str,
        work_id: str,
        record: WorkRecord | None,
    ) -> InteractionSubmissionOutcome | None:
        if record is None:
            return None
        return InteractionSubmissionOutcome(
            interaction_id=interaction_id,
            work_id=work_id,
            state=record.state,
            topic_id=record.result_ref if record.state == WorkState.SUCCEEDED else None,
            error_class=record.last_error.error_class if record.last_error else None,
        )


__all__ = [
    "InteractionOrigin",
    "InteractionSubmission",
    "InteractionSubmissionCodec",
    "InteractionSubmissionHandler",
    "InteractionSubmissionOutcome",
    "InteractionSubmissionQueue",
    "InteractionSubmissionReceipt",
    "InteractionSubmissionResult",
]
