"""Patchouli Interaction Submission 的通用队列适配。

这里的队列只保存 versioned codec 生成的 JSON bytes；Perception DTO 只在
handler 的单次 attempt 中重新构造。这样提交方封口后，外部继续修改原始
``InteractionPayload`` 也不会影响已经接受的 work。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from hivememory.components.events.bus import RuntimeEventSink
from hivememory.components.work_queue import (
    TERMINAL_WORK_STATES,
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
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.core.protocol.models import InteractionPayload
from hivememory.infrastructure.work_queue import InMemoryWorkStore
from hivememory.patchouli.contracts.errors import TopicBusyError
from hivememory.patchouli.contracts.interaction_submission import (
    InteractionOrigin,
    InteractionSubmissionResult,
)


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_exact_keys(
    payload: dict[object, object],
    *,
    expected: set[str],
    field_name: str,
) -> None:
    """拒绝缺字段和冲突投影，确保 versioned payload 只有一份身份事实。"""
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(str(key) for key in actual - expected)
        raise ValueError(f"{field_name} schema mismatch: missing={missing}, extra={extra}")


@dataclass(frozen=True)
class InteractionSubmission:
    """进入 Patchouli 摄入队列的一次交互提交快照。

    ``belong_to`` 与 ``from_actor`` 分别承载归属与本轮发起者；
    ``payload`` 只承载内容与生成意图，不内嵌身份事实。
    ``new_topic_title``/``new_topic_summary`` 是提交方（Gateway 判定新建
    话题时）给出的新话题标题与摘要，只在 ``requested_topic_id`` 为
    ``NEW_TOPIC`` 时允许非空；缺省时感知以默认标题建立话题。
    """

    belong_to: WorkspaceIdentity
    from_actor: ActorIdentity
    interaction_id: str
    payload: InteractionPayload
    requested_topic_id: str
    ordering_key: str
    origin: InteractionOrigin
    correlation: Mapping[str, str]
    new_topic_title: str | None = None
    new_topic_summary: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.belong_to, WorkspaceIdentity):
            raise TypeError("belong_to must be a WorkspaceIdentity")
        if not isinstance(self.from_actor, ActorIdentity):
            raise TypeError("from_actor must be an ActorIdentity")
        _require_text(self.interaction_id, field_name="interaction_id")
        _require_text(self.requested_topic_id, field_name="requested_topic_id")
        _require_text(self.ordering_key, field_name="ordering_key")
        if self.origin not in {"active_chat", "import_bus", "workspace_port"}:
            raise ValueError("origin must be active_chat, import_bus or workspace_port")
        if not isinstance(self.correlation, Mapping):
            raise TypeError("correlation must be a mapping")
        for field_name in ("new_topic_title", "new_topic_summary"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        if self.requested_topic_id != "NEW_TOPIC" and (
            self.new_topic_title is not None or self.new_topic_summary is not None
        ):
            # 已有话题的标题与摘要由话题自身持有，提交不能借此改写它们。
            raise ValueError("new topic title and summary require requested_topic_id NEW_TOPIC")

        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.correlation.items()
        ):
            raise TypeError("correlation keys and values must be strings")


class InteractionSubmissionCodec:
    """InteractionSubmission 的 v6 canonical JSON codec，携带新话题的标题与摘要。"""

    kind = "patchouli.interaction_submission"
    schema_version = 6

    def encode(self, submission: InteractionSubmission) -> object:
        if not isinstance(submission, InteractionSubmission):
            raise TypeError("interaction submission payload has an unexpected type")
        return {
            "belong_to": submission.belong_to.model_dump(mode="json"),
            "from_actor": submission.from_actor.model_dump(mode="json"),
            "interaction_id": submission.interaction_id,
            "payload": submission.payload.model_dump(mode="json"),
            "requested_topic_id": submission.requested_topic_id,
            "ordering_key": submission.ordering_key,
            "origin": submission.origin,
            "correlation": dict(submission.correlation),
            "new_topic_title": submission.new_topic_title,
            "new_topic_summary": submission.new_topic_summary,
        }

    def decode(self, payload: object) -> InteractionSubmission:
        if not isinstance(payload, dict):
            raise TypeError("interaction submission payload must be an object")
        _require_exact_keys(
            payload,
            expected={
                "belong_to",
                "from_actor",
                "interaction_id",
                "payload",
                "requested_topic_id",
                "ordering_key",
                "origin",
                "correlation",
                "new_topic_title",
                "new_topic_summary",
            },
            field_name="interaction submission payload",
        )
        raw_payload = payload.get("payload")
        raw_scope = payload.get("belong_to")
        raw_actor = payload.get("from_actor")
        correlation = payload.get("correlation")
        if not isinstance(raw_payload, dict):
            raise TypeError("interaction submission payload.payload must be an object")
        if not isinstance(raw_scope, dict):
            raise TypeError("interaction submission payload.belong_to must be an object")
        if not isinstance(raw_actor, dict):
            raise TypeError("interaction submission payload.from_actor must be an object")
        if not isinstance(correlation, dict):
            raise TypeError("interaction submission payload.correlation must be an object")
        submission = InteractionSubmission(
            belong_to=WorkspaceIdentity.model_validate(raw_scope),
            from_actor=ActorIdentity.model_validate(raw_actor),
            interaction_id=payload["interaction_id"],
            payload=InteractionPayload.model_validate(raw_payload),
            requested_topic_id=payload["requested_topic_id"],
            ordering_key=payload["ordering_key"],
            origin=payload["origin"],
            correlation=correlation,
            new_topic_title=payload["new_topic_title"],
            new_topic_summary=payload["new_topic_summary"],
        )
        # 部分嵌套 Pydantic DTO 为兼容历史入口会忽略额外字段；codec 边界必须
        # 重新编码完整领域对象并要求规范等值，防止任何层级的篡改被静默吞掉。
        if self.encode(submission) != payload:
            raise ValueError("interaction submission payload is not canonical")
        return submission


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


ApplyInteraction = Callable[..., Awaitable[str]]


class TransientInteractionSubmissionError(RuntimeError):
    """显式标记可由同一 interaction identity 安全重试的瞬态失败。"""


class InteractionSubmissionHandler(
    WorkHandlerPort[InteractionSubmission, InteractionSubmissionResult]
):
    """把通用 work attempt 适配到 PerceptionFamiliar.apply_interaction。"""

    def __init__(self, apply_interaction: ApplyInteraction) -> None:
        self._apply_interaction = apply_interaction

    async def execute(
        self,
        payload: InteractionSubmission,
        context: WorkExecutionContext,
    ) -> InteractionSubmissionResult:
        # 只有实际进入 attachment_context 的 bound refs 才进入 binding
        # 投影（按 payload 顺序一次性投影）；selected 但被编译跳过的项不建立
        # binding，handler 也不按文件名或当前选择重新推导。
        asset_refs = tuple(payload.payload.used_attachments)
        topic_id = await self._apply_interaction(
            payload.payload,
            belong_to=payload.belong_to,
            from_actor=payload.from_actor,
            target_topic_id=payload.requested_topic_id,
            interaction_id=payload.interaction_id,
            asset_refs=asset_refs,
            new_topic_title=payload.new_topic_title,
            new_topic_summary=payload.new_topic_summary,
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
        """仅重试已明确分类、且可复用同一 interaction identity 的失败。"""

        if isinstance(
            error, (ConnectionError, TransientInteractionSubmissionError, TopicBusyError)
        ):
            return FailureDecision(
                action=FailureAction.RETRY,
                retry_after_seconds=0.05,
                reason="interaction_submission_transient_failure",
            )
        return FailureDecision(
            action=FailureAction.FAIL,
            reason="interaction_submission_failed",
        )


@dataclass
class _StoredSubmission:
    """旁路索引条目：只保存载荷摘要，冲突比较不需要留住整份载荷。"""

    receipt: InteractionSubmissionReceipt
    payload_digest: bytes
    belong_to: WorkspaceIdentity
    # 第几次提交：应用失败后以同一标识重新提交时递增，决定新工作项的标识。
    attempt: int = 0


@dataclass(frozen=True)
class InteractionRecordView:
    """提交队列对一个交互的记录投影：归属、工作状态与成功时的真实落点。

    ``state`` 为空表示旁路索引仍在、工作记录已超出终态保留窗口，接纳过但
    结局不再可知；``topic_id`` 只在成功时给出，``NEW_TOPIC`` 已解析为真实话题。
    """

    belong_to: WorkspaceIdentity
    state: WorkState | None
    topic_id: str | None


def _digest(payload_bytes: bytes) -> bytes:
    return hashlib.sha256(payload_bytes).digest()


def _ended_without_success(state: WorkState) -> bool:
    """工作项已到终态但没有成功应用：只有这种交互允许以同一标识重新提交。"""
    return state in TERMINAL_WORK_STATES and state is not WorkState.SUCCEEDED


class InteractionSubmissionQueue:
    """Patchouli 交互摄入 lane 的轻量业务队列。"""

    LANE = "patchouli.interaction_submission"

    def __init__(
        self,
        apply_interaction: ApplyInteraction,
        *,
        store: InMemoryWorkStore | None = None,
        runtime_events: RuntimeEventSink | None = None,
        policy: QueuePolicy | None = None,
    ) -> None:
        self._codecs = WorkPayloadCodecRegistry()
        # 队列只驻留于进程内，重启后无需保留旧载荷版本。
        self._codecs.register(InteractionSubmissionCodec())
        lane_policy = policy or QueuePolicy(
            capacity=256,
            max_concurrency=4,
            ordered_by_key=True,
            cancellable=False,
            timeout_seconds=30.0,
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
            handler=InteractionSubmissionHandler(apply_interaction),
            policy=lane_policy,
        )
        self._max_submission_entries = lane_policy.capacity + lane_policy.terminal_retention
        self._submissions: OrderedDict[str, _StoredSubmission] = OrderedDict()
        self._submit_lock = asyncio.Lock()
        self._stopped = asyncio.Event()

    @property
    def runtime(self) -> WorkQueueRuntime:
        return self._runtime

    @property
    def started(self) -> bool:
        return self._runtime.started

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    async def start(self) -> None:
        await self._runtime.start()

    async def stop(self):
        try:
            return await self._runtime.stop()
        finally:
            # 唤醒仍在等待 applied gate 的调用方，由其把非终态解释为运行时不可用。
            self._stopped.set()

    async def submit(self, submission: InteractionSubmission) -> InteractionSubmissionReceipt:
        """规范化并接受一次 submission；重复 interaction_id 只返回原收据。

        已应用失败的交互允许以同一标识重新提交，载荷可以不同（例如落位的话题
        已被驱逐，提交方重新路由后再交）：新的提交以递增的尝试号作为工作项
        标识入队。等待中与已成功的交互仍按原规则幂等，载荷不同时报冲突。
        """
        if not isinstance(submission, InteractionSubmission):
            raise TypeError("submission must be an InteractionSubmission")

        payload_bytes = self._codecs.encode(
            InteractionSubmissionCodec.kind,
            InteractionSubmissionCodec.schema_version,
            submission,
        )
        base_work_id = f"interaction:{submission.interaction_id}"

        async with self._submit_lock:
            existing = self._submissions.get(submission.interaction_id)
            if existing is not None:
                existing_state = await self._runtime.get(existing.receipt.work_id)
                if existing_state is not None and _ended_without_success(existing_state.state):
                    return await self._enqueue_locked(
                        submission, payload_bytes, attempt=existing.attempt + 1
                    )
                if existing.payload_digest != _digest(payload_bytes):
                    raise ValueError(
                        f"interaction_id '{submission.interaction_id}' already belongs to another payload"
                    )
                return existing.receipt

            # 调用方可能在 store 已接纳、receipt 尚未投影时被取消。稳定 work ID 允许
            # 后续 finalize 恢复该 admission，而不是再次创建 interaction。
            existing_record = await self._runtime.get(base_work_id)
            if existing_record is not None:
                if _ended_without_success(existing_record.state):
                    return await self._enqueue_locked(submission, payload_bytes, attempt=1)
                if existing_record.item.payload != payload_bytes:
                    raise ValueError(
                        f"interaction_id '{submission.interaction_id}' already belongs to another payload"
                    )
                domain_receipt = self._receipt_from_record(
                    submission,
                    existing_record,
                )
                self._submissions[submission.interaction_id] = _StoredSubmission(
                    receipt=domain_receipt,
                    payload_digest=_digest(payload_bytes),
                    belong_to=submission.belong_to,
                )
                await self._trim_submissions_locked()
                return domain_receipt

            return await self._enqueue_locked(submission, payload_bytes, attempt=0)

    async def _enqueue_locked(
        self, submission: InteractionSubmission, payload_bytes: bytes, *, attempt: int
    ) -> InteractionSubmissionReceipt:
        """入队一次提交并登记旁路索引；调用方持有提交锁。"""
        work_id = f"interaction:{submission.interaction_id}"
        if attempt > 0:
            work_id = f"{work_id}#{attempt}"
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
            payload_digest=_digest(payload_bytes),
            belong_to=submission.belong_to,
            attempt=attempt,
        )
        await self._trim_submissions_locked()
        return domain_receipt

    async def is_accepted(self, interaction_id: str) -> bool:
        """判断稳定 interaction ID 是否已经由 queue 接管。"""
        return await self.lookup(interaction_id) is not None

    async def lookup(self, interaction_id: str) -> InteractionRecordView | None:
        """返回已由 queue 接管的同名 submission 的记录投影；未接纳时返回 ``None``。

        归属优先取旁路索引；索引缺失、工作记录仍在保留窗口内时（例如调用方
        在 store 已接纳、索引尚未写入时被取消），从记录的规范载荷解码。两者
        都已裁剪的交互视为未接纳。
        """
        async with self._submit_lock:
            stored = self._submissions.get(interaction_id)
            work_id = (
                stored.receipt.work_id if stored is not None else f"interaction:{interaction_id}"
            )
            record = await self._runtime.get(work_id)
            if stored is not None:
                belong_to = stored.belong_to
            elif record is not None:
                submission: InteractionSubmission = self._codecs.decode(
                    record.item.kind, record.item.schema_version, record.item.payload
                )
                belong_to = submission.belong_to
            else:
                return None
            if record is None:
                return InteractionRecordView(belong_to=belong_to, state=None, topic_id=None)
            topic_id = record.result_ref if record.state is WorkState.SUCCEEDED else None
            return InteractionRecordView(belong_to=belong_to, state=record.state, topic_id=topic_id)

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
        wait_task = asyncio.create_task(self._runtime.wait(stored.receipt.work_id, timeout=timeout))
        stopped_task = asyncio.create_task(self._stopped.wait())
        try:
            done, _ = await asyncio.wait(
                {wait_task, stopped_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task in done:
                record = wait_task.result()
            else:
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
                record = await self._runtime.get(stored.receipt.work_id)
        finally:
            if not wait_task.done():
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
            stopped_task.cancel()
            await asyncio.gather(stopped_task, return_exceptions=True)
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

    @staticmethod
    def _receipt_from_record(
        submission: InteractionSubmission,
        record: WorkRecord,
    ) -> InteractionSubmissionReceipt:
        return InteractionSubmissionReceipt(
            interaction_id=submission.interaction_id,
            work_id=record.work_id,
            ordering_key=submission.ordering_key,
            state=record.state,
            enqueued_at=record.enqueued_at,
        )


__all__ = [
    "InteractionRecordView",
    "InteractionSubmission",
    "InteractionSubmissionCodec",
    "InteractionSubmissionHandler",
    "InteractionSubmissionOutcome",
    "InteractionSubmissionQueue",
    "InteractionSubmissionReceipt",
    "TransientInteractionSubmissionError",
]
