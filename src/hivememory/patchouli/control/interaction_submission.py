"""Patchouli Interaction Submission 的通用队列适配。

这里的队列只保存 versioned codec 生成的 JSON bytes；Perception DTO 只在
handler 的单次 attempt 中重新构造。这样提交方封口后，外部继续修改原始
``InteractionPayload`` 也不会影响已经接受的 work。

话题折叠以独立工作项排在同一条 lane 上，与该话题的提交共用排序键：交互写入
话题即完成，折叠的耗时不计入交互的处理时限；折叠执行期间，同一话题的后续
提交在队列中等待，等待同样不计时。
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

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
    WorkQueueCapacityError,
    WorkQueueRuntime,
    WorkQueueStoppedError,
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

logger = logging.getLogger(__name__)


def topic_ordering_key(topic_id: str) -> str:
    """同一话题的全部工作项共用的排序键；lane 按它串行执行同一话题的提交与折叠。"""
    return f"topic:{topic_id}"


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


class TopicCompactionPort(Protocol):
    """提交队列对话题折叠的需要：判断话题是否需要折叠，以及执行一次折叠。"""

    def needs_compact(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool: ...

    async def compact_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> None: ...


@dataclass(frozen=True)
class TopicCompaction:
    """排在话题提交之间的一次折叠；执行期间挡住同一话题的后续提交。"""

    belong_to: WorkspaceIdentity
    topic_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.belong_to, WorkspaceIdentity):
            raise TypeError("belong_to must be a WorkspaceIdentity")
        _require_text(self.topic_id, field_name="topic_id")


class TopicCompactionCodec:
    """TopicCompaction 的 v1 canonical JSON codec。"""

    kind = "patchouli.topic_compaction"
    schema_version = 1

    def encode(self, compaction: TopicCompaction) -> object:
        if not isinstance(compaction, TopicCompaction):
            raise TypeError("topic compaction payload has an unexpected type")
        return {
            "belong_to": compaction.belong_to.model_dump(mode="json"),
            "topic_id": compaction.topic_id,
        }

    def decode(self, payload: object) -> TopicCompaction:
        if not isinstance(payload, dict):
            raise TypeError("topic compaction payload must be an object")
        _require_exact_keys(
            payload,
            expected={"belong_to", "topic_id"},
            field_name="topic compaction payload",
        )
        raw_scope = payload["belong_to"]
        if not isinstance(raw_scope, dict):
            raise TypeError("topic compaction payload.belong_to must be an object")
        compaction = TopicCompaction(
            belong_to=WorkspaceIdentity.model_validate(raw_scope),
            topic_id=payload["topic_id"],
        )
        if self.encode(compaction) != payload:
            raise ValueError("topic compaction payload is not canonical")
        return compaction


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
AfterApply = Callable[[WorkspaceIdentity, str], Awaitable[None]]
RunCompaction = Callable[[TopicCompaction], Awaitable[None]]


class TransientInteractionSubmissionError(RuntimeError):
    """显式标记可由同一 interaction identity 安全重试的瞬态失败。"""


class InteractionSubmissionHandler(
    WorkHandlerPort[InteractionSubmission | TopicCompaction, InteractionSubmissionResult | None]
):
    """把通用 work attempt 适配到交互应用与话题折叠。

    lane 只有一个 handler，按载荷类型分派。交互应用的时限只包住应用本身；
    应用完成后由 ``after_apply`` 决定是否排入折叠，折叠的耗时不计入这次应用。
    """

    def __init__(
        self,
        apply_interaction: ApplyInteraction,
        *,
        apply_timeout_seconds: float | None = None,
        after_apply: AfterApply | None = None,
        run_compaction: RunCompaction | None = None,
    ) -> None:
        self._apply_interaction = apply_interaction
        self._apply_timeout_seconds = apply_timeout_seconds
        self._after_apply = after_apply
        self._run_compaction = run_compaction

    async def execute(
        self,
        payload: InteractionSubmission | TopicCompaction,
        context: WorkExecutionContext,
    ) -> InteractionSubmissionResult | None:
        if isinstance(payload, TopicCompaction):
            if self._run_compaction is None:
                raise RuntimeError("topic compaction work requires a compaction port")
            await self._run_compaction(payload)
            return None
        # 只有实际进入 attachment_context 的 bound refs 才进入 binding
        # 投影（按 payload 顺序一次性投影）；selected 但被编译跳过的项不建立
        # binding，handler 也不按文件名或当前选择重新推导。
        asset_refs = tuple(payload.payload.used_attachments)
        async with asyncio.timeout(self._apply_timeout_seconds):
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
        if self._after_apply is not None:
            try:
                await self._after_apply(payload.belong_to, topic_id)
            except Exception:
                # 交互已经写入，这次应用的结果是成功；排入折叠失败不改变它，
                # 话题保持原样，下一次写入该话题时会重新判断是否需要折叠。
                logger.exception("交互应用后排入话题折叠失败: topic_id=%s", topic_id)
        return InteractionSubmissionResult(
            interaction_id=payload.interaction_id,
            topic_id=topic_id,
        )

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision:
        """仅重试已明确分类、且可复用同一 interaction identity 的失败。

        折叠只在话题正被占用时重试；其他失败结束这个折叠项，话题保持原样，
        下一次写入该话题时会重新判断是否需要折叠。
        """
        if context.kind == TopicCompactionCodec.kind:
            if isinstance(error, TopicBusyError):
                return FailureDecision(
                    action=FailureAction.RETRY,
                    retry_after_seconds=0.05,
                    reason="topic_compaction_busy",
                )
            return FailureDecision(action=FailureAction.FAIL, reason="topic_compaction_failed")

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
    """Patchouli 交互摄入 lane 的轻量业务队列。

    提供 ``compaction`` 时，交互应用后话题需要折叠就排入一个折叠项；不提供时
    只应用交互，不折叠。交互应用的时限由 ``apply_timeout_seconds`` 施加，lane
    的默认策略不设统一时限，折叠的时限由摘要调用自身承担。
    """

    LANE = "patchouli.interaction_submission"

    def __init__(
        self,
        apply_interaction: ApplyInteraction,
        *,
        compaction: TopicCompactionPort | None = None,
        apply_timeout_seconds: float | None = 30.0,
        store: InMemoryWorkStore | None = None,
        runtime_events: RuntimeEventSink | None = None,
        policy: QueuePolicy | None = None,
    ) -> None:
        self._codecs = WorkPayloadCodecRegistry()
        # 队列只驻留于进程内，重启后无需保留旧载荷版本。
        self._codecs.register(InteractionSubmissionCodec())
        self._codecs.register(TopicCompactionCodec())
        if apply_timeout_seconds is not None and apply_timeout_seconds <= 0:
            raise ValueError("apply_timeout_seconds must be greater than 0")
        lane_policy = policy or QueuePolicy(
            capacity=256,
            max_concurrency=4,
            ordered_by_key=True,
            cancellable=False,
            max_attempts=3,
            terminal_retention=512,
        )
        if compaction is not None and not lane_policy.ordered_by_key:
            # 折叠项靠与话题提交共用排序键挡住后续提交；不按键串行时两者会争用话题。
            raise ValueError("topic compaction requires a lane ordered by key")
        self._runtime = WorkQueueRuntime(
            store=store or InMemoryWorkStore(),
            payload_codecs=self._codecs,
            runtime_events=runtime_events,
            worker_poll_interval_seconds=0.02,
            shutdown_wait_seconds=2.0,
        )
        self._compaction = compaction
        self._runtime.register_lane(
            self.LANE,
            handler=InteractionSubmissionHandler(
                apply_interaction,
                apply_timeout_seconds=apply_timeout_seconds,
                after_apply=self._schedule_compaction if compaction is not None else None,
                run_compaction=self._run_compaction if compaction is not None else None,
            ),
            policy=lane_policy,
        )
        self._max_submission_entries = lane_policy.capacity + lane_policy.terminal_retention
        self._submissions: OrderedDict[str, _StoredSubmission] = OrderedDict()
        self._submit_lock = asyncio.Lock()
        self._stopped = asyncio.Event()
        # 每个话题至多一个未结束的折叠项：话题坐标到折叠项的工作标识。
        self._compactions: dict[tuple[WorkspaceIdentity, str], str] = {}
        self._compaction_sequence = itertools.count(1)
        self._compactions_closed = asyncio.Event()

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
        self.close_compactions()
        try:
            return await self._runtime.stop()
        finally:
            # 唤醒仍在等待 applied gate 的调用方，由其把非终态解释为运行时不可用。
            self._stopped.set()

    def close_compactions(self) -> None:
        """停止折叠：不再排入折叠项，排队中的折叠项直接结束，进行中的折叠被取消。

        系统停止前调用，使排空队列时不再等待摘要模型；折叠在摘要返回后才写回，
        取消不改动话题，随后的结算带走话题的完整 blocks。
        """
        self._compactions_closed.set()

    async def _schedule_compaction(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """交互应用后话题需要折叠时排入一个折叠项。

        折叠项与该话题的提交共用排序键，排入之后的同一话题提交都排在它之后。
        话题已有未结束的折叠项、或折叠已关闭时不再排入；队列已满或已停止时放弃
        这一次，交互已经写入，下一次写入该话题时会重新判断。
        """
        port = self._compaction
        if port is None or self._compactions_closed.is_set():
            return
        if not port.needs_compact(belong_to, topic_id):
            return
        key = (belong_to, topic_id)
        async with self._submit_lock:
            await self._forget_finished_compactions_locked()
            if key in self._compactions:
                return
            work_id = f"topic_compaction:{topic_id}#{next(self._compaction_sequence)}"
            compaction = TopicCompaction(belong_to=belong_to, topic_id=topic_id)
            item = WorkItem(
                work_id=work_id,
                lane=self.LANE,
                kind=TopicCompactionCodec.kind,
                schema_version=TopicCompactionCodec.schema_version,
                payload=self._codecs.encode(
                    TopicCompactionCodec.kind, TopicCompactionCodec.schema_version, compaction
                ),
                ordering_key=topic_ordering_key(topic_id),
                correlation_id=topic_id,
            )
            try:
                await self._runtime.enqueue(item)
            except (WorkQueueCapacityError, WorkQueueStoppedError) as error:
                logger.warning(
                    "话题折叠未能排入提交队列，留待下次写入: topic_id=%s, reason=%s",
                    topic_id,
                    type(error).__name__,
                )
                return
            self._compactions[key] = work_id

    async def _forget_finished_compactions_locked(self) -> None:
        """移除已到终态的折叠项记录；调用方持有提交锁。"""
        for key, work_id in list(self._compactions.items()):
            record = await self._runtime.get(work_id)
            if record is None or record.state in TERMINAL_WORK_STATES:
                del self._compactions[key]

    async def _run_compaction(self, compaction: TopicCompaction) -> None:
        """执行一个折叠项；折叠已关闭时直接结束，执行中关闭则取消折叠。"""
        port = self._compaction
        if port is None or self._compactions_closed.is_set():
            return
        compact_task = asyncio.create_task(
            port.compact_topic(compaction.belong_to, compaction.topic_id)
        )
        closed_task = asyncio.create_task(self._compactions_closed.wait())
        try:
            await asyncio.wait({compact_task, closed_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            compact_task.cancel()
            closed_task.cancel()
            await asyncio.gather(compact_task, closed_task, return_exceptions=True)
        if compact_task.cancelled():
            return
        error = compact_task.exception()
        if error is not None:
            if isinstance(error, TopicBusyError):  # 话题正被结算等占用，由失败分类重试
                raise error
            logger.warning(
                "话题折叠失败，话题保持原样: topic_id=%s",
                compaction.topic_id,
                exc_info=error,
            )
            raise error

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
        """等待已知交互进入终态，返回成功的交互数量。

        不限排序键时一并等待已排入的折叠项：系统停止时先关闭折叠再排空，
        进行中的折叠在这里收尾并释放话题的 lease，之后的结算不会遇到占用。
        """
        entries = [
            stored
            for stored in self._submissions.values()
            if ordering_key is None or stored.receipt.ordering_key == ordering_key
        ]
        compaction_ids = list(self._compactions.values()) if ordering_key is None else []
        if not entries and not compaction_ids:
            return 0
        results = await asyncio.gather(
            *(self._runtime.wait(stored.receipt.work_id, timeout=timeout) for stored in entries),
            *(self._runtime.wait(work_id, timeout=timeout) for work_id in compaction_ids),
        )
        return sum(
            1
            for record in results[: len(entries)]
            if record is not None and record.state == WorkState.SUCCEEDED
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
    "TopicCompaction",
    "TopicCompactionCodec",
    "TopicCompactionPort",
    "TransientInteractionSubmissionError",
    "topic_ordering_key",
]
