"""Local Work Queue Runtime 的 lane registry 与执行编排。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink
from hivememory.system.runtime.publisher import Severity
from hivememory.system.runtime.work_queue.cancellation import WorkCancellationToken
from hivememory.system.runtime.work_queue.events import WorkQueueEventEmitter
from hivememory.system.runtime.work_queue.exceptions import (
    DuplicateWorkLaneError,
    UnknownWorkLaneError,
    UnsupportedWorkQueueFeatureError,
    WorkQueueStoppedError,
)
from hivememory.system.runtime.work_queue.models import (
    WorkErrorSnapshot,
    WorkExecutionContext,
    WorkItem,
    WorkQueueShutdownSummary,
    WorkReceipt,
    WorkRecord,
    WorkState,
)
from hivememory.system.runtime.work_queue.policies import (
    FailureAction,
    FailureDecision,
    QueuePolicy,
)
from hivememory.system.runtime.work_queue.ports import WorkHandlerPort, WorkStorePort
from hivememory.system.runtime.work_queue.supervisor import WorkQueueSupervisor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkLane:
    """已注册 lane 的公开只读描述。"""

    name: str
    policy: QueuePolicy


@dataclass(frozen=True)
class _LaneBinding:
    lane: WorkLane
    handler: WorkHandlerPort[Any, Any]


class WorkQueueRuntime:
    """一套进程内 runtime、多条相互隔离的业务 lane。"""

    def __init__(
        self,
        *,
        store: WorkStorePort,
        runtime_events: RuntimeEventSink | None = None,
        worker_poll_interval_seconds: float = 0.2,
        lease_seconds: float = 300.0,
        shutdown_wait_seconds: float = 10.0,
    ) -> None:
        self._store = store
        self._bindings: dict[str, _LaneBinding] = {}
        self._events = WorkQueueEventEmitter(runtime_events or NullRuntimeEventSink())
        self._supervisor = WorkQueueSupervisor(
            store=store,
            execute_work=self._execute_record,
            worker_poll_interval_seconds=worker_poll_interval_seconds,
            lease_seconds=lease_seconds,
            shutdown_wait_seconds=shutdown_wait_seconds,
        )
        self._accepting = True
        self._stopped = False

    @property
    def lanes(self) -> tuple[WorkLane, ...]:
        return tuple(binding.lane for binding in self._bindings.values())

    @property
    def started(self) -> bool:
        return self._supervisor.started

    def register_lane(
        self,
        name: str,
        *,
        handler: WorkHandlerPort[Any, Any],
        policy: QueuePolicy,
    ) -> WorkLane:
        """在 start 前注册一条拥有独立 policy 与 handler 的 lane。"""

        if self._stopped:
            raise WorkQueueStoppedError("Cannot register lane after runtime stop")
        if self.started:
            raise RuntimeError("Cannot register work queue lane after runtime start")
        if not name.strip():
            raise ValueError("lane name must not be blank")
        if name in self._bindings:
            raise DuplicateWorkLaneError(f"Work queue lane '{name}' is already registered")
        if policy.priority_enabled:
            raise UnsupportedWorkQueueFeatureError(
                "priority_enabled is not implemented by the Q1 in-memory runtime"
            )

        lane = WorkLane(name=name, policy=policy)
        self._store.configure_lane(name, policy)
        self._bindings[name] = _LaneBinding(lane=lane, handler=handler)
        self._supervisor.register_lane(name, policy)
        return lane

    async def start(self) -> None:
        if self._stopped:
            raise WorkQueueStoppedError("Work queue runtime cannot restart after stop")
        await self._supervisor.start()

    async def stop(self) -> WorkQueueShutdownSummary:
        """停止接收和 claim，按 lane drain 后返回未完成工作摘要。"""

        self._accepting = False
        self._stopped = True
        return await self._supervisor.stop()

    async def enqueue(self, item: WorkItem[Any]) -> WorkReceipt:
        binding = self._bindings.get(item.lane)
        if binding is None:
            await self._emit_rejected(item, reason="unknown_lane")
            raise UnknownWorkLaneError(f"Work queue lane '{item.lane}' is not registered")
        if not self._accepting:
            await self._emit_rejected(item, reason="runtime_stopped", policy=binding.lane.policy)
            raise WorkQueueStoppedError("Work queue runtime is not accepting new work")

        try:
            record = await self._store.enqueue(item)
        except Exception as exc:
            await self._emit_rejected(
                item,
                reason=type(exc).__name__,
                policy=binding.lane.policy,
            )
            raise

        await self._emit_transition(
            RuntimeEventType.WORK_QUEUED,
            binding=binding,
            record=record,
        )
        return WorkReceipt(
            work_id=record.work_id,
            lane=record.lane,
            state=record.state,
            enqueued_at=record.enqueued_at,
        )

    async def get(self, work_id: str) -> WorkRecord[Any] | None:
        return await self._store.get(work_id)

    async def wait(
        self,
        work_id: str,
        timeout: float | None = None,
    ) -> WorkRecord[Any] | None:
        return await self._store.wait(work_id, timeout=timeout)

    async def cancel(
        self,
        work_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool:
        """按 lane policy 幂等取消 queued/retry-wait/running work。"""

        record = await self._store.get(work_id)
        if record is None:
            return False
        binding = self._bindings.get(record.lane)
        if binding is None or not binding.lane.policy.cancellable:
            return False
        if record.state == WorkState.CANCELLED:
            return True
        if record.state in {
            WorkState.SUCCEEDED,
            WorkState.FAILED,
            WorkState.DEAD_LETTER,
        }:
            return False

        if record.state == WorkState.RUNNING:
            requested = await self._supervisor.request_running_cancellation(
                record.lane,
                work_id,
                reason=reason,
            )
            if requested is not None:
                if requested:
                    await self._emit_transition(
                        RuntimeEventType.WORK_CANCEL_REQUESTED,
                        binding=binding,
                        record=record,
                        reason=reason,
                    )
                return True

        cancelled = await self._store.cancel(work_id)
        if not cancelled:
            return False
        await self._emit_transition(
            RuntimeEventType.WORK_CANCEL_REQUESTED,
            binding=binding,
            record=record,
            reason=reason,
        )
        cancelled_record = await self._latest_or(
            record,
            state=WorkState.CANCELLED,
            finished_at=datetime.now(UTC),
            lease_until=None,
        )
        await self._emit_transition(
            RuntimeEventType.WORK_CANCELLED,
            binding=binding,
            record=cancelled_record,
            reason=reason,
        )
        return True

    async def _execute_record(
        self,
        lane_name: str,
        record: WorkRecord[Any],
        cancellation: WorkCancellationToken,
    ) -> None:
        binding = self._bindings[lane_name]
        context = WorkExecutionContext(
            work_id=record.work_id,
            lane=record.lane,
            kind=record.item.kind,
            schema_version=record.item.schema_version,
            attempt_count=record.attempt_count,
            correlation_id=record.item.correlation_id,
            idempotency_key=record.item.idempotency_key,
            cancellation=cancellation,
        )
        await self._emit_transition(
            RuntimeEventType.WORK_STARTED,
            binding=binding,
            record=record,
        )

        try:
            if binding.lane.policy.timeout_seconds is None:
                result = await binding.handler.execute(record.item.payload, context)
            else:
                async with asyncio.timeout(binding.lane.policy.timeout_seconds):
                    result = await binding.handler.execute(record.item.payload, context)

            # handler 即使吞掉 CancelledError，也不能覆盖已经记录的取消请求。
            if cancellation.requested:
                await self._finish_cancelled(binding, record, cancellation)
                return

            result_ref = self._result_ref(result)
            await self._store.mark_succeeded(record.work_id, result_ref=result_ref)
            succeeded = await self._latest_or(
                record,
                state=WorkState.SUCCEEDED,
                finished_at=datetime.now(UTC),
                lease_until=None,
                result_ref=result_ref,
            )
            await self._emit_transition(
                RuntimeEventType.WORK_SUCCEEDED,
                binding=binding,
                record=succeeded,
                result_ref=result_ref,
            )
        except asyncio.CancelledError:
            await self._finish_cancelled(binding, record, cancellation)
        except Exception as exc:
            await self._handle_failure(binding, record, context, exc)

    async def _finish_cancelled(
        self,
        binding: _LaneBinding,
        record: WorkRecord[Any],
        cancellation: WorkCancellationToken,
    ) -> None:
        reason = cancellation.reason or "runtime_cancelled"
        if reason == "shutdown_timeout":
            await self._emit_transition(
                RuntimeEventType.WORK_CANCEL_REQUESTED,
                binding=binding,
                record=record,
                reason=reason,
            )
        cancelled = await self._store.cancel(record.work_id)
        if not cancelled:
            return
        latest = await self._latest_or(
            record,
            state=WorkState.CANCELLED,
            finished_at=datetime.now(UTC),
            lease_until=None,
        )
        await self._emit_transition(
            RuntimeEventType.WORK_CANCELLED,
            binding=binding,
            record=latest,
            reason=reason,
        )

    async def _handle_failure(
        self,
        binding: _LaneBinding,
        record: WorkRecord[Any],
        context: WorkExecutionContext,
        error: Exception,
    ) -> None:
        try:
            decision = binding.handler.classify_failure(error, context)
            if not isinstance(decision, FailureDecision):
                raise TypeError("classify_failure must return FailureDecision")
        except Exception:
            logger.exception(
                "Work failure classifier failed: lane=%s, work_id=%s",
                record.lane,
                record.work_id,
            )
            decision = FailureDecision(
                action=FailureAction.FAIL,
                reason="failure_classifier_error",
            )

        error_snapshot = WorkErrorSnapshot(
            error_class=type(error).__name__,
            # reason 由业务 classifier 提供安全摘要，通用层不记录原始异常正文。
            message=decision.reason,
        )
        policy = binding.lane.policy

        if decision.action == FailureAction.TREAT_AS_SUCCESS:
            await self._store.mark_succeeded(record.work_id)
            succeeded = await self._latest_or(
                record,
                state=WorkState.SUCCEEDED,
                finished_at=datetime.now(UTC),
                lease_until=None,
            )
            await self._emit_transition(
                RuntimeEventType.WORK_SUCCEEDED,
                binding=binding,
                record=succeeded,
                reason=decision.reason,
            )
            return

        retry_exhausted = (
            decision.action == FailureAction.RETRY
            and policy.max_attempts > 0
            and record.attempt_count >= policy.max_attempts
        )
        if decision.action == FailureAction.RETRY and not retry_exhausted:
            retry_at = datetime.now(UTC) + timedelta(seconds=decision.retry_after_seconds or 0.0)
            await self._store.schedule_retry(
                record.work_id,
                available_at=retry_at,
                error=error_snapshot,
            )
            retrying = await self._latest_or(
                record,
                state=WorkState.RETRY_WAIT,
                available_at=retry_at,
                lease_until=None,
                last_error=error_snapshot,
            )
            await self._emit_transition(
                RuntimeEventType.WORK_RETRY_SCHEDULED,
                binding=binding,
                record=retrying,
                severity="warning",
                reason=decision.reason,
                error_class=error_snapshot.error_class,
                next_retry_at=retry_at,
            )
            return

        if decision.action == FailureAction.DEAD_LETTER or retry_exhausted:
            reason = "max_attempts_exhausted" if retry_exhausted else decision.reason
            terminal_error = replace(error_snapshot, message=reason)
            await self._store.mark_dead_lettered(record.work_id, terminal_error)
            dead_lettered = await self._latest_or(
                record,
                state=WorkState.DEAD_LETTER,
                finished_at=datetime.now(UTC),
                lease_until=None,
                last_error=terminal_error,
            )
            await self._emit_transition(
                RuntimeEventType.WORK_DEAD_LETTERED,
                binding=binding,
                record=dead_lettered,
                severity="error",
                reason=reason,
                error_class=terminal_error.error_class,
            )
            return

        await self._store.mark_failed(record.work_id, error_snapshot)
        failed = await self._latest_or(
            record,
            state=WorkState.FAILED,
            finished_at=datetime.now(UTC),
            lease_until=None,
            last_error=error_snapshot,
        )
        await self._emit_transition(
            RuntimeEventType.WORK_FAILED,
            binding=binding,
            record=failed,
            severity="error",
            reason=decision.reason,
            error_class=error_snapshot.error_class,
        )

    async def _emit_transition(
        self,
        event_type: RuntimeEventType,
        *,
        binding: _LaneBinding,
        record: WorkRecord[Any],
        severity: Severity = "info",
        reason: str | None = None,
        error_class: str | None = None,
        next_retry_at: datetime | None = None,
        result_ref: str | None = None,
    ) -> None:
        try:
            snapshot = await self._store.snapshot(record.lane)
        except Exception:
            snapshot = None
        self._events.emit(
            event_type,
            item=record.item,
            state=record.state,
            attempt_count=record.attempt_count,
            record=record,
            policy=binding.lane.policy,
            snapshot=snapshot,
            severity=severity,
            reason=reason,
            error_class=error_class,
            next_retry_at=next_retry_at,
            result_ref=result_ref,
        )

    async def _emit_rejected(
        self,
        item: WorkItem[Any],
        *,
        reason: str,
        policy: QueuePolicy | None = None,
    ) -> None:
        snapshot = None
        if policy is not None:
            try:
                snapshot = await self._store.snapshot(item.lane)
            except Exception:
                pass
        self._events.emit(
            RuntimeEventType.WORK_REJECTED,
            item=item,
            policy=policy,
            snapshot=snapshot,
            severity="warning",
            reason=reason,
        )

    async def _latest_or(
        self,
        record: WorkRecord[Any],
        **updates: Any,
    ) -> WorkRecord[Any]:
        return await self._store.get(record.work_id) or replace(record, **updates)

    @staticmethod
    def _result_ref(result: Any) -> str | None:
        if result is None or isinstance(result, str):
            return result
        result_ref = getattr(result, "result_ref", None)
        return result_ref if isinstance(result_ref, str) else None


__all__ = ["WorkLane", "WorkQueueRuntime"]
