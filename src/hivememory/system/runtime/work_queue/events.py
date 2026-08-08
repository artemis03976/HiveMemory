"""Local Work Queue Runtime 的通用观测事件构造。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher, Severity
from hivememory.system.runtime.work_queue.models import (
    WorkItem,
    WorkLaneSnapshot,
    WorkRecord,
    WorkState,
)
from hivememory.system.runtime.work_queue.policies import QueuePolicy


def _identifier_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _duration_ms(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    try:
        return round(max(0.0, (end - start).total_seconds() * 1000), 3)
    except (TypeError, ValueError):
        return None


class WorkQueueEventEmitter:
    """只发布安全摘要、不暴露业务 payload 的 work 事件。"""

    def __init__(self, sink: RuntimeEventSink) -> None:
        self._publisher = RuntimeEventPublisher(
            sink,
            subsystem="system",
            source="system.work_queue",
            component="work_queue",
        )

    def emit(
        self,
        event_type: RuntimeEventType,
        *,
        item: WorkItem[Any],
        state: WorkState | None = None,
        attempt_count: int = 0,
        record: WorkRecord[Any] | None = None,
        policy: QueuePolicy | None = None,
        snapshot: WorkLaneSnapshot | None = None,
        severity: Severity = "info",
        reason: str | None = None,
        error_class: str | None = None,
        next_retry_at: datetime | None = None,
        result_ref: str | None = None,
    ) -> None:
        now = datetime.now(record.enqueued_at.tzinfo) if record is not None else None
        data: dict[str, object] = {
            "work_id": item.work_id,
            "lane": item.lane,
            "kind": item.kind,
            "schema_version": item.schema_version,
            "state": state.value if state is not None else None,
            "attempt_count": attempt_count,
            "ordering_key_digest": _identifier_digest(item.ordering_key),
            "correlation_id_digest": _identifier_digest(item.correlation_id),
            "idempotency_key_digest": _identifier_digest(item.idempotency_key),
            "error_class": error_class,
            "next_retry_at": next_retry_at.isoformat() if next_retry_at is not None else None,
            "result_ref": result_ref,
        }
        if record is not None:
            data["queue_latency_ms"] = _duration_ms(record.enqueued_at, record.started_at)
            if record.finished_at is not None:
                data["execution_latency_ms"] = _duration_ms(
                    record.started_at,
                    record.finished_at,
                )
            elif state == WorkState.RUNNING:
                data["execution_latency_ms"] = _duration_ms(record.started_at, now)
        if policy is not None:
            data.update(
                {
                    "capacity": policy.capacity,
                    "max_concurrency": policy.max_concurrency,
                }
            )
        if snapshot is not None:
            data.update(
                {
                    "queued_count": snapshot.queued,
                    "running_count": snapshot.running,
                    "retry_wait_count": snapshot.retry_wait,
                }
            )

        self._publisher.bind(task_type="background", task_id=item.work_id).emit(
            event_type,
            status=state.value if state is not None else None,
            severity=severity,
            reason=reason,
            data={key: value for key, value in data.items() if value is not None},
        )


__all__ = ["WorkQueueEventEmitter"]
