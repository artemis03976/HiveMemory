"""结构化队列任务适配器与类型化句柄契约测试。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from hivememory.system.runtime.work_queue import (
    QueueTaskIdentity,
    TaskHandle,
    WorkPayloadCodecRegistry,
    WorkRecord,
    WorkState,
    adapt_queue_task,
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    value: str


class _TaskAdapter:
    kind = "test.structured_task"
    schema_version = 1

    @staticmethod
    def identity(task: _Task) -> QueueTaskIdentity:
        return QueueTaskIdentity(
            work_id=f"test:{task.task_id}",
            ordering_key=task.task_id,
            correlation_id=task.task_id,
            idempotency_key=task.task_id,
        )

    @staticmethod
    def encode(task: _Task) -> object:
        return {"task_id": task.task_id, "value": task.value}

    @staticmethod
    def decode(payload: object) -> _Task:
        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")
        return _Task(task_id=payload["task_id"], value=payload["value"])


def _record(item, *, state: WorkState, error=None) -> WorkRecord:
    now = datetime.now(UTC)
    return WorkRecord(
        item=item,
        state=state,
        attempt_count=1,
        enqueued_at=now,
        available_at=now,
        started_at=now,
        finished_at=now if state == WorkState.SUCCEEDED else None,
        last_error=error,
    )


def test_adapter_builds_private_envelope_and_round_trips_structured_task():
    task = _Task(task_id="task-1", value="payload")
    adapter = _TaskAdapter()
    codecs = WorkPayloadCodecRegistry()
    codecs.register(adapter)

    item = adapt_queue_task(
        task,
        lane="test-lane",
        adapter=adapter,
        codecs=codecs,
    )

    assert item.work_id == "test:task-1"
    assert item.ordering_key == "task-1"
    assert item.idempotency_key == "task-1"
    assert codecs.decode(item.kind, item.schema_version, item.payload) == task


@pytest.mark.asyncio
async def test_handle_combines_record_with_stable_typed_result():
    task = _Task(task_id="task-1", value="payload")
    adapter = _TaskAdapter()
    codecs = WorkPayloadCodecRegistry()
    codecs.register(adapter)
    item = adapt_queue_task(
        task,
        lane="test-lane",
        adapter=adapter,
        codecs=codecs,
    )
    record = _record(item, state=WorkState.SUCCEEDED)
    queue = AsyncMock()
    queue.wait.return_value = record
    handle = TaskHandle[_Task, tuple[str, ...]](
        task=task,
        task_id=task.task_id,
        work_id=item.work_id,
        queue=queue,
    )
    handle._record_execution_result(("result",))

    first = await handle.wait()
    second = await handle.wait()

    assert first is not None
    assert first.succeeded is True
    assert first.result == ("result",)
    assert second == first


@pytest.mark.asyncio
async def test_handle_rolls_back_cancel_marker_when_queue_cancel_raises():
    task = _Task(task_id="task-1", value="payload")
    queue = AsyncMock()
    queue.cancel.side_effect = RuntimeError("queue unavailable")
    handle = TaskHandle[_Task, None](
        task=task,
        task_id=task.task_id,
        work_id="test:task-1",
        queue=queue,
    )

    with pytest.raises(RuntimeError, match="queue unavailable"):
        await handle.cancel(reason="user_requested")

    assert handle.cancel_requested is False
    assert handle.cancel_reason is None


@pytest.mark.asyncio
async def test_handle_exposes_accepted_cancel_reason_before_terminal_cancel():
    task = _Task(task_id="task-1", value="payload")
    adapter = _TaskAdapter()
    codecs = WorkPayloadCodecRegistry()
    codecs.register(adapter)
    item = adapt_queue_task(
        task,
        lane="test-lane",
        adapter=adapter,
        codecs=codecs,
    )
    queue = AsyncMock()
    queue.cancel.return_value = True
    queue.get.return_value = _record(item, state=WorkState.RUNNING)
    handle = TaskHandle[_Task, None](
        task=task,
        task_id=task.task_id,
        work_id=item.work_id,
        queue=queue,
    )

    assert await handle.cancel(reason="user_requested") is True
    outcome = await handle.snapshot()

    assert outcome is not None
    assert outcome.record.state == WorkState.RUNNING
    assert outcome.cancel_reason == "user_requested"
