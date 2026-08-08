"""Local Work Queue Runtime 的 worker 与 shutdown 生命周期宿主。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from hivememory.system.runtime.work_queue.cancellation import WorkCancellationToken
from hivememory.system.runtime.work_queue.models import (
    WorkLaneShutdownSummary,
    WorkQueueShutdownSummary,
    WorkRecord,
    WorkState,
)
from hivememory.system.runtime.work_queue.policies import QueuePolicy
from hivememory.system.runtime.work_queue.ports import WorkStorePort

logger = logging.getLogger(__name__)

type ExecuteWork = Callable[
    [str, WorkRecord[Any], WorkCancellationToken],
    Awaitable[None],
]


@dataclass
class _LaneWorker:
    name: str
    policy: QueuePolicy
    shutdown_wait_seconds: float
    running: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    cancellation_tokens: dict[str, WorkCancellationToken] = field(default_factory=dict)
    slot_available: asyncio.Event = field(default_factory=asyncio.Event)
    dispatcher: asyncio.Task[None] | None = None


class WorkQueueSupervisor:
    """为每条 lane 维护独立 dispatcher、并发槽位与 drain 窗口。"""

    def __init__(
        self,
        *,
        store: WorkStorePort,
        execute_work: ExecuteWork,
        worker_poll_interval_seconds: float = 0.2,
        lease_seconds: float = 300.0,
        shutdown_wait_seconds: float = 10.0,
    ) -> None:
        if worker_poll_interval_seconds <= 0:
            raise ValueError("worker_poll_interval_seconds must be greater than 0")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")
        if shutdown_wait_seconds < 0:
            raise ValueError("shutdown_wait_seconds must not be negative")

        self._store = store
        self._execute_work = execute_work
        self._poll_interval = worker_poll_interval_seconds
        self._lease_seconds = lease_seconds
        self._shutdown_wait_seconds = shutdown_wait_seconds
        self._lanes: dict[str, _LaneWorker] = {}
        self._started = False
        self._stopping = False
        self._last_summary: WorkQueueShutdownSummary | None = None
        self._stop_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._started

    def register_lane(self, name: str, policy: QueuePolicy) -> None:
        wait_seconds = (
            policy.shutdown_wait_seconds
            if policy.shutdown_wait_seconds is not None
            else self._shutdown_wait_seconds
        )
        self._lanes[name] = _LaneWorker(
            name=name,
            policy=policy,
            shutdown_wait_seconds=wait_seconds,
        )

    async def start(self) -> None:
        if self._started:
            return
        if self._stopping or self._last_summary is not None:
            raise RuntimeError("Work queue supervisor cannot restart after stop")

        self._started = True
        for lane in self._lanes.values():
            lane.dispatcher = asyncio.create_task(
                self._dispatch_lane(lane),
                name=f"work_queue_dispatch_{lane.name}",
            )

    async def request_running_cancellation(
        self,
        lane_name: str,
        work_id: str,
        *,
        reason: str | None,
    ) -> bool | None:
        lane = self._lanes[lane_name]
        task = lane.running.get(work_id)
        token = lane.cancellation_tokens.get(work_id)
        if task is None or token is None:
            return None
        if token.requested:
            return False

        token.request(reason)
        task.cancel()
        return True

    async def stop(self) -> WorkQueueShutdownSummary:
        async with self._stop_lock:
            return await self._stop_once()

    async def _stop_once(self) -> WorkQueueShutdownSummary:
        if self._last_summary is not None:
            return replace(self._last_summary, already_stopped=True)

        self._stopping = True
        dispatchers = [lane.dispatcher for lane in self._lanes.values() if lane.dispatcher]
        for dispatcher in dispatchers:
            dispatcher.cancel()
        if dispatchers:
            await asyncio.gather(*dispatchers, return_exceptions=True)

        summaries = await asyncio.gather(*(self._drain_lane(lane) for lane in self._lanes.values()))
        self._started = False
        self._last_summary = WorkQueueShutdownSummary(lanes=tuple(summaries))
        return self._last_summary

    async def _dispatch_lane(self, lane: _LaneWorker) -> None:
        try:
            while not self._stopping:
                available_slots = lane.policy.max_concurrency - len(lane.running)
                if available_slots <= 0:
                    lane.slot_available.clear()
                    if len(lane.running) >= lane.policy.max_concurrency:
                        await lane.slot_available.wait()
                    continue

                lease_seconds = max(
                    self._lease_seconds,
                    (lane.policy.timeout_seconds or 0.0) + 1.0,
                )
                try:
                    records = await self._store.claim_ready(
                        lane.name,
                        limit=available_slots,
                        lease_seconds=lease_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Work queue claim failed: lane=%s", lane.name)
                    await asyncio.sleep(self._poll_interval)
                    continue

                if not records:
                    await self._store.wait_for_ready(lane.name, self._poll_interval)
                    continue

                for record in records:
                    token = WorkCancellationToken()
                    task = asyncio.create_task(
                        self._execute_work(lane.name, record, token),
                        name=f"work_queue_{lane.name}_{record.work_id}",
                    )
                    lane.running[record.work_id] = task
                    lane.cancellation_tokens[record.work_id] = token
                    task.add_done_callback(
                        lambda completed, *, current_lane=lane, work_id=record.work_id: (
                            self._work_done(current_lane, work_id, completed)
                        )
                    )
        except asyncio.CancelledError:
            return

    def _work_done(
        self,
        lane: _LaneWorker,
        work_id: str,
        task: asyncio.Task[None],
    ) -> None:
        lane.running.pop(work_id, None)
        lane.cancellation_tokens.pop(work_id, None)
        lane.slot_available.set()
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.error(
                "Work queue execution escaped isolation: lane=%s, work_id=%s",
                lane.name,
                work_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _drain_lane(self, lane: _LaneWorker) -> WorkLaneShutdownSummary:
        tasks = set(lane.running.values())
        timed_out = False
        cancellation_requested_ids: set[str] = set()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=lane.shutdown_wait_seconds)
            timed_out = bool(pending)
            if pending and lane.policy.cancellable:
                for work_id, task in tuple(lane.running.items()):
                    if task not in pending or task.done():
                        continue
                    token = lane.cancellation_tokens.get(work_id)
                    if token is not None:
                        token.request("shutdown_timeout")
                    cancellation_requested_ids.add(work_id)
                    task.cancel()
                if pending:
                    await asyncio.wait(pending, timeout=0.1)

        cancelled_during_shutdown = 0
        for work_id in cancellation_requested_ids:
            record = await self._store.get(work_id)
            if record is None or record.state == WorkState.CANCELLED:
                cancelled_during_shutdown += 1

        snapshot = await self._store.snapshot(lane.name)
        return WorkLaneShutdownSummary(
            lane=lane.name,
            queued=snapshot.queued,
            retry_wait=snapshot.retry_wait,
            running=snapshot.running,
            cancelled_during_shutdown=cancelled_during_shutdown,
            drain_timed_out=timed_out,
            in_memory_loss_risk=snapshot.active if not self._store.is_durable else 0,
        )


__all__ = ["WorkQueueSupervisor"]
