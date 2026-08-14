"""Local Work Queue Runtime 的进程内状态存储实现。

该 adapter 只适用于单进程、单 event loop。进程退出后所有 queued、running 与
retry-wait work 都会丢失，RuntimeEvent 也不能用于恢复这些状态。
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from hivememory.system.runtime.work_queue.exceptions import (
    DuplicateWorkItemError,
    DuplicateWorkLaneError,
    UnknownWorkLaneError,
    WorkQueueCapacityError,
    WorkStateConflictError,
)
from hivememory.system.runtime.work_queue.models import (
    TERMINAL_WORK_STATES,
    WorkErrorSnapshot,
    WorkItem,
    WorkLaneSnapshot,
    WorkRecord,
    WorkState,
    can_transition_work_state,
)
from hivememory.system.runtime.work_queue.policies import QueuePolicy


@dataclass
class _LaneState:
    policy: QueuePolicy
    order: deque[str] = field(default_factory=deque)
    terminal_ids: deque[str] = field(default_factory=deque)


class InMemoryWorkStore:
    """按 lane 分桶、由单把异步锁保护原子状态迁移的内存 store。"""

    def __init__(self) -> None:
        self._records: dict[str, WorkRecord] = {}
        self._lanes: dict[str, _LaneState] = {}
        self._condition = asyncio.Condition()

    @property
    def is_durable(self) -> bool:
        return False

    def configure_lane(self, lane: str, policy: QueuePolicy) -> None:
        """在 runtime 启动前注册 lane 的通用存储策略。"""

        existing = self._lanes.get(lane)
        if existing is None:
            self._lanes[lane] = _LaneState(policy=policy)
            return
        if existing.policy != policy:
            raise DuplicateWorkLaneError(f"Work store lane '{lane}' is already configured")

    async def enqueue(self, item: WorkItem) -> WorkRecord:
        """原子检查 work ID 与 lane capacity 后接受工作项。"""

        async with self._condition:
            lane = self._lane_for(item.lane)
            if item.work_id in self._records:
                raise DuplicateWorkItemError(f"Work item '{item.work_id}' already exists")
            if self._active_count(lane) >= lane.policy.capacity:
                raise WorkQueueCapacityError(item.lane, lane.policy.capacity)

            now = datetime.now(UTC)
            record = WorkRecord(
                item=item,
                state=WorkState.QUEUED,
                attempt_count=0,
                enqueued_at=now,
                available_at=now,
            )
            self._records[item.work_id] = record
            lane.order.append(item.work_id)
            self._condition.notify_all()
            return record

    async def claim_ready(
        self,
        lane: str,
        *,
        limit: int,
    ) -> list[WorkRecord]:
        """按原始入队顺序原子 claim 当前可执行项。"""

        if limit < 1:
            return []

        async with self._condition:
            lane_state = self._lane_for(lane)
            now = datetime.now(UTC)
            work_ids = self._claimable_work_ids(lane_state, now=now, limit=limit)
            claimed: list[WorkRecord] = []
            for work_id in work_ids:
                current = self._records[work_id]
                # retry-wait 到期后在同一临界区完成 QUEUED -> RUNNING 两步迁移。
                if current.state == WorkState.RETRY_WAIT:
                    current = replace(current, state=WorkState.QUEUED)
                running = replace(
                    current,
                    state=WorkState.RUNNING,
                    attempt_count=current.attempt_count + 1,
                    started_at=now,
                    finished_at=None,
                )
                self._records[work_id] = running
                claimed.append(running)

            if claimed:
                self._condition.notify_all()
            return claimed

    async def mark_succeeded(
        self,
        work_id: str,
        result_ref: str | None = None,
    ) -> WorkRecord:
        return await self._mark_terminal(
            work_id,
            state=WorkState.SUCCEEDED,
            result_ref=result_ref,
        )

    async def schedule_retry(
        self,
        work_id: str,
        *,
        available_at: datetime,
        error: WorkErrorSnapshot,
    ) -> WorkRecord:
        if available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")

        async with self._condition:
            lane, current = self._record_with_lane(work_id)
            self._require_transition(current, WorkState.RETRY_WAIT)
            retrying = replace(
                current,
                state=WorkState.RETRY_WAIT,
                available_at=available_at,
                last_error=error,
            )
            self._records[work_id] = retrying
            self._condition.notify_all()
            self._prune_terminal(lane)
            return retrying

    async def mark_failed(
        self,
        work_id: str,
        error: WorkErrorSnapshot,
    ) -> WorkRecord:
        return await self._mark_terminal(
            work_id,
            state=WorkState.FAILED,
            error=error,
        )

    async def mark_dead_lettered(
        self,
        work_id: str,
        error: WorkErrorSnapshot,
    ) -> WorkRecord:
        return await self._mark_terminal(
            work_id,
            state=WorkState.DEAD_LETTER,
            error=error,
        )

    async def cancel(self, work_id: str) -> WorkRecord | None:
        """幂等取消非终态工作；返回已提交的取消记录或未接纳标记。"""

        async with self._condition:
            current = self._records.get(work_id)
            if current is None:
                return None
            if current.state == WorkState.CANCELLED:
                return current
            if current.state in TERMINAL_WORK_STATES:
                return None

            lane = self._lane_for(current.lane)
            self._require_transition(current, WorkState.CANCELLED)
            cancelled = replace(
                current,
                state=WorkState.CANCELLED,
                finished_at=datetime.now(UTC),
            )
            self._records[work_id] = cancelled
            lane.terminal_ids.append(work_id)
            self._prune_terminal(lane)
            self._condition.notify_all()
            return cancelled

    async def get(self, work_id: str) -> WorkRecord | None:
        async with self._condition:
            return self._records.get(work_id)

    async def wait(
        self,
        work_id: str,
        timeout: float | None = None,
    ) -> WorkRecord | None:
        """等待 work 进入终态；超时返回当时的最新快照。"""

        async with self._condition:
            current = self._records.get(work_id)
            if current is None or current.state in TERMINAL_WORK_STATES:
                return current

            def completed() -> bool:
                record = self._records.get(work_id)
                return record is None or record.state in TERMINAL_WORK_STATES

            try:
                if timeout is None:
                    await self._condition.wait_for(completed)
                else:
                    async with asyncio.timeout(max(0.0, timeout)):
                        await self._condition.wait_for(completed)
            except TimeoutError:
                pass
            return self._records.get(work_id)

    async def wait_for_ready(self, lane: str, timeout: float) -> None:
        """等待 lane 出现可 claim 项，超时用于推进 retry visibility。"""

        async with self._condition:
            lane_state = self._lane_for(lane)

            def ready() -> bool:
                return bool(
                    self._claimable_work_ids(
                        lane_state,
                        now=datetime.now(UTC),
                        limit=1,
                    )
                )

            if ready():
                return
            try:
                async with asyncio.timeout(max(0.0, timeout)):
                    await self._condition.wait_for(ready)
            except TimeoutError:
                return

    async def snapshot(self, lane: str) -> WorkLaneSnapshot:
        async with self._condition:
            lane_state = self._lane_for(lane)
            counts = Counter(
                self._records[work_id].state
                for work_id in lane_state.order
                if work_id in self._records
            )
            return WorkLaneSnapshot(
                lane=lane,
                queued=counts[WorkState.QUEUED],
                running=counts[WorkState.RUNNING],
                retry_wait=counts[WorkState.RETRY_WAIT],
                succeeded=counts[WorkState.SUCCEEDED],
                failed=counts[WorkState.FAILED],
                dead_letter=counts[WorkState.DEAD_LETTER],
                cancelled=counts[WorkState.CANCELLED],
            )

    async def _mark_terminal(
        self,
        work_id: str,
        *,
        state: WorkState,
        error: WorkErrorSnapshot | None = None,
        result_ref: str | None = None,
    ) -> WorkRecord:
        async with self._condition:
            lane, current = self._record_with_lane(work_id)
            self._require_transition(current, state)
            terminal = replace(
                current,
                state=state,
                finished_at=datetime.now(UTC),
                last_error=error,
                result_ref=result_ref,
            )
            self._records[work_id] = terminal
            lane.terminal_ids.append(work_id)
            self._prune_terminal(lane)
            self._condition.notify_all()
            return terminal

    def _claimable_work_ids(
        self,
        lane: _LaneState,
        *,
        now: datetime,
        limit: int,
    ) -> list[str]:
        claimed: list[str] = []
        blocked_ordering_keys: set[str] = set()

        for work_id in lane.order:
            record = self._records.get(work_id)
            if record is None or record.state in TERMINAL_WORK_STATES:
                continue

            ordering_key = record.item.ordering_key if lane.policy.ordered_by_key else None
            if ordering_key is not None:
                if ordering_key in blocked_ordering_keys:
                    continue
                # 同 key 的第一个非终态项无论是否 ready，都阻止后续项越过。
                blocked_ordering_keys.add(ordering_key)

            if record.state not in {WorkState.QUEUED, WorkState.RETRY_WAIT}:
                continue
            if record.available_at > now:
                continue

            claimed.append(work_id)
            if len(claimed) >= limit:
                break

        return claimed

    def _active_count(self, lane: _LaneState) -> int:
        return sum(
            1
            for work_id in lane.order
            if (record := self._records.get(work_id)) is not None
            and record.state not in TERMINAL_WORK_STATES
        )

    def _record_with_lane(self, work_id: str) -> tuple[_LaneState, WorkRecord]:
        current = self._records.get(work_id)
        if current is None:
            raise KeyError(work_id)
        return self._lane_for(current.lane), current

    def _lane_for(self, lane: str) -> _LaneState:
        lane_state = self._lanes.get(lane)
        if lane_state is None:
            raise UnknownWorkLaneError(f"Work store lane '{lane}' is not configured")
        return lane_state

    @staticmethod
    def _require_transition(record: WorkRecord, target: WorkState) -> None:
        if not can_transition_work_state(record.state, target):
            raise WorkStateConflictError(
                f"Invalid work state transition: {record.state.value} -> {target.value}"
            )

    def _prune_terminal(self, lane: _LaneState) -> None:
        while len(lane.terminal_ids) > lane.policy.terminal_retention:
            work_id = lane.terminal_ids.popleft()
            record = self._records.get(work_id)
            if record is None or record.state not in TERMINAL_WORK_STATES:
                continue
            del self._records[work_id]
            try:
                lane.order.remove(work_id)
            except ValueError:
                pass


__all__ = ["InMemoryWorkStore"]
