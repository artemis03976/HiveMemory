"""Patchouli 记忆生成任务的控制面。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.events import (
    MemoryTaskEventEmitter,
)
from hivememory.patchouli.control.memory_generation.queue import (
    MemoryGenerationHandle,
    MemoryGenerationQueue,
    MemoryGenerationResults,
)
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationResult,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
    MemoryGenerationTaskWaitResult,
    MemoryGenerationTaskWaitSummary,
    MemoryGenerationWork,
)
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from hivememory.system.runtime.work_queue import (
    QueuePolicy,
    TaskOutcome,
    WorkQueueShutdownSummary,
    WorkState,
)

logger = logging.getLogger(__name__)


@dataclass
class _MemoryTaskEntry:
    """控制器持有的领域投影元数据。

    ``WorkRecord`` 仍是唯一执行状态源；此处只关联不可变工作定义、类型化句柄、
    对外终态快照以及发布侧的同步原语。
    """

    work: MemoryGenerationWork
    created: MemoryGenerationTask
    handle: MemoryGenerationHandle | None = None
    finalizer: asyncio.Task[MemoryGenerationTask] | None = None
    final_snapshot: MemoryGenerationTask | None = None
    running_published: bool = False
    finalize_lock: asyncio.Lock | None = None

    def lock(self) -> asyncio.Lock:
        """按需创建单任务锁，串行化取消与终态 finalize。"""

        if self.finalize_lock is None:
            self.finalize_lock = asyncio.Lock()
        return self.finalize_lock


class MemoryGenerationTaskController:
    """提交结构化工作，并对外提供只读领域快照。

    通用队列负责接纳、执行与终态；控制器负责把 ``WorkRecord`` 投影为
    ``MemoryGenerationTask``，并在队列终态确定后独立完成 settlement 与领域
    状态事件发布。
    """

    def __init__(
        self,
        *,
        bus: Any,
        runtime_events: RuntimeEventSink | None = None,
        memory_queue: MemoryGenerationQueue | None = None,
        queue_policy: QueuePolicy | None = None,
    ) -> None:
        self._bus = bus
        event_sink = runtime_events or NullRuntimeEventSink()
        self._task_events = MemoryTaskEventEmitter(RuntimeEventPublisher(event_sink))
        self._queue = memory_queue or MemoryGenerationQueue(
            self._execute_generation,
            runtime_events=event_sink,
            policy=queue_policy,
        )

        self._entries: OrderedDict[str, _MemoryTaskEntry] = OrderedDict()

    @property
    def queue(self) -> MemoryGenerationQueue:
        """访问记忆生成业务队列。"""

        return self._queue

    async def start(self) -> None:
        """启动记忆生成 lane。"""

        await self._queue.start()

    async def stop(self) -> WorkQueueShutdownSummary:
        """停止队列，并给已确定的领域终态一个短暂收敛窗口。"""

        summary = await self._queue.stop()
        # queue 先完成 running work 的 drain/cancel，再等待领域 finalize 发布终态，
        # 避免随后卸载 local routes 时丢失取消或失败快照。
        await self.wait_all(timeout=0.5)
        return summary

    async def submit_generation(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationTask:
        """提交一个不可变工作定义，并返回其接纳快照。"""

        work = MemoryGenerationWork(task_id=str(uuid.uuid4()), spec=spec)
        created = MemoryGenerationTask.from_work(work, created_at=datetime.now(UTC))
        entry = _MemoryTaskEntry(work=work, created=created)
        self._entries[work.task_id] = entry

        self._task_events.created(created)

        try:
            # Controller 也可独立用于测试与嵌入式调用，因此保留幂等懒启动；
            # PatchouliSystem 的正常生命周期仍会显式启动该 lane。
            await self._queue.start()
            entry.handle = await self._queue.submit(work)
        except asyncio.CancelledError:
            self._entries.pop(work.task_id, None)
            raise
        except Exception as exc:
            failed = created.as_failed(
                str(exc) or type(exc).__name__,
                finished_at=datetime.now(UTC),
            )
            entry.final_snapshot = failed
            await self._publish_terminal(entry, failed)
            self._retain_terminal_entries()
            return failed

        entry.finalizer = asyncio.create_task(
            self._finalize(entry),
            name=f"memory_task_finalize_{work.task_id[:8]}",
        )
        # 返回接纳时刻的值对象；后续状态变化通过查询、等待和事件获得，不原地
        # 修改这个快照。
        return created

    async def submit_generation_many(
        self,
        specs: list[MemoryGenerationTaskSpec],
    ) -> list[MemoryGenerationTask]:
        """逐项提交任务；接纳失败时返回失败状态的快照。"""

        return [await self.submit_generation(spec) for spec in specs]

    async def _execute_generation(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> list[MemoryGenerationResult]:
        """由队列 handler 调用现有的记忆生成数据面。"""

        return await self._bus.request(
            PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
            spec,
        )

    async def _finalize(self, entry: _MemoryTaskEntry) -> MemoryGenerationTask:
        """观察执行开始与队列终态，并完成一次领域终态结算。

        started 信号只用于及时发布 ``RUNNING``，不参与状态判定；最终状态始终
        取自 handle 返回的 ``WorkRecord``。该协程独立于发起请求的客户端生命周期。
        """

        handle = entry.handle
        if handle is None:
            raise RuntimeError("accepted memory work is missing its task handle")

        outcome_waiter = asyncio.create_task(
            handle.wait(),
            name=f"memory_task_outcome_{entry.work.task_id[:8]}",
        )
        started_waiter = asyncio.create_task(
            handle.wait_started(),
            name=f"memory_task_started_{entry.work.task_id[:8]}",
        )
        try:
            # 同时等待开始信号和终态，既避免轮询，也覆盖工作在观察前已经快速
            # 完成的情况；无论哪一个先到，最终都必须等待 outcome。
            done, _ = await asyncio.wait(
                {outcome_waiter, started_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if started_waiter in done:
                current = await handle.snapshot()
                if current is not None:
                    await self._publish_running_if_needed(entry, current)
            outcome = await outcome_waiter
        finally:
            for waiter in (outcome_waiter, started_waiter):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(
                outcome_waiter,
                started_waiter,
                return_exceptions=True,
            )

        async with entry.lock():
            # 与 cancel_task 共用单任务锁，确保取消请求和领域终态事件不会交错
            # 覆盖彼此的最终快照。
            snapshot = await self._settle_outcome(entry, outcome)
            entry.final_snapshot = snapshot
            self._retain_terminal_entries()
            return snapshot

    async def _settle_outcome(
        self,
        entry: _MemoryTaskEntry,
        outcome: TaskOutcome[MemoryGenerationResults] | None,
    ) -> MemoryGenerationTask:
        """把队列终态投影为领域终态，并执行终态后的独立副作用。"""

        if outcome is None:
            snapshot = entry.created.as_failed(
                "memory generation work record missing",
                finished_at=datetime.now(UTC),
            )
            await self._publish_terminal(entry, snapshot)
            return snapshot

        await self._publish_running_if_needed(entry, outcome)

        record = outcome.record
        snapshot = MemoryGenerationTask.from_outcome(
            entry.created,
            outcome,
            expose_terminal=True,
        )

        if record.state == WorkState.SUCCEEDED:
            results = list(outcome.result or ())
            # 队列成功已经锁定 chat/生成执行终态；settlement 和领域事件属于后续
            # finalize，失败只影响可观测副作用，不回写通用 work 状态。
            await self._publish_settlements(results)
        elif record.state != WorkState.CANCELLED:
            logger.error(
                "Memory generation failed: label=%s, err=%s",
                entry.work.label,
                snapshot.error,
            )
        await self._publish_terminal(entry, snapshot)
        return snapshot

    async def get_task(self, task_id: str) -> MemoryGenerationTask | None:
        """按任务标识查询当前只读领域快照。"""

        entry = self._entries.get(task_id)
        if entry is None:
            return None
        return await self._snapshot_entry(entry)

    async def list_tasks(self) -> list[MemoryGenerationTask]:
        """列出控制器当前仍保留的任务快照。"""

        snapshots: list[MemoryGenerationTask] = []
        for entry in self._entries.values():
            snapshots.append(await self._snapshot_entry(entry))
        return snapshots

    async def _snapshot_entry(self, entry: _MemoryTaskEntry) -> MemoryGenerationTask:
        """从终态缓存或最新 ``WorkRecord`` 构造领域快照。"""

        if entry.final_snapshot is not None:
            return entry.final_snapshot
        if entry.handle is None:
            return entry.created
        outcome = await entry.handle.snapshot()
        if outcome is None:
            return entry.created.as_failed(
                "memory generation work record missing",
            )
        return MemoryGenerationTask.from_outcome(
            entry.created,
            outcome,
            expose_terminal=False,
        )

    async def wait_task(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> MemoryGenerationTaskWaitResult:
        """等待指定记忆生成任务完成，不取消或接管其执行。"""

        entry = self._entries.get(task_id)
        if entry is None:
            return MemoryGenerationTaskWaitResult.not_found(task_id)
        if entry.final_snapshot is not None:
            return MemoryGenerationTaskWaitResult.from_task(entry.final_snapshot)
        if entry.finalizer is None:
            return MemoryGenerationTaskWaitResult.from_task(
                await self._snapshot_entry(entry)
            )
        try:
            if timeout is None:
                snapshot = await asyncio.shield(entry.finalizer)
            else:
                snapshot = await asyncio.wait_for(
                    asyncio.shield(entry.finalizer),
                    timeout=max(0.0, timeout),
                )
        except TimeoutError:
            return MemoryGenerationTaskWaitResult.from_task(
                await self._snapshot_entry(entry),
                timed_out=True,
            )
        return MemoryGenerationTaskWaitResult.from_task(snapshot)

    async def wait_many(
        self,
        task_ids: list[str],
        timeout: float | None = None,
    ) -> MemoryGenerationTaskWaitSummary:
        """在共享超时窗口内等待一批指定任务。"""

        if not task_ids:
            return MemoryGenerationTaskWaitSummary.from_results([])

        waiters = [
            asyncio.create_task(self.wait_task(task_id), name=f"memory_task_wait_{task_id[:8]}")
            for task_id in task_ids
        ]
        done, pending = await asyncio.wait(waiters, timeout=timeout)

        results: list[MemoryGenerationTaskWaitResult] = []
        for task_id, waiter in zip(task_ids, waiters):
            if waiter in done:
                results.append(waiter.result())
                continue
            waiter.cancel()
            entry = self._entries.get(task_id)
            if entry is None:
                results.append(MemoryGenerationTaskWaitResult.not_found(task_id))
            else:
                results.append(
                    MemoryGenerationTaskWaitResult.from_task(
                        await self._snapshot_entry(entry),
                        timed_out=True,
                    )
                )
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return MemoryGenerationTaskWaitSummary.from_results(results)

    async def wait_all(
        self,
        timeout: float | None = None,
    ) -> MemoryGenerationTaskWaitSummary:
        """等待调用时所有尚未进入领域终态的记忆生成任务。"""

        task_ids = [
            task_id
            for task_id, entry in self._entries.items()
            if entry.final_snapshot is None and entry.finalizer is not None
        ]
        return await self.wait_many(task_ids, timeout=timeout)

    async def cancel_task(
        self,
        task_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool:
        """请求取消指定任务，并发布被接纳的取消请求快照。"""

        entry = self._entries.get(task_id)
        if entry is None or entry.handle is None or entry.final_snapshot is not None:
            return False
        async with entry.lock():
            if entry.final_snapshot is not None:
                return False
            accepted = await entry.handle.cancel(reason=reason)
            if accepted:
                current_outcome = await entry.handle.snapshot()
                if current_outcome is not None:
                    await self._publish_running_if_needed(entry, current_outcome)
                snapshot = (await self._snapshot_entry(entry)).with_cancel_request(reason)
                self._task_events.cancel_requested(
                    snapshot,
                    reason=reason,
                )
            return accepted

    async def cancel_many(
        self,
        task_ids: list[str],
        *,
        reason: str = "user_requested",
    ) -> int:
        """逐项请求取消并返回运行时接纳的数量。"""

        return sum([
            await self.cancel_task(task_id, reason=reason)
            for task_id in task_ids
        ])

    def _retain_terminal_entries(self) -> None:
        """按队列保留策略淘汰最早的领域终态及进程内结果。"""

        terminal_ids = [
            task_id
            for task_id, entry in self._entries.items()
            if entry.final_snapshot is not None
        ]
        excess = len(terminal_ids) - self._queue.terminal_retention
        for task_id in terminal_ids[: max(0, excess)]:
            entry = self._entries.pop(task_id)
            if entry.handle is not None:
                self._queue.release(entry.handle)
    
        async def _publish_running_if_needed(
        self,
        entry: _MemoryTaskEntry,
        outcome: TaskOutcome[MemoryGenerationResults],
    ) -> None:
        """在首次执行已经开始后至多发布一次运行中快照。"""

        if entry.running_published or outcome.record.started_at is None:
            return
        entry.running_published = True
        running = MemoryGenerationTask.from_outcome(
            entry.created,
            outcome,
            expose_terminal=False,
        )
        self._task_events.running(running)
        await self._publish_memory_task_item_status(running)

    async def _publish_terminal(
        self,
        entry: _MemoryTaskEntry,
        snapshot: MemoryGenerationTask,
    ) -> None:
        """以 best-effort 方式发布领域终态及关联 PendingAtom 事件。"""

        pending_alias = entry.work.pending_alias
        if pending_alias is not None:
            if snapshot.status == MemoryGenerationTaskStatus.CANCELLED:
                await self._publish_best_effort(
                    self._publish_pending_atom_cancelled(pending_alias),
                    f"pending atom cancel publish failed: {pending_alias}",
                )
            elif snapshot.status == MemoryGenerationTaskStatus.FAILED:
                await self._publish_best_effort(
                    self._publish_pending_atom_failed(pending_alias),
                    f"pending atom failed publish failed: {pending_alias}",
                )
        self._task_events.terminal(
            snapshot,
            reason=snapshot.cancel_reason,
        )
        await self._publish_best_effort(
            self._publish_memory_task_item_status(snapshot),
            f"memory task terminal publish failed: {snapshot.task_id}",
        )

    async def _publish_settlements(
        self,
        results: list[MemoryGenerationResult],
    ) -> None:
        """发布主动链路的 ``PendingAtom`` settlement 事件。"""

        for result in results:
            if result.settlement is None:
                continue
            try:
                await self._bus.publish(
                    PatchouliLocalEvents.PENDING_ATOM_SETTLED,
                    settlement=result.settlement,
                )
            except Exception as error:
                logger.warning(
                    "Settlement publish failed for %s: %s",
                    result.settlement.pending_alias,
                    error,
                )
                await self._publish_pending_atom_failed(
                    result.settlement.pending_alias
                )

    async def _publish_pending_atom_cancelled(self, pending_alias: str) -> None:
        """发布主动链路 ``PendingAtom`` 取消事件。"""

        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
                pending_alias=pending_alias,
            )
        except Exception as error:
            logger.warning("CANCELLED event publish error: %s", error)

    async def _publish_pending_atom_failed(self, pending_alias: str) -> None:
        """发布主动链路 ``PendingAtom`` 失败事件。"""

        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_FAILED,
                pending_alias=pending_alias,
            )
        except Exception as error:
            logger.warning("FAILED event publish error: %s", error)

    async def _publish_memory_task_item_status(
        self,
        snapshot: MemoryGenerationTask,
    ) -> None:
        """向 Patchouli 本地功能总线发布任务状态快照。"""

        try:
            await self._bus.publish(
                PatchouliLocalEvents.MEMORY_TASK_ITEM_STATUS,
                task_id=snapshot.task_id,
                pending_alias=snapshot.pending_alias,
                status=snapshot.status.value,
                canonical_alias=snapshot.canonical_alias,
            )
        except Exception as error:
            logger.warning("MEMORY_TASK_ITEM_STATUS publish error: %s", error)

    async def _publish_best_effort(self, awaitable, warning: str) -> None:
        """执行可观测副作用；失败只记录日志，不改变任务终态。"""

        try:
            await awaitable
        except asyncio.CancelledError:
            logger.warning(warning, exc_info=True)
        except Exception:
            logger.warning(warning, exc_info=True)

__all__ = ["MemoryGenerationTaskController"]
