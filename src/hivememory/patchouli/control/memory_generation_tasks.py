"""记忆生成任务控制器。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation_queue import (
    MemoryGenerationQueue,
)
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationResult,
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
    MemoryGenerationTaskWaitResult,
    MemoryGenerationTaskWaitSummary,
    memory_task_to_payload,
)
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink
from hivememory.system.runtime.work_queue import (
    TERMINAL_WORK_STATES,
    QueuePolicy,
    WorkQueueShutdownSummary,
    WorkRecord,
    WorkState,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset(
    {
        MemoryGenerationTaskStatus.COMPLETED,
        MemoryGenerationTaskStatus.CANCELLED,
        MemoryGenerationTaskStatus.FAILED,
    }
)


class MemoryGenerationTaskController:
    """Patchouli 记忆生成任务控制面。"""

    def __init__(
        self,
        *,
        bus: Any,
        task_registry: Optional[MemoryGenerationTaskRegistry] = None,
        runtime_events: RuntimeEventSink | None = None,
        memory_queue: MemoryGenerationQueue | None = None,
        queue_policy: QueuePolicy | None = None,
    ) -> None:
        self._bus = bus
        self._task_registry = task_registry or MemoryGenerationTaskRegistry()
        self._events = runtime_events or NullRuntimeEventSink()
        self._queue = memory_queue or MemoryGenerationQueue(
            self._execute_generation,
            runtime_events=self._events,
            policy=queue_policy,
        )
        self._work_ids: dict[str, str] = {}
        self._cancel_reasons: dict[str, str] = {}

    @property
    def queue(self) -> MemoryGenerationQueue:
        """访问 memory generation 业务队列。"""
        return self._queue

    async def start(self) -> None:
        """启动 memory generation lane。"""
        await self._queue.start()

    async def stop(self) -> WorkQueueShutdownSummary:
        """停止 memory generation lane。"""
        summary = await self._queue.stop()
        # queue 先确认 running work 的 drain/cancel，再给领域投影一个短窗口完成
        # 终态事件，避免随后卸载 local routes 时丢失取消或失败快照。
        await self.wait_all(timeout=0.5)
        return summary

    async def submit_generation(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationTask:
        """提交单个规范化生成任务。"""
        return await self._create_and_run_task(spec=spec)

    async def submit_generation_many(
        self,
        specs: List[MemoryGenerationTaskSpec],
    ) -> List[MemoryGenerationTask]:
        """提交多个规范化生成任务。"""
        memory_tasks: List[MemoryGenerationTask] = []
        for spec in specs:
            memory_tasks.append(await self.submit_generation(spec))
        return memory_tasks

    async def _create_and_run_task(
        self,
        *,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationTask:
        """创建领域任务句柄并提交到 memory generation lane。"""
        memory_task = MemoryGenerationTask(
            task_id=str(uuid.uuid4()),
            topic_id=spec.topic_id,
            label=spec.label,
            source=spec.source,
            pending_alias=spec.pending_alias,
        )
        self._task_registry.register(memory_task)
        self._emit_memory_task_event(
            RuntimeEventType.MEMORY_TASK_CREATED,
            memory_task,
            message="Memory generation task created.",
        )

        try:
            # Controller 可单独用于本地测试与嵌入式调用，因此保留幂等懒启动；
            # PatchouliSystem 的正常生命周期仍会显式启动该 lane。
            await self._queue.start()
            receipt = await self._queue.submit(memory_task.task_id, spec)
        except Exception as exc:
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.FAILED,
                error=str(exc),
                pending_alias=spec.pending_alias,
            )
            raise

        self._work_ids[memory_task.task_id] = receipt.work_id
        # 兼容字段只承载 WorkRecord -> 领域状态投影，不再执行业务生成流程。
        projection_task = asyncio.create_task(
            self._project_work(memory_task, spec, receipt.work_id),
            name=f"memory_task_projection_{memory_task.task_id[:8]}",
        )
        memory_task.attach_task(projection_task)
        return memory_task

    async def _execute_generation(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> List[MemoryGenerationResult]:
        """由 queue handler 调用现有生成数据面。"""
        return await self._bus.request(
            PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
            spec,
        )

    async def _project_work(
        self,
        memory_task: MemoryGenerationTask,
        spec: MemoryGenerationTaskSpec,
        work_id: str,
    ) -> None:
        """持续把 WorkRecord 真相投影到 MemoryGenerationTask。"""
        while True:
            record = await self._queue.get(work_id)
            if record is None:
                await self._finish_task(
                    memory_task,
                    MemoryGenerationTaskStatus.FAILED,
                    error="memory generation work record missing",
                    pending_alias=spec.pending_alias,
                )
                return

            await self._project_non_terminal_state(memory_task, record)
            if record.state not in TERMINAL_WORK_STATES:
                await asyncio.sleep(0.01)
                continue

            if record.state == WorkState.SUCCEEDED:
                results = list(self._queue.take_results(work_id) or ())
                self._backfill_memory_task_result(
                    memory_task,
                    results,
                    pending_alias=spec.pending_alias,
                )
                await self._publish_settlements(results)
                await self._finish_task(
                    memory_task,
                    MemoryGenerationTaskStatus.COMPLETED,
                )
                return

            if record.state == WorkState.CANCELLED:
                self._queue.take_error(work_id)
                reason = self._cancel_reasons.pop(
                    memory_task.task_id,
                    "runtime_cancelled",
                )
                await self._finish_task(
                    memory_task,
                    MemoryGenerationTaskStatus.CANCELLED,
                    pending_alias=spec.pending_alias,
                    reason=reason,
                )
                # 兼容既有本地调用：被取消任务的 _bg_task 仍表现为 cancelled；
                # 这里取消的只是投影观察器，业务执行已由 queue 完成终止确认。
                raise asyncio.CancelledError

            error = self._queue.take_error(work_id)
            if error is None and record.last_error is not None:
                error = record.last_error.message or record.last_error.error_class
            logger.error(
                "Memory generation failed: label=%s, err=%s",
                spec.label,
                error,
            )
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.FAILED,
                error=error or "memory generation work failed",
                pending_alias=spec.pending_alias,
            )
            return

    async def _project_non_terminal_state(
        self,
        memory_task: MemoryGenerationTask,
        record: WorkRecord,
    ) -> None:
        """映射 queued/retry-wait/running，并补齐可能错过的 running 快照。"""
        if memory_task.status in _TERMINAL_STATUSES:
            return
        if record.state == WorkState.RUNNING or (
            record.state in TERMINAL_WORK_STATES
            and record.started_at is not None
            and memory_task.started_at is None
        ):
            if memory_task.status != MemoryGenerationTaskStatus.RUNNING:
                await self._start_task(
                    memory_task,
                    started_at=record.started_at,
                )
            return
        if record.state in {WorkState.QUEUED, WorkState.RETRY_WAIT}:
            memory_task.status = MemoryGenerationTaskStatus.PENDING

    def _backfill_memory_task_result(
        self,
        memory_task: MemoryGenerationTask,
        results: List[MemoryGenerationResult],
        pending_alias: Optional[str] = None,
    ) -> None:
        """从生成结果中选择 canonical_alias 并写回运行时任务句柄。"""
        canonical_alias = self._select_canonical_alias(
            results,
            pending_alias=pending_alias,
        )
        if canonical_alias:
            memory_task.canonical_alias = canonical_alias

    def _select_canonical_alias(
        self,
        results: List[MemoryGenerationResult],
        pending_alias: Optional[str] = None,
    ) -> Optional[str]:
        """选择任务展示用 canonical_alias。"""
        candidates = results
        if pending_alias:
            matched = [
                result
                for result in results
                if result.pending_alias == pending_alias
                or (
                    result.settlement is not None
                    and result.settlement.pending_alias == pending_alias
                )
            ]
            if matched:
                candidates = matched

        for result in candidates:
            if result.settlement is not None and result.settlement.canonical_alias:
                return result.settlement.canonical_alias
            if result.canonical_alias:
                return result.canonical_alias
            if result.atom is not None:
                get_alias = getattr(result.atom, "get_alias", None)
                if callable(get_alias):
                    alias = get_alias()
                    if alias:
                        return alias
        return None

    async def _publish_settlements(
        self,
        results: List[MemoryGenerationResult],
    ) -> None:
        """发布主动链路的 PendingAtom settlement 事件。"""
        for result in results:
            if result.settlement is None:
                continue
            try:
                await self._bus.publish(
                    PatchouliLocalEvents.PENDING_ATOM_SETTLED,
                    settlement=result.settlement,
                )
            except Exception as pub_err:
                logger.warning(
                    f"Settlement publish failed for "
                    f"{result.settlement.pending_alias}: {pub_err}"
                )
                await self._publish_pending_atom_failed(
                    result.settlement.pending_alias
                )

    async def _publish_pending_atom_cancelled(self, pending_alias: str) -> None:
        """发布主动链路 PendingAtom 取消事件。"""
        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
                pending_alias=pending_alias,
            )
        except Exception as pub_err:
            logger.warning(f"CANCELLED event publish error: {pub_err}")

    async def _publish_pending_atom_failed(self, pending_alias: str) -> None:
        """发布主动链路 PendingAtom 失败事件。"""
        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_FAILED,
                pending_alias=pending_alias,
            )
        except Exception as pub_err:
            logger.warning(f"FAILED event publish error: {pub_err}")

    async def _publish_memory_task_status(
        self,
        memory_task: MemoryGenerationTask,
        *,
        reason: str | None = None,
    ) -> None:
        """发布 MemoryGenerationTask 实时状态快照。"""
        self._emit_memory_task_event(
            self._event_type_for_task_status(memory_task.status),
            memory_task,
            reason=reason,
        )
        try:
            await self._bus.publish(
                PatchouliLocalEvents.MEMORY_TASK_ITEM_STATUS,
                task_id=memory_task.task_id,
                pending_alias=memory_task.pending_alias,
                status=memory_task.status.value,
                canonical_alias=memory_task.canonical_alias,
            )
        except Exception as pub_err:
            logger.warning(f"MEMORY_TASK_ITEM_STATUS publish error: {pub_err}")

    async def _start_task(
        self,
        memory_task: MemoryGenerationTask,
        *,
        started_at: datetime | None = None,
    ) -> None:
        """
        进入 RUNNING 状态并发布运行中快照。

        唯一允许写入 RUNNING/started_at 的入口。若任务已处于终态，
        说明取消或失败已经先发生，不再回退到 running。
        """
        if memory_task.status in _TERMINAL_STATUSES:
            return
        memory_task.status = MemoryGenerationTaskStatus.RUNNING
        if memory_task.started_at is None:
            memory_task.started_at = started_at or datetime.now(timezone.utc)
        await self._publish_memory_task_status(memory_task)

    async def _finish_task(
        self,
        memory_task: MemoryGenerationTask,
        status: MemoryGenerationTaskStatus,
        *,
        error: str | None = None,
        pending_alias: str | None = None,
        reason: str | None = None,
    ) -> None:
        """
        完成 MemoryGenerationTask 的唯一终态入口。

        先落业务事实：status/error/finished_at；再 best-effort 发布
        PendingAtom 和 memory.task 终态事件；最后无论发布是否失败都保留终态快照。

        第一次终态调用胜出，后续调用 no-op。防止取消、异常和重复 cleanup 互相覆盖终态。
        """
        if memory_task._terminal_finish_started:
            return
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"Memory task finish status must be terminal: {status}")

        memory_task._terminal_finish_started = True
        memory_task.status = status
        if error is not None:
            memory_task.error = error
        if memory_task.finished_at is None:
            memory_task.finished_at = datetime.now(timezone.utc)

        try:
            # failed/cancelled 由任务终态发布；completed settlement 已由结果发布。
            if pending_alias is not None:
                if status == MemoryGenerationTaskStatus.CANCELLED:
                    await self._publish_best_effort(
                        self._publish_pending_atom_cancelled(pending_alias),
                        f"pending atom cancel publish failed: {pending_alias}",
                    )
                elif status == MemoryGenerationTaskStatus.FAILED:
                    await self._publish_best_effort(
                        self._publish_pending_atom_failed(pending_alias),
                        f"pending atom failed publish failed: {pending_alias}",
                    )

            await self._publish_best_effort(
                self._publish_memory_task_status(memory_task, reason=reason),
                f"memory task terminal publish failed: {memory_task.task_id}",
            )
        finally:
            self._task_registry.retain_terminal(memory_task.task_id)
            self._work_ids.pop(memory_task.task_id, None)

    async def _publish_best_effort(self, awaitable, warning: str) -> None:
        """发布可观测副作用；失败只记日志，不改变任务终态。"""
        try:
            await awaitable
        except asyncio.CancelledError:
            logger.warning(warning, exc_info=True)
        except Exception:
            logger.warning(warning, exc_info=True)

    def get_task(self, task_id: str) -> Optional[MemoryGenerationTask]:
        """按 task_id 查询运行时任务。"""
        return self._task_registry.get(task_id)

    def list_tasks(self) -> List[MemoryGenerationTask]:
        """列出当前 registry 中仍保留的任务。"""
        return self._task_registry.list_all()

    async def wait_task(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> MemoryGenerationTaskWaitResult:
        """等待指定记忆生成 work 完成，不取消或接管其执行。"""
        memory_task = self._task_registry.get(task_id)
        if memory_task is None:
            return MemoryGenerationTaskWaitResult.not_found(task_id)

        if memory_task.status in _TERMINAL_STATUSES:
            return MemoryGenerationTaskWaitResult.from_task(memory_task)

        work_id = self._work_ids.get(task_id)
        if work_id is None:
            return MemoryGenerationTaskWaitResult.from_task(memory_task)

        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + max(0.0, timeout)
        record = await self._queue.wait(work_id, timeout=timeout)
        if record is None or record.state not in TERMINAL_WORK_STATES:
            if record is not None:
                await self._project_non_terminal_state(memory_task, record)
            return MemoryGenerationTaskWaitResult.from_task(
                memory_task,
                timed_out=True,
            )

        projection_task = memory_task._bg_task
        if projection_task is not None and not projection_task.done():
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - loop.time())
            )
            try:
                if remaining is None:
                    await asyncio.shield(projection_task)
                else:
                    await asyncio.wait_for(
                        asyncio.shield(projection_task),
                        timeout=remaining,
                    )
            except TimeoutError:
                return MemoryGenerationTaskWaitResult.from_task(
                    memory_task,
                    timed_out=True,
                )
            except asyncio.CancelledError:
                if not projection_task.done():
                    raise

        return MemoryGenerationTaskWaitResult.from_task(memory_task)

    async def wait_many(
        self,
        task_ids: List[str],
        timeout: float | None = None,
    ) -> MemoryGenerationTaskWaitSummary:
        """等待一批指定后台记忆生成任务完成。"""
        if not task_ids:
            return MemoryGenerationTaskWaitSummary.from_results([])

        waiters = [
            asyncio.create_task(
                self.wait_task(task_id, timeout=None),
                name=f"memory_task_wait_{task_id[:8]}",
            )
            for task_id in task_ids
        ]
        done, pending = await asyncio.wait(waiters, timeout=timeout)

        results: List[MemoryGenerationTaskWaitResult] = []
        for task_id, waiter in zip(task_ids, waiters):
            if waiter in done:
                results.append(waiter.result())
            else:
                waiter.cancel()
                memory_task = self._task_registry.get(task_id)
                if memory_task is None:
                    results.append(
                        MemoryGenerationTaskWaitResult.not_found(task_id)
                    )
                else:
                    results.append(
                        MemoryGenerationTaskWaitResult.from_task(
                            memory_task,
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
        """等待调用瞬间所有仍未进入领域终态的记忆生成任务完成。"""
        task_ids = [
            memory_task.task_id
            for memory_task in self._task_registry.list_all()
            if memory_task.status not in _TERMINAL_STATUSES
            and memory_task.task_id in self._work_ids
        ]
        return await self.wait_many(task_ids, timeout=timeout)

    async def cancel_task(
        self,
        task_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool:
        """请求取消指定记忆生成任务。"""
        memory_task = self._task_registry.get(task_id)
        if memory_task is None or memory_task.status in _TERMINAL_STATUSES:
            return False
        work_id = self._work_ids.get(task_id)
        if work_id is None:
            return False

        already_requested = memory_task.cancelled
        memory_task.request_cancel()
        if not already_requested:
            # 先保存原因，避免 queued cancel 在 queue.cancel 返回前已被投影。
            self._cancel_reasons[task_id] = reason
        ok = await self._queue.cancel(work_id, reason=reason)
        if not ok and not already_requested:
            self._cancel_reasons.pop(task_id, None)
            memory_task.cancel_event.clear()
        if ok and not already_requested:
            self._emit_memory_task_event(
                RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED,
                memory_task,
                reason=reason,
            )
        return ok

    async def cancel_many(
        self,
        task_ids: List[str],
        *,
        reason: str = "user_requested",
    ) -> int:
        cancelled = 0
        for task_id in task_ids:
            if await self.cancel_task(task_id, reason=reason):
                cancelled += 1
        return cancelled

    def _emit_memory_task_event(
        self,
        event_type: RuntimeEventType,
        memory_task: MemoryGenerationTask,
        *,
        reason: str | None = None,
        message: str | None = None,
    ) -> None:
        payload = memory_task_to_payload(memory_task, reason=reason)
        self._events.emit(
            RuntimeEvent(
                event_type=event_type,
                task_type="background",
                task_id=memory_task.task_id,
                topic_id=memory_task.topic_id,
                status=memory_task.status.value,
                reason=reason,
                message=message,
                severity="error" if event_type == RuntimeEventType.MEMORY_TASK_FAILED else "info",
                data=payload,
            )
        )

    @staticmethod
    def _event_type_for_task_status(status: MemoryGenerationTaskStatus) -> RuntimeEventType:
        if status == MemoryGenerationTaskStatus.COMPLETED:
            return RuntimeEventType.MEMORY_TASK_COMPLETED
        if status == MemoryGenerationTaskStatus.CANCELLED:
            return RuntimeEventType.MEMORY_TASK_CANCELLED
        if status == MemoryGenerationTaskStatus.FAILED:
            return RuntimeEventType.MEMORY_TASK_FAILED
        return RuntimeEventType.MEMORY_TASK_STATUS


__all__ = ["MemoryGenerationTaskController"]
