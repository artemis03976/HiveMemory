"""记忆生成任务控制器。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
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
    ) -> None:
        self._bus = bus
        self._task_registry = task_registry or MemoryGenerationTaskRegistry()
        self._events = runtime_events or NullRuntimeEventSink()

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
        """创建运行时任务句柄并调度后台协程。"""
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

        bg_task = asyncio.create_task(
            self._run_task(memory_task, spec),
            name=f"memory_task_{memory_task.task_id[:8]}",
        )
        memory_task.attach_task(bg_task)
        return memory_task

    async def _run_task(
        self,
        memory_task: MemoryGenerationTask,
        spec: MemoryGenerationTaskSpec,
    ) -> None:
        """执行单个规范化生成任务的控制流程。"""
        if memory_task.cancelled:
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.CANCELLED,
                pending_alias=spec.pending_alias,
            )
            return

        await self._start_task(memory_task)
        try:
            results = await self._bus.request(
                PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
                spec,
            )
            self._backfill_memory_task_result(
                memory_task,
                results,
                pending_alias=spec.pending_alias,
            )
            await self._publish_settlements(results)
        except asyncio.CancelledError:
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.CANCELLED,
                pending_alias=spec.pending_alias,
            )
            raise
        except Exception as exc:
            logger.error(
                f"Memory generation failed: label={spec.label}, err={exc}",
                exc_info=True,
            )
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.FAILED,
                error=str(exc),
                pending_alias=spec.pending_alias,
            )
        else:
            await self._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)

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
    ) -> None:
        """发布 MemoryGenerationTask 实时状态快照。"""
        self._emit_memory_task_event(
            self._event_type_for_task_status(memory_task.status),
            memory_task,
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

    async def _start_task(self, memory_task: MemoryGenerationTask) -> None:
        """
        进入 RUNNING 状态并发布运行中快照。

        唯一允许写入 RUNNING/started_at 的入口。若任务已处于终态，
        说明取消或失败已经先发生，不再回退到 running。
        """
        if memory_task.status in _TERMINAL_STATUSES:
            return
        memory_task.status = MemoryGenerationTaskStatus.RUNNING
        if memory_task.started_at is None:
            memory_task.started_at = datetime.now(timezone.utc)
        await self._publish_memory_task_status(memory_task)

    async def _finish_task(
        self,
        memory_task: MemoryGenerationTask,
        status: MemoryGenerationTaskStatus,
        *,
        error: str | None = None,
        pending_alias: str | None = None,
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
                self._publish_memory_task_status(memory_task),
                f"memory task terminal publish failed: {memory_task.task_id}",
            )
        finally:
            self._task_registry.retain_terminal(memory_task.task_id)

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
        """等待指定后台记忆生成任务完成，不取消或接管后台任务。"""
        memory_task = self._task_registry.get(task_id)
        if memory_task is None:
            return MemoryGenerationTaskWaitResult.not_found(task_id)

        bg_task = memory_task._bg_task
        if bg_task is None or bg_task.done():
            return MemoryGenerationTaskWaitResult.from_task(memory_task)

        try:
            await asyncio.wait_for(asyncio.shield(bg_task), timeout=timeout)
        except TimeoutError:
            return MemoryGenerationTaskWaitResult.from_task(
                memory_task,
                timed_out=True,
            )
        except asyncio.CancelledError:
            if bg_task.done():
                return MemoryGenerationTaskWaitResult.from_task(memory_task)
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
        """等待调用瞬间所有仍在后台执行的记忆生成任务完成。"""
        task_ids = [
            memory_task.task_id
            for memory_task in self._task_registry.list_all()
            if memory_task._bg_task is not None and not memory_task._bg_task.done()
        ]
        return await self.wait_many(task_ids, timeout=timeout)

    async def cancel_task(self, task_id: str) -> bool:
        """请求取消指定记忆生成任务。"""
        memory_task = self._task_registry.get(task_id)
        ok = self._task_registry.cancel(task_id)
        if ok and memory_task is not None:
            self._emit_memory_task_event(
                RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED,
                memory_task,
                reason="user_requested",
            )
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.CANCELLED,
                pending_alias=memory_task.pending_alias,
            )
        return ok

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
