"""
AsyncMaintenanceScheduler — 异步维护调度器基类

纯 asyncio 维护调度器，是所有系统级定时任务执行的基座。不创建线程，
也不创建隐藏事件循环。

Responsibilities:
    - 任务注册表，按 owner 分区
    - 在主 asyncio 循环内按 tick 调度
    - 防重入保护与异常隔离
    - 带 drain 超时的优雅关停
    - 自省与运行时状态查询
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from hivememory.components.events.bus import NullRuntimeEventSink, safe_runtime_event_value
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.scheduler.models import (
    MaintenanceTaskSpec,
    TaskRuntimeState,
)
from hivememory.core.contracts.runtime_events import RuntimeEventType

logger = logging.getLogger(__name__)


class AsyncMaintenanceScheduler:
    """
    纯 asyncio 维护调度器基类。

    所有系统层调度器都继承自它。内部字典以 `spec.task_key`（即 `{owner}.{name}`）为键。
    """

    def __init__(
        self,
        tick_seconds: float = 1.0,
        shutdown_wait_seconds: float = 5.0,
        runtime_events: RuntimeEventPublisher | None = None,
    ):
        self._tick_seconds = tick_seconds
        self._shutdown_wait_seconds = shutdown_wait_seconds
        self._runtime_events = (
            runtime_events or RuntimeEventPublisher(NullRuntimeEventSink())
        ).scoped(component="maintenance_scheduler")
        self._tasks: dict[str, TaskRuntimeState] = {}
        self._shutdown: asyncio.Event | None = None
        self._loop_task: asyncio.Task | None = None
        self._started = False

    # ========== 任务注册 ==========

    def register(
        self,
        spec: MaintenanceTaskSpec,
        callback: Callable[[], Awaitable[Any]],
    ) -> None:
        """注册维护任务。

        ``callback`` 以返回值表明这一周期是否做了事：返回假值（None 或空容器）表示
        没有做事，调度器不发事件；其余返回值随 ``maintenance.task.completed`` 发出；
        抛出异常时发出 ``maintenance.task.failed``。
        """
        key = spec.task_key
        if key in self._tasks:
            logger.warning("Maintenance task '%s' already registered, overwriting", key)

        now = monotonic()
        jitter = random.uniform(0, spec.jitter_seconds) if spec.jitter_seconds > 0 else 0.0
        next_run = now + spec.interval_seconds + jitter

        self._tasks[key] = TaskRuntimeState(
            spec=spec,
            callback=callback,
            next_run_at=next_run,
        )
        logger.info(
            "Maintenance task registered: key=%s, interval=%ss, enabled=%s",
            key,
            spec.interval_seconds,
            spec.enabled,
        )

    def unregister(self, task_key: str) -> bool:
        state = self._tasks.pop(task_key, None)
        if state is None:
            return False
        if state.current_task and not state.current_task.done():
            state.current_task.cancel()
        logger.info("Maintenance task unregistered: %s", task_key)
        return True

    def unregister_owner(self, owner: str) -> int:
        keys_to_remove = [k for k, v in self._tasks.items() if v.spec.owner == owner]
        for key in keys_to_remove:
            self.unregister(key)
        if keys_to_remove:
            logger.info("Unregistered %s task(s) for owner '%s'", len(keys_to_remove), owner)
        return len(keys_to_remove)

    def set_enabled(self, task_key: str, enabled: bool) -> bool:
        state = self._tasks.get(task_key)
        if state is None:
            return False
        state.spec.enabled = enabled
        logger.info("Maintenance task '%s' enabled=%s", task_key, enabled)
        return True

    def list_tasks(self) -> list[MaintenanceTaskSpec]:
        return [state.spec for state in self._tasks.values()]

    # ========== 生命周期 ==========

    def start(self) -> None:
        if self._started:
            logger.warning("AsyncMaintenanceScheduler already running")
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "AsyncMaintenanceScheduler.start() must be called within a running asyncio event loop"
            ) from exc
        self._shutdown = asyncio.Event()
        self._loop_task = asyncio.create_task(self._run_loop())
        self._started = True
        logger.info(
            "AsyncMaintenanceScheduler started: tick=%ss, tasks=%s",
            self._tick_seconds,
            len(self._tasks),
        )

    async def stop(self) -> None:
        if not self._started:
            return
        assert self._shutdown is not None
        self._shutdown.set()

        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=self._shutdown_wait_seconds)
            except TimeoutError:
                logger.warning("Scheduler loop did not exit within timeout, force cancelling")
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass

        running_tasks = [
            s.current_task
            for s in self._tasks.values()
            if s.current_task and not s.current_task.done()
        ]
        if running_tasks:
            logger.info(
                "Waiting for %s running maintenance task(s) to finish...", len(running_tasks)
            )
            done, pending = await asyncio.wait(running_tasks, timeout=self._shutdown_wait_seconds)
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        self._started = False
        logger.info("AsyncMaintenanceScheduler stopped")

    # ========== 调度循环 ==========

    async def _run_loop(self) -> None:
        assert self._shutdown is not None
        while not self._shutdown.is_set():
            now = monotonic()
            for state in list(self._tasks.values()):
                if not state.spec.enabled:
                    continue
                if now >= state.next_run_at:
                    self._dispatch_task(state, now)

            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._tick_seconds,
                )
                break
            except TimeoutError:
                pass

    def _dispatch_task(self, state: TaskRuntimeState, now: float) -> None:
        if state.spec.non_reentrant and state.current_task and not state.current_task.done():
            if state.spec.skip_if_running:
                state.skip_count += 1
                state.next_run_at = now + state.spec.interval_seconds
                return

        state.next_run_at = now + state.spec.interval_seconds
        state.current_task = asyncio.create_task(
            self._execute_task(state),
            name=f"maintenance:{state.spec.task_key}",
        )

    async def _execute_task(self, state: TaskRuntimeState) -> None:
        key = state.spec.task_key
        start = monotonic()
        state.last_started_at = start
        state.run_count += 1
        try:
            result = await state.callback()
            state.last_finished_at = monotonic()
            state.last_error = None
            # 空转的周期不发事件：高频任务的心跳会把真正的状态转换挤出事件缓冲。
            if not result:
                return
            self._emit_task_event(
                RuntimeEventType.MAINTENANCE_TASK_COMPLETED,
                state,
                status="completed",
                duration_ms=(state.last_finished_at - start) * 1000,
                result=result,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            state.failure_count += 1
            state.last_error = str(e)
            state.last_finished_at = monotonic()
            # 事件只带原因码与异常类名，原始异常与堆栈只进日志。
            self._emit_task_event(
                RuntimeEventType.MAINTENANCE_TASK_FAILED,
                state,
                status="failed",
                duration_ms=(state.last_finished_at - start) * 1000,
                error_class=type(e).__name__,
            )
            logger.error("Maintenance task failed: task_key=%s", key, exc_info=True)

    def _emit_task_event(
        self,
        event_type: RuntimeEventType,
        state: TaskRuntimeState,
        *,
        status: str,
        duration_ms: float | None = None,
        result: Any = None,
        error_class: str | None = None,
    ) -> None:
        # 维护任务事件是旁路观测事实，不驱动调度、重试或业务状态推进。
        spec = state.spec
        failed = event_type == RuntimeEventType.MAINTENANCE_TASK_FAILED
        data = {
            "task_key": spec.task_key,
            "owner": spec.owner,
            "name": spec.name,
            "run_count": state.run_count,
            "failure_count": state.failure_count,
            "skip_count": state.skip_count,
            "duration_ms": duration_ms,
            "result": safe_runtime_event_value(result),
            "error_class": error_class,
        }
        self._runtime_events.scoped(
            subsystem=self._subsystem_for_owner(spec.owner), source=spec.task_key
        ).bind(task_type="background").emit(
            event_type,
            severity="error" if failed else "info",
            status=status,
            reason="maintenance_task_failed" if failed else None,
            data=data,
        )

    @staticmethod
    def _subsystem_for_owner(owner: str) -> str:
        if owner == "patchouli" or owner.startswith("patchouli."):
            return "patchouli"
        if owner == "alice" or owner.startswith("alice."):
            return "alice"
        return "system"

    # ========== 自省 ==========

    def get_status(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, state in self._tasks.items():
            running = state.current_task is not None and not state.current_task.done()
            result[key] = {
                "owner": state.spec.owner,
                "name": state.spec.name,
                "enabled": state.spec.enabled,
                "interval_seconds": state.spec.interval_seconds,
                "running": running,
                "run_count": state.run_count,
                "failure_count": state.failure_count,
                "skip_count": state.skip_count,
                "last_error": state.last_error,
            }
        return result

    @property
    def is_running(self) -> bool:
        return self._started

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(running={self._started}, " f"tasks={len(self._tasks)})"
