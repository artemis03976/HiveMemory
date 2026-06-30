"""
AsyncMaintenanceScheduler — 异步维护调度器基类

Pure asyncio maintenance scheduler serving as the foundation for all
system-level scheduled task execution. Does not create threads or
hidden event loops.

Responsibilities:
    - Task registry with owner-based partitioning
    - Tick-based scheduling in the main asyncio loop
    - Non-reentrant protection and exception isolation
    - Graceful shutdown with drain timeout
    - Introspection and runtime status queries
"""

from __future__ import annotations

import asyncio
import logging
import random
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, List, Optional

from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec, TaskRuntimeState

logger = logging.getLogger(__name__)


class AsyncMaintenanceScheduler:
    """
    Pure-asyncio maintenance scheduler base class.

    All system-layer schedulers inherit from this. The internal dict is
    keyed by `spec.task_key` (i.e. `{owner}.{name}`).
    """

    def __init__(
        self,
        tick_seconds: float = 1.0,
        shutdown_wait_seconds: float = 5.0,
        runtime_events: RuntimeEventSink | None = None,
    ):
        self._tick_seconds = tick_seconds
        self._shutdown_wait_seconds = shutdown_wait_seconds
        self._runtime_events = runtime_events or NullRuntimeEventSink()
        self._tasks: Dict[str, TaskRuntimeState] = {}
        self._shutdown: Optional[asyncio.Event] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._started = False

    # ========== Task Registration ==========

    def register(
        self,
        spec: MaintenanceTaskSpec,
        callback: Callable[[], Awaitable[Any]],
    ) -> None:
        key = spec.task_key
        if key in self._tasks:
            logger.warning(f"Maintenance task '{key}' already registered, overwriting")

        now = monotonic()
        jitter = random.uniform(0, spec.jitter_seconds) if spec.jitter_seconds > 0 else 0.0
        next_run = now + spec.interval_seconds + jitter

        self._tasks[key] = TaskRuntimeState(
            spec=spec,
            callback=callback,
            next_run_at=next_run,
        )
        logger.info(
            f"Maintenance task registered: key={key}, "
            f"interval={spec.interval_seconds}s, enabled={spec.enabled}"
        )

    def unregister(self, task_key: str) -> bool:
        state = self._tasks.pop(task_key, None)
        if state is None:
            return False
        if state.current_task and not state.current_task.done():
            state.current_task.cancel()
        logger.info(f"Maintenance task unregistered: {task_key}")
        return True

    def unregister_owner(self, owner: str) -> int:
        keys_to_remove = [k for k, v in self._tasks.items() if v.spec.owner == owner]
        for key in keys_to_remove:
            self.unregister(key)
        if keys_to_remove:
            logger.info(f"Unregistered {len(keys_to_remove)} task(s) for owner '{owner}'")
        return len(keys_to_remove)

    def set_enabled(self, task_key: str, enabled: bool) -> bool:
        state = self._tasks.get(task_key)
        if state is None:
            return False
        state.spec.enabled = enabled
        logger.info(f"Maintenance task '{task_key}' enabled={enabled}")
        return True

    def list_tasks(self) -> List[MaintenanceTaskSpec]:
        return [state.spec for state in self._tasks.values()]

    # ========== Lifecycle ==========

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
            f"AsyncMaintenanceScheduler started: "
            f"tick={self._tick_seconds}s, tasks={len(self._tasks)}"
        )

    async def stop(self) -> None:
        if not self._started:
            return
        assert self._shutdown is not None
        self._shutdown.set()

        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=self._shutdown_wait_seconds)
            except asyncio.TimeoutError:
                logger.warning("Scheduler loop did not exit within timeout, force cancelling")
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass

        running_tasks = [
            s.current_task for s in self._tasks.values()
            if s.current_task and not s.current_task.done()
        ]
        if running_tasks:
            logger.info(f"Waiting for {len(running_tasks)} running maintenance task(s) to finish...")
            done, pending = await asyncio.wait(
                running_tasks, timeout=self._shutdown_wait_seconds
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        self._started = False
        logger.info("AsyncMaintenanceScheduler stopped")

    # ========== Scheduling Loop ==========

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
            except asyncio.TimeoutError:
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
        self._emit_task_event(
            RuntimeEventType.MAINTENANCE_TASK_STARTED,
            state,
            status="started",
            duration_ms=0.0,
        )
        try:
            result = await state.callback()
            state.last_finished_at = monotonic()
            state.last_error = None
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
            self._emit_task_event(
                RuntimeEventType.MAINTENANCE_TASK_FAILED,
                state,
                status="failed",
                duration_ms=(state.last_finished_at - start) * 1000,
                error=str(e),
            )
            logger.error(
                f"Maintenance task '{key}' failed: {e}",
                exc_info=True,
            )

    def _emit_task_event(
        self,
        event_type: RuntimeEventType,
        state: TaskRuntimeState,
        *,
        status: str,
        duration_ms: float | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        # 维护任务事件是旁路观测事实，不驱动调度、重试或业务状态推进。
        spec = state.spec
        data = {
            "task_key": spec.task_key,
            "owner": spec.owner,
            "name": spec.name,
            "run_count": state.run_count,
            "failure_count": state.failure_count,
            "skip_count": state.skip_count,
            "duration_ms": duration_ms,
            "result": self._safe_event_value(result),
            "error": error,
        }
        self._runtime_events.emit(
            RuntimeEvent(
                event_type=event_type,
                task_type="background",
                source=spec.task_key,
                subsystem=self._subsystem_for_owner(spec.owner),
                component="maintenance_scheduler",
                severity="error" if event_type == RuntimeEventType.MAINTENANCE_TASK_FAILED else "info",
                status=status,
                reason=error,
                data=data,
            )
        )

    @staticmethod
    def _subsystem_for_owner(owner: str) -> str:
        if owner == "patchouli" or owner.startswith("patchouli."):
            return "patchouli"
        if owner == "alice" or owner.startswith("alice."):
            return "alice"
        return "system"

    @staticmethod
    def _safe_event_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(k): AsyncMaintenanceScheduler._safe_event_value(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                AsyncMaintenanceScheduler._safe_event_value(item)
                for item in value
            ]
        return repr(value)

    # ========== Introspection ==========

    def get_status(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
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
        return (
            f"{self.__class__.__name__}(running={self._started}, "
            f"tasks={len(self._tasks)})"
        )
