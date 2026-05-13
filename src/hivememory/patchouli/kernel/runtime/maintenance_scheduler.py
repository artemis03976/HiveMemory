"""
SystemAsyncScheduler — 全局异步维护调度器

纯 asyncio 实现的系统级定时任务调度器，由 PatchouliSystem 唯一持有。
消除各组件自行维护 APScheduler BackgroundScheduler 导致的
跨线程跨事件循环风险。

职责边界:
    - 维护任务注册表与运行时状态
    - 在主 asyncio loop 中统一调度和执行维护任务
    - 非重入保护、异常隔离、关闭排空

    不负责: 各业务组件的 flush / GC / cleanup 具体逻辑

设计文档: docs/mod/PatchouliUnifiedMaintenanceSchedulerDesign.md
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ========== 任务协议与配置 ==========

@runtime_checkable
class MaintenanceTask(Protocol):
    """维护任务协议 — 业务组件实现此接口即可接入调度器"""

    @property
    def name(self) -> str: ...

    @property
    def interval_seconds(self) -> float: ...

    @property
    def enabled(self) -> bool: ...

    async def run_once(self) -> None: ...


@dataclass
class MaintenanceTaskSpec:
    """任务注册规格"""
    name: str
    interval_seconds: float
    enabled: bool = True
    non_reentrant: bool = True
    skip_if_running: bool = True
    jitter_seconds: float = 0.0


@dataclass
class TaskRuntimeState:
    """单个任务的运行时状态（调度器内部使用）"""
    spec: MaintenanceTaskSpec
    callback: Callable[[], Awaitable[Any]]
    next_run_at: float = 0.0
    last_started_at: Optional[float] = None
    last_finished_at: Optional[float] = None
    last_error: Optional[str] = None
    run_count: int = 0
    failure_count: int = 0
    skip_count: int = 0
    current_task: Optional[asyncio.Task] = None


# ========== 调度器 ==========

class SystemAsyncScheduler:
    """
    全局异步维护调度器

    运行在 PatchouliSystem 所在的主 asyncio loop 中，
    不创建额外线程或事件循环。
    """

    def __init__(self, tick_seconds: float = 1.0, shutdown_wait_seconds: float = 5.0):
        self._tick_seconds = tick_seconds
        self._shutdown_wait_seconds = shutdown_wait_seconds
        self._tasks: Dict[str, TaskRuntimeState] = {}
        self._shutdown = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None
        self._started = False

    # ========== 任务注册 ==========

    def register(
        self,
        spec: MaintenanceTaskSpec,
        callback: Callable[[], Awaitable[Any]],
    ) -> None:
        if spec.name in self._tasks:
            logger.warning(f"维护任务 '{spec.name}' 已注册，将覆盖")

        now = monotonic()
        jitter = random.uniform(0, spec.jitter_seconds) if spec.jitter_seconds > 0 else 0.0
        next_run = now + spec.interval_seconds + jitter

        self._tasks[spec.name] = TaskRuntimeState(
            spec=spec,
            callback=callback,
            next_run_at=next_run,
        )
        logger.info(
            f"维护任务已注册: name={spec.name}, "
            f"interval={spec.interval_seconds}s, enabled={spec.enabled}"
        )

    def unregister(self, name: str) -> bool:
        state = self._tasks.pop(name, None)
        if state is None:
            return False
        if state.current_task and not state.current_task.done():
            state.current_task.cancel()
        logger.info(f"维护任务已注销: {name}")
        return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        state = self._tasks.get(name)
        if state is None:
            return False
        state.spec.enabled = enabled
        logger.info(f"维护任务 '{name}' enabled={enabled}")
        return True

    # ========== 生命周期 ==========

    def start(self) -> None:
        if self._started:
            logger.warning("SystemAsyncScheduler 已在运行")
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "SystemAsyncScheduler.start() 必须在运行中的 asyncio 事件循环内调用"
            ) from exc
        self._shutdown.clear()
        self._loop_task = asyncio.create_task(self._run_loop())
        self._started = True
        logger.info(
            f"SystemAsyncScheduler 已启动: "
            f"tick={self._tick_seconds}s, tasks={len(self._tasks)}"
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._shutdown.set()

        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=self._shutdown_wait_seconds)
            except asyncio.TimeoutError:
                logger.warning("调度循环未在超时内退出，强制取消")
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
            logger.info(f"等待 {len(running_tasks)} 个运行中的维护任务完成...")
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
        logger.info("SystemAsyncScheduler 已停止")

    # ========== 调度循环 ==========

    async def _run_loop(self) -> None:
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
                logger.debug(f"维护任务 '{state.spec.name}' 仍在运行，跳过本轮 (skip_count={state.skip_count})")
                return

        state.next_run_at = now + state.spec.interval_seconds
        state.current_task = asyncio.create_task(
            self._execute_task(state),
            name=f"maintenance:{state.spec.name}",
        )

    async def _execute_task(self, state: TaskRuntimeState) -> None:
        name = state.spec.name
        start = monotonic()
        state.last_started_at = start
        state.run_count += 1
        try:
            await state.callback()
            elapsed = monotonic() - start
            state.last_finished_at = monotonic()
            state.last_error = None
            logger.debug(f"维护任务 '{name}' 完成: elapsed={elapsed:.3f}s")
        except asyncio.CancelledError:
            logger.info(f"维护任务 '{name}' 被取消")
            raise
        except Exception as e:
            elapsed = monotonic() - start
            state.failure_count += 1
            state.last_error = str(e)
            state.last_finished_at = monotonic()
            logger.error(
                f"维护任务 '{name}' 执行失败: {e} (elapsed={elapsed:.3f}s)",
                exc_info=True,
            )

    # ========== 内省 ==========

    def get_status(self) -> Dict[str, Any]:
        result = {}
        for name, state in self._tasks.items():
            running = state.current_task is not None and not state.current_task.done()
            result[name] = {
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


__all__ = [
    "MaintenanceTask",
    "MaintenanceTaskSpec",
    "TaskRuntimeState",
    "SystemAsyncScheduler",
]
