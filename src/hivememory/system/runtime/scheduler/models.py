from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class MaintenanceTaskSpec:
    """维护任务注册规格 — 含 owner 维度的全局唯一任务描述。"""

    owner: str
    name: str
    interval_seconds: float
    enabled: bool = True
    non_reentrant: bool = True
    skip_if_running: bool = True
    jitter_seconds: float = 0.0

    @property
    def task_key(self) -> str:
        return f"{self.owner}.{self.name}"


@dataclass
class TaskRuntimeState:
    """单个任务的运行时状态（调度器内部使用）。"""

    spec: MaintenanceTaskSpec
    callback: Callable[[], Awaitable[Any]]
    next_run_at: float = 0.0
    last_started_at: Optional[float] = None
    last_finished_at: Optional[float] = None
    last_error: Optional[str] = None
    run_count: int = 0
    failure_count: int = 0
    skip_count: int = 0
    current_task: Optional[asyncio.Task] = field(default=None, repr=False)
