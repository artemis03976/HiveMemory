"""任务进程表 — 进程记录、进程内注册表与 stop API 控制面。

进程表是 workspace 进程内共享设施：以 ``process_id`` 为唯一标识登记
每个任务进程，只向同 owner/workspace 的控制请求暴露进程记录。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import (
    IdentityScope,
    require_identity_scope,
)


class ProcessPhase(str, Enum):
    """任务进程当前所在的编排阶段。"""

    CREATED = "created"
    GATEWAY = "gateway"
    PREPARE = "prepare"
    ALICE = "alice"
    FINALIZE = "finalize"
    TERMINAL = "terminal"


class ProcessOutcome(str, Enum):
    """任务进程的持久终态事实。"""

    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class StopResult:
    """一次 stop 请求的即时判定。"""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class CancelResult:
    """Stop API 对外保持的结构化结果。"""

    process_id: str
    cancelled: bool
    status: str
    reason: str


@dataclass(frozen=True)
class ProcessStatusSnapshot:
    """通过 scoped control plane 暴露的任务进程状态。"""

    process_id: str
    phase: str
    status: str
    reason: str | None


@dataclass
class ProcessRecord:
    """一次任务进程的阶段引用与终态事实。

    ``process_id`` 是任意任务进程的唯一标识（Q-16）：由 server 入口在进入
    编排服务前生成并冻结，进程表以它为稳定键，进程内不保存第二份生成事实。
    """

    identity_scope: IdentityScope
    process_id: str
    phase: ProcessPhase = ProcessPhase.CREATED
    outcome: ProcessOutcome = ProcessOutcome.RUNNING
    stop_reason: str | None = None
    active_task: asyncio.Task[object] | None = None

    def bind_phase(self, phase: ProcessPhase, task: asyncio.Task[object]) -> None:
        """绑定当前可被 stop 中断的阶段 task。"""
        self.phase = phase
        self.active_task = task

    def unbind_phase(self, task: asyncio.Task[object]) -> None:
        """仅按 task 身份解绑，避免旧 task 清空新阶段引用。"""
        if self.active_task is task:
            self.active_task = None

    def enter_phase(self, phase: ProcessPhase) -> None:
        """进入没有可中断 task 的阶段或阶段交接窗口。"""
        self.phase = phase
        self.active_task = None

    def try_enter_finalizing(self) -> bool:
        """同步进入 finalize；已接受 stop 时拒绝进入。"""
        if self.outcome in {ProcessOutcome.STOP_REQUESTED, ProcessOutcome.CANCELLED}:
            return False
        if self.phase is ProcessPhase.TERMINAL:
            return False
        self.phase = ProcessPhase.FINALIZE
        self.active_task = None
        return True

    def mark_cancelled(self) -> None:
        """记录进程级 cancelled 终态。"""
        self.outcome = ProcessOutcome.CANCELLED
        self.phase = ProcessPhase.TERMINAL
        self.active_task = None

    def mark_completed(self) -> None:
        """记录进程级 completed 终态。"""
        self.outcome = ProcessOutcome.COMPLETED
        self.phase = ProcessPhase.TERMINAL
        self.active_task = None

    def mark_failed(self) -> None:
        """记录进程级 failed 终态。"""
        self.outcome = ProcessOutcome.FAILED
        self.phase = ProcessPhase.TERMINAL
        self.active_task = None

    def request_stop(self, reason: str = "user_requested") -> StopResult:
        """同步记录 stop，并取消当前唯一的可中断阶段 task。"""
        if self.outcome in {ProcessOutcome.STOP_REQUESTED, ProcessOutcome.CANCELLED}:
            return StopResult(
                accepted=True,
                reason=self.stop_reason or reason,
            )

        if self.phase in {ProcessPhase.FINALIZE, ProcessPhase.TERMINAL}:
            return StopResult(
                accepted=False,
                reason=(
                    "already_finalizing"
                    if self.phase is ProcessPhase.FINALIZE
                    else "already_terminal"
                ),
            )

        self.outcome = ProcessOutcome.STOP_REQUESTED
        self.stop_reason = reason

        task = self.active_task
        if task is not None and not task.done():
            task.cancel()

        return StopResult(
            accepted=True,
            reason=reason,
        )


class ProcessTable:
    """进程内任务进程注册表与 stop API 控制面。"""

    def __init__(self) -> None:
        self._runs: dict[str, ProcessRecord] = {}

    def register(self, run: ProcessRecord) -> None:
        require_identity_scope(run.identity_scope)
        existing = self._runs.get(run.process_id)
        if existing is not None:
            raise WorkspaceDomainError(
                "process_id 已被注册，拒绝覆盖现有任务进程",
                details={"process_id": run.process_id},
            )
        self._runs[run.process_id] = run

    def get(
        self,
        process_id: str,
        identity_scope: IdentityScope,
    ) -> ProcessRecord | None:
        """只向同 owner/workspace 的控制请求暴露进程记录。"""
        identity_scope = require_identity_scope(identity_scope)
        run = self._runs.get(process_id)
        if run is None or not self._same_resource_scope(run.identity_scope, identity_scope):
            return None
        return run

    def cancel(
        self,
        process_id: str,
        identity_scope: IdentityScope,
        reason: str = "user_requested",
    ) -> CancelResult:
        run = self.get(process_id, identity_scope)
        if run is None:
            return CancelResult(
                process_id=process_id,
                cancelled=False,
                status="not_found",
                reason=reason,
            )

        result = run.request_stop(reason)
        return CancelResult(
            process_id=process_id,
            cancelled=result.accepted,
            status=run.outcome.value,
            reason=result.reason,
        )

    def status(
        self,
        process_id: str,
        identity_scope: IdentityScope,
    ) -> ProcessStatusSnapshot | None:
        """查询 scoped 状态；跨 scope 与不存在统一返回 ``None``。"""
        run = self.get(process_id, identity_scope)
        if run is None:
            return None
        return ProcessStatusSnapshot(
            process_id=run.process_id,
            phase=run.phase.value,
            status=run.outcome.value,
            reason=run.stop_reason,
        )

    def close(self, run: ProcessRecord) -> None:
        """移除已由进程编排记录终态的进程记录。"""
        if self._runs.get(run.process_id) is run:
            self._runs.pop(run.process_id, None)

    @staticmethod
    def _same_resource_scope(
        registered: IdentityScope,
        requested: IdentityScope,
    ) -> bool:
        return registered.workspace_identity == requested.workspace_identity


__all__ = [
    "CancelResult",
    "ProcessOutcome",
    "ProcessPhase",
    "ProcessRecord",
    "ProcessStatusSnapshot",
    "ProcessTable",
    "StopResult",
]
