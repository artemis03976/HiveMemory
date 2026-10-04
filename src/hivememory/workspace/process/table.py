"""任务进程表 — 进程记录与唯一的进程注册表。

进程表是 workspace 进程内共享设施，也是唯一的进程注册表：以
``process_id`` 为键登记任务进程（``TaskProcess`` 容器），进程记录作为
进程的控制面经进程取得，不单独登记（任务进程 Idea 1.2 的 2026-10-04 注）。
注册、注销与进程控制授权由注册入口（``workspace.process.service``）负责；
进程记录只持有访问 context 与进程自身的元数据，不保存身份字段（A1 访问
边界返工第 4.4 节，I-8）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from hivememory.core.errors import WorkspaceDomainError

# TaskProcess 只做类型检查时导入：task_process 在运行时从本模块导入进程
# 记录与相关枚举，反向的运行时导入会形成循环。
if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.workspace.process.events import BoundProcessEvents
    from hivememory.workspace.process.task_process import TaskProcess


class ProcessPhase(str, Enum):
    """任务进程当前所在的编排阶段。"""

    CREATED = "created"
    GATEWAY = "gateway"
    PREPARE = "prepare"
    ACTOR = "actor"
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
    """一次任务进程的阶段引用、访问 context 与终态事实（与进程同寿）。

    ``process_id`` 是任意任务进程的唯一标识（Q-16）：由 server 入口在进入
    编排服务前生成并冻结，进程表以它为稳定键，进程内不保存第二份生成事实。
    ``access`` 是注册入口经两阶段认证取得、绑定本进程的访问 context：它是
    记录持有的唯一身份凭据，只交给操作授权者用于授权与控制比对，进程以
    任何结局关闭时由注册入口使其失效（P-6）。记录不保存 actor、驻留
    workspace 等身份字段：阶段调用的目标取自任务参数，观测标签（``events``）
    在注册时用通过认证的声明绑定一次（I-8 选项 C）；``events`` 是记录与
    事件发布器之间唯一的持有关系，发布器不回指记录。
    """

    process_id: str
    access: WorkspaceAccessContext
    events: BoundProcessEvents
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
    """进程内唯一的任务进程注册表：登记 ``process_id → TaskProcess``。

    登记与注销由注册入口负责；进程记录经 ``process.record`` 取得。本表
    不做 scope 过滤，控制请求的授权比对由注册入口经操作授权者完成。
    """

    def __init__(self) -> None:
        self._processes: dict[str, TaskProcess] = {}

    def register(self, process: TaskProcess) -> None:
        """登记任务进程；``process_id`` 已被占用时拒绝，不覆盖现有进程。"""
        process_id = process.record.process_id
        if process_id in self._processes:
            raise WorkspaceDomainError(
                "process_id 已被注册，拒绝覆盖现有任务进程",
                details={"process_id": process_id},
            )
        self._processes[process_id] = process

    def get(self, process_id: str) -> TaskProcess | None:
        """按 ``process_id`` 原样取回已登记的任务进程；找不到返回 ``None``。"""
        return self._processes.get(process_id)

    def close(self, process: TaskProcess) -> None:
        """注销任务进程；只移除正是该对象的登记，不误删同 id 的其他进程。"""
        process_id = process.record.process_id
        if self._processes.get(process_id) is process:
            self._processes.pop(process_id, None)


__all__ = [
    "CancelResult",
    "ProcessOutcome",
    "ProcessPhase",
    "ProcessRecord",
    "ProcessStatusSnapshot",
    "ProcessTable",
    "StopResult",
]
