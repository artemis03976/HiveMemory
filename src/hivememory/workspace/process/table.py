"""任务进程表 — 进程记录与唯一的进程注册表。

进程表是 workspace 进程内共享设施，也是唯一的进程注册表：以
``process_id`` 为键登记任务进程（``TaskProcess`` 容器），进程记录作为
进程的控制面经进程取得，不单独登记。
注册、注销与进程控制授权由注册入口（``workspace.process.service``）负责，
阶段推进与终态记录由编排骨架（``workspace.process.runner``）负责；
进程记录只持有访问 context、观测标签与进程自身的元数据，不保存身份字段。
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
    from hivememory.core.models import ExecutionLabels
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


#: System 关闭时经进程记录送达的停止原因。
SYSTEM_SHUTDOWN_STOP_REASON = "system_shutdown"

#: 交付方离开时的停止原因：流式交付被关闭，或响应在交付开始前就结束。
CLIENT_DISCONNECTED_STOP_REASON = "client_disconnected"

#: 不是用户决定的停止：交付方已离开或系统关闭，部分输出不写入会话、不提交。
DISCARDING_STOP_REASONS = frozenset({CLIENT_DISCONNECTED_STOP_REASON, SYSTEM_SHUTDOWN_STOP_REASON})


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


@dataclass
class ProcessRecord:
    """一次任务进程的阶段引用、访问 context 与终态事实（与进程同寿）。

    ``process_id`` 是任意任务进程的唯一标识：由 server 入口在进入
    编排服务前生成并冻结，进程表以它为稳定键，进程内不保存第二份生成事实。
    ``access`` 是注册入口经两阶段认证取得、绑定本进程的访问 context：它是
    记录持有的唯一身份凭据，只交给操作授权者用于授权与控制比对，进程以
    任何结局关闭时由注册入口使其失效。记录不保存 actor、驻留
    workspace 等身份字段：阶段调用的目标取自任务参数。``labels`` 是注册时
    用通过认证的声明创建一次的观测标签，供事件、会话记录、CPU 输入清单与
    线程描述使用，不参与授权；``events`` 是记录与事件发布器之间唯一的持有
    关系，发布器不回指记录。
    """

    process_id: str
    access: WorkspaceAccessContext
    labels: ExecutionLabels
    events: BoundProcessEvents
    phase: ProcessPhase = ProcessPhase.CREATED
    outcome: ProcessOutcome = ProcessOutcome.RUNNING
    stop_reason: str | None = None
    active_task: asyncio.Task[object] | None = None

    @property
    def is_terminal(self) -> bool:
        """进程是否已记录终态（completed / cancelled / failed）。

        编排骨架每次记录终态后紧接着写入终态产出，中间没有会挂起的
        ``await``，因此它同时表示“终态产出已经写入”；交付方离开时据此判断
        是否还需要请求停止，关闭流程据此判断是否需要终态兜底。
        """
        return self.phase is ProcessPhase.TERMINAL

    @property
    def discards_output(self) -> bool:
        """已接受的停止是否不由用户决定；此时已产生的输出既不写会话也不提交。"""
        return (
            self.outcome is ProcessOutcome.STOP_REQUESTED
            and self.stop_reason in DISCARDING_STOP_REASONS
        )

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
        """同步进入 finalize；用户停止不阻止结算已产生的输出，已收口或丢弃时拒绝。"""
        if self.outcome is ProcessOutcome.CANCELLED or self.discards_output:
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
        """同步记录 stop，并取消当前唯一的可中断阶段 task。

        判定顺序：进入 finalize 后一律拒绝；已经接受过的 stop 返回首次原因
        （含已以取消结束的进程）；其他已终态化的进程拒绝。
        """
        if self.phase is ProcessPhase.FINALIZE:
            return StopResult(accepted=False, reason="already_finalizing")
        if self.outcome in {ProcessOutcome.STOP_REQUESTED, ProcessOutcome.CANCELLED}:
            return StopResult(
                accepted=True,
                reason=self.stop_reason or reason,
            )
        if self.phase is ProcessPhase.TERMINAL:
            return StopResult(accepted=False, reason="already_terminal")

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

    def snapshot(self) -> tuple[TaskProcess, ...]:
        """关闭入口遍历当前登记；快照避免注销改变迭代集合。"""
        return tuple(self._processes.values())

    def get(self, process_id: str) -> TaskProcess | None:
        """按 ``process_id`` 原样取回已登记的任务进程；找不到返回 ``None``。"""
        return self._processes.get(process_id)

    def close(self, process: TaskProcess) -> None:
        """注销任务进程；只移除正是该对象的登记，不误删同 id 的其他进程。"""
        process_id = process.record.process_id
        if self._processes.get(process_id) is process:
            self._processes.pop(process_id, None)


__all__ = [
    "CLIENT_DISCONNECTED_STOP_REASON",
    "CancelResult",
    "ProcessOutcome",
    "ProcessPhase",
    "ProcessRecord",
    "ProcessTable",
    "SYSTEM_SHUTDOWN_STOP_REASON",
    "StopResult",
]
