"""任务进程的线程管理：线程表、线程对象，以及派生、结束与解析的唯一入口。

执行线程是任务进程在 Actor 阶段内的执行主体，回答“这一步是谁在执行”：它
恰好属于一个进程，共享进程的资源（驻留 Workspace 与 principal 在进程
context 中，派发话题在工作集中，会话属于会话服务），有自己的身份、凭据与
生命周期。每个进程以线程表（:class:`ProcessThreadTable`，即
``TaskProcess.threads``）登记自己的全部线程；共享的 :class:`ProcessThreads`
是派生、结束与解析的唯一入口，线程表与线程对象的写方法只由它调用。

线程的生命周期是“派生 → 运行 → 结束”，结束即同步撤销线程 context，没有
“停止中”状态。存活只有一个判定：线程未结束，且进程仍在运行。进程收到停止
即视为全部线程已结束，停止路径不需要通知线程表；凭据仍在 Actor 阶段收尾时
统一撤销，两者之间的提交按线程已结束拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from hivememory.core.access import ThreadContextLifecycle, WorkspaceAccessContext
from hivememory.core.errors import ScopeRequiredError, ThreadEndedError
from hivememory.core.models.agent import AgentProfile
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts.threads import ExecutionThread
from hivememory.workspace.process.table import ProcessOutcome, ProcessTable
from hivememory.workspace.thread_ports import ResolvedThread

# TaskProcess 只做类型检查时导入：task_process 在运行时从本模块导入线程表，
# 反向的运行时导入会形成循环。
if TYPE_CHECKING:
    from hivememory.workspace.process.task_process import TaskProcess

#: 主线程的标识；认证一侧派生主线程 context 时写入同一个值。
MAIN_THREAD_ID = "main"


class ThreadState(str, Enum):
    """执行线程的状态：派生成功即运行，结束后不再恢复。"""

    RUNNING = "running"
    ENDED = "ended"


@dataclass(eq=False, kw_only=True)
class ProcessThread:
    """一个执行线程：标识、父线程、线程 context 与状态。

    线程的身份只在线程 context 中，线程对象不另存 actor 字段。``parent_id``
    主线程为空、子线程为主线程的标识。线程对象按对象身份识别，不离开
    :class:`ProcessThreads`。
    """

    thread_id: str
    parent_id: str | None
    access: WorkspaceAccessContext
    state: ThreadState = ThreadState.RUNNING

    @property
    def is_running(self) -> bool:
        """线程是否仍在运行；进程是否仍在运行另由进程记录判断。"""
        return self.state is ThreadState.RUNNING

    def mark_ended(self) -> None:
        """记录线程已结束；凭据由 :class:`ProcessThreads` 先行撤销。"""
        self.state = ThreadState.ENDED


class ProcessThreadTable:
    """一个进程的线程表：按 ``thread_id`` 登记本进程派生成功的全部线程。

    随进程创建，创建时为空；只在 Actor 阶段内登记，随进程一起丢弃。已结束的
    线程留在表中直到进程关闭：标识不复用，已结束的线程仍可按标识识别。子线程
    标识按派生顺序分配（``child-1``、``child-2``……），计数器只在登记成功时
    前进，派生失败不留下任何痕迹。表只登记线程：进程的资源、CPU 一侧的执行
    状态与写入意图都不在这里。
    """

    def __init__(self) -> None:
        self._threads: dict[str, ProcessThread] = {}
        self._child_count = 0

    def get(self, thread_id: str) -> ProcessThread | None:
        """按标识取回已登记的线程（含已结束的线程）；未登记返回 ``None``。"""
        return self._threads.get(thread_id)

    def snapshot(self) -> tuple[ProcessThread, ...]:
        """按派生顺序返回全部已登记线程；快照避免遍历时登记变化。"""
        return tuple(self._threads.values())

    def next_child_id(self) -> str:
        """下一个子线程的标识；只读取，不消耗编号。"""
        return f"child-{self._child_count + 1}"

    def register(self, thread: ProcessThread) -> None:
        """登记一个派生成功的线程；标识已被占用时拒绝，不覆盖已有线程。"""
        if thread.thread_id in self._threads:
            raise ScopeRequiredError(
                "执行线程标识已被登记", details={"thread_id": thread.thread_id}
            )
        self._threads[thread.thread_id] = thread
        if thread.parent_id is not None:
            self._child_count += 1


class ProcessThreads:
    """共享线程管理器：派生、结束与解析执行线程的唯一入口。

    只持有进程表、线程 context 生命周期端口与操作授权者，不保存任何进程的
    状态，线程全部登记在所属进程的线程表中。它实现能力层的
    ``ExecutionThreadTable`` 端口，由组合根注入操作入口与 actor API；执行器
    经它派生主线程并在 Actor 阶段收尾时结束全部线程。
    """

    def __init__(
        self,
        table: ProcessTable,
        *,
        lifecycle: ThreadContextLifecycle,
        authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._table = table
        self._lifecycle = lifecycle
        self._authorizer = authorizer

    def derive_main(self, process: TaskProcess) -> WorkspaceAccessContext:
        """进入 Actor 阶段时派生并登记唯一主线程，返回交给 CPU 的线程 context。"""
        if process.threads.snapshot():
            raise ScopeRequiredError("进程已经登记执行线程")
        access = self._lifecycle.derive_thread_context(process.record.access)
        process.threads.register(
            ProcessThread(thread_id=MAIN_THREAD_ID, parent_id=None, access=access)
        )
        return access

    def derive_child(
        self, *, parent: WorkspaceAccessContext, profile: AgentProfile
    ) -> ExecutionThread:
        """只接受存活的主线程派生；同步完成全部检查、签发、快照读取与登记。

        调用方在 Profile 冷读之后才到达这里，存活在此重新判定。被调用方的
        Workspace 准入由认证一侧在签发时检查；签发之后的快照读取或登记失败时
        撤销刚签发的 context，不留下凭据或登记，异常沿原类型传播。
        """
        process, caller = self._resolve(parent)
        if caller.parent_id is not None:
            raise ScopeRequiredError("只有主线程可以派生子线程")
        if not profile.agent_id:
            raise ValueError("CALL 目标 Profile 缺少 agent_id")
        thread_id = process.threads.next_child_id()
        child = self._lifecycle.derive_child_thread_context(
            process.record.access, agent_id=profile.agent_id, thread_id=thread_id
        )
        try:
            operations = self._authorizer.get_stage_operations(
                child, process.request.target_workspace
            )
            process.threads.register(
                ProcessThread(thread_id=thread_id, parent_id=caller.thread_id, access=child)
            )
        except BaseException:
            self._lifecycle.revoke_thread_context(child)
            raise
        return ExecutionThread(agent_profile=profile, allowed_operations=operations, access=child)

    def resolve(self, access: WorkspaceAccessContext) -> ResolvedThread:
        """唯一的存活判定：线程未结束且进程仍在运行时，返回一次解析结果。

        不检查阶段：线程只在 Actor 阶段内存在，Actor 阶段结束时全部结束。
        派发话题由 CPU 分配在派生主线程之前冻结，缺失属于接线缺陷。
        """
        process, thread = self._resolve(access)
        topic_id = process.working_set.dispatch_topic_id
        if not topic_id:
            raise ScopeRequiredError("执行线程所属进程没有派发话题")
        return ResolvedThread(
            process_id=process.record.process_id,
            thread_id=thread.thread_id,
            dispatch_topic_id=topic_id,
        )

    def end_all(self, process: TaskProcess) -> None:
        """结束进程的全部线程：Actor 阶段收尾与关闭兜底共用，同步且幂等。"""
        for thread in process.threads.snapshot():
            self._end(thread)

    def _end(self, thread: ProcessThread) -> None:
        """先撤销凭据再记录结束，已结束的线程是空操作；不等待任何外部操作。"""
        if not thread.is_running:
            return
        self._lifecycle.revoke_thread_context(thread.access)
        thread.mark_ended()

    def _resolve(self, access: WorkspaceAccessContext) -> tuple[TaskProcess, ProcessThread]:
        """依次判定凭据、登记与存活；context 按对象身份比较，不读取密封内容。

        凭据已撤销（授权者拒绝）、线程已结束或进程已不在运行，都以
        ``ThreadEndedError`` 拒绝；不是线程 context、进程或线程未登记、登记的
        不是这一个 context，属于接线缺陷，以 ``ScopeRequiredError`` 拒绝。
        """
        binding = self._authorizer.get_thread_binding(access)
        process = self._table.get(binding.run_id)
        thread = None
        if process is not None and binding.thread_id is not None:
            thread = process.threads.get(binding.thread_id)
        if process is None or thread is None or thread.access is not access:
            raise ScopeRequiredError("执行线程未登记在所属进程的线程表中")
        if not thread.is_running:
            raise ThreadEndedError("Thread has ended")
        if process.record.outcome is not ProcessOutcome.RUNNING:
            raise ThreadEndedError("Process is no longer running")
        return process, thread


__all__ = [
    "MAIN_THREAD_ID",
    "ProcessThread",
    "ProcessThreadTable",
    "ProcessThreads",
    "ThreadState",
]
