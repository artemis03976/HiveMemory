"""任务进程的线程管理：线程表、线程对象，以及派生、结束与解析的唯一入口。

执行线程是任务进程在 Actor 阶段内的执行主体，回答“这一步是谁在执行”：它
恰好属于一个进程，共享进程的资源（驻留 Workspace 与 principal 随执行者的
工作 actor，派发话题在工作集中，会话属于会话服务），有自己的工作 actor 与
生命周期。每个进程以线程表（:class:`ProcessThreadTable`，即
``TaskProcess.threads``）登记自己的全部线程；共享的 :class:`ProcessThreads`
是派生、结束与解析的唯一入口，线程表与线程对象的写方法只由它调用。

线程的生命周期是“派生 → 运行 → 结束”，结束只是标记，没有“停止中”状态。三个
结束时点：子线程随 CALL 返回由 CPU 经操作入口结束；进程收到停止即视为全部
线程已结束，停止路径不需要通知线程表；Actor 阶段结束时全部兜底结束。存活
只有一个判定：线程未结束，且进程仍在运行。

主线程与子线程以同一种线程描述交给 CPU：描述由线程表在派生时生成，观测
标签取线程自己的身份，操作快照按该线程自己的工作 actor 读取。CPU 只拿到
不透明的线程句柄，线程表按对象身份核对句柄，句柄与描述都不含工作 actor。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from hivememory.core.access import SubActorAdmission, WorkingActor
from hivememory.core.errors import ScopeRequiredError, ThreadEndedError
from hivememory.core.models import ExecutionLabels
from hivememory.core.models.agent import AgentProfile
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts.threads import ExecutionThread, ThreadDescriptor, ThreadHandle
from hivememory.workspace.process.table import ProcessOutcome, ProcessTable
from hivememory.workspace.thread_ports import ResolvedThread

# TaskProcess 只做类型检查时导入：task_process 在运行时从本模块导入线程表，
# 反向的运行时导入会形成循环。
if TYPE_CHECKING:
    from hivememory.workspace.process.task_process import TaskProcess

#: 主线程的标识；子线程按派生顺序编号，不会与它重名。
MAIN_THREAD_ID = "main"


class ThreadState(str, Enum):
    """执行线程的状态：派生成功即运行，结束后不再恢复。"""

    RUNNING = "running"
    ENDED = "ended"


@dataclass(eq=False, kw_only=True)
class ProcessThread:
    """一个执行线程：可变的控制状态（状态）与登记时确定的句柄、工作 actor、父线程和描述。

    ``actor`` 是线程以之执行 actor 操作的工作 actor：主线程为进程的执行者，
    子线程为通过准入的被调用方；它只交给 actor API 授权，不进入线程描述或
    句柄。``handle`` 是交给 CPU 驱动的不透明句柄，线程表按对象身份核对它。
    ``parent_id`` 主线程为空、子线程为主线程的标识。线程对象按对象身份识别，
    不离开 :class:`ProcessThreads`。
    """

    descriptor: ThreadDescriptor
    handle: ThreadHandle
    actor: WorkingActor
    parent_id: str | None
    state: ThreadState = ThreadState.RUNNING

    @property
    def thread_id(self) -> str:
        """线程标识，由线程描述携带。"""
        return self.descriptor.thread_id

    @property
    def is_running(self) -> bool:
        """线程是否仍在运行；进程是否仍在运行另由进程记录判断。"""
        return self.state is ThreadState.RUNNING

    def mark_ended(self) -> None:
        """记录线程已结束；重复结束是空操作。"""
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

    只持有进程表、子工作 actor 的准入端口与操作授权者，不保存任何进程的
    状态，线程全部登记在所属进程的线程表中。它实现能力层的
    ``ExecutionThreadTable`` 端口，由组合根注入操作入口与 actor API；执行器
    经它派生主线程并在 Actor 阶段收尾时结束全部线程。
    """

    def __init__(
        self,
        table: ProcessTable,
        *,
        admission: SubActorAdmission,
        authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._table = table
        self._admission = admission
        self._authorizer = authorizer

    def derive_main(self, process: TaskProcess, profile: AgentProfile) -> ExecutionThread:
        """进入 Actor 阶段时以进程的执行者与图纸派生并登记唯一主线程，返回交给 CPU 的执行线程。

        主线程的观测标签是注册标签；操作快照按执行者读取，读取同时重新确认
        准入。读取失败时不登记，不留下任何状态。
        """
        if process.threads.snapshot():
            raise ScopeRequiredError("进程已经登记执行线程")
        return self._register(
            process,
            actor=process.record.actor,
            thread_id=MAIN_THREAD_ID,
            parent_id=None,
            labels=process.record.labels,
            profile=profile,
        )

    def derive_child(self, *, parent: ThreadHandle, profile: AgentProfile) -> ExecutionThread:
        """只接受存活的主线程派生；同步完成全部检查、准入、快照读取与登记。

        调用方在 Profile 冷读之后才到达这里，存活在此重新判定。被调用方的
        Workspace 准入经认证一侧的准入端口检查；子线程的观测标签取被调用方
        与注册时的 workspace。准入或快照读取失败时不登记，异常沿原类型传播。
        """
        process, caller = self._resolve(parent)
        if caller.parent_id is not None:
            raise ScopeRequiredError("只有主线程可以派生子线程")
        if not profile.agent_id:
            raise ValueError("CALL 目标 Profile 缺少 agent_id")
        child_actor = self._admission.admit_sub_actor(caller.actor, profile.agent_id)
        labels = ExecutionLabels(
            agent_id=profile.agent_id,
            workspace_id=process.record.labels.workspace_id,
        )
        return self._register(
            process,
            actor=child_actor,
            thread_id=process.threads.next_child_id(),
            parent_id=caller.thread_id,
            labels=labels,
            profile=profile,
        )

    def resolve(self, thread: ThreadHandle) -> ResolvedThread:
        """唯一的存活判定：线程未结束且进程仍在运行时，返回一次解析结果。

        不检查阶段：线程只在 Actor 阶段内存在，Actor 阶段结束时全部结束。
        派发话题由 CPU 分配在派生主线程之前冻结，缺失属于接线缺陷。
        """
        process, registered = self._resolve(thread)
        topic_id = process.working_set.dispatch_topic_id
        if not topic_id:
            raise ScopeRequiredError("执行线程所属进程没有派发话题")
        return ResolvedThread(
            process_id=process.record.process_id,
            thread_id=registered.thread_id,
            dispatch_topic_id=topic_id,
            actor=registered.actor,
        )

    def end_child(self, thread: ThreadHandle) -> None:
        """CPU 在 CALL 收尾时结束子线程：同步标记结束，不检查进程是否仍在运行。

        已结束的线程（含所属进程已经关闭）是空操作；主线程只由 Actor 阶段收尾
        结束，CPU 结束主线程或句柄不是登记的那一个都属于接线缺陷。
        """
        try:
            _process, registered = self._locate(thread)
        except ThreadEndedError:
            return
        if registered.parent_id is None:
            raise ScopeRequiredError("主线程只由任务进程结束")
        registered.mark_ended()

    def end_all(self, process: TaskProcess) -> None:
        """结束进程的全部线程：Actor 阶段收尾与关闭兜底共用，同步且幂等。"""
        for thread in process.threads.snapshot():
            thread.mark_ended()

    def _register(
        self,
        process: TaskProcess,
        *,
        actor: WorkingActor,
        thread_id: str,
        parent_id: str | None,
        labels: ExecutionLabels,
        profile: AgentProfile,
    ) -> ExecutionThread:
        """按线程的工作 actor 读取操作快照，生成描述与句柄并登记；登记是最后一步。"""
        descriptor = ThreadDescriptor(
            thread_id=thread_id,
            labels=labels,
            agent_profile=profile,
            allowed_operations=self._authorizer.allowed_operations(actor),
        )
        handle = ThreadHandle(process_id=process.record.process_id, thread_id=thread_id)
        process.threads.register(
            ProcessThread(descriptor=descriptor, handle=handle, actor=actor, parent_id=parent_id)
        )
        return ExecutionThread(descriptor=descriptor, handle=handle)

    def _resolve(self, thread: ThreadHandle) -> tuple[TaskProcess, ProcessThread]:
        """依次判定登记与存活：线程已结束或进程已不在运行，都以 ``ThreadEndedError`` 拒绝。"""
        process, registered = self._locate(thread)
        if not registered.is_running:
            raise ThreadEndedError("Thread has ended")
        if process.record.outcome is not ProcessOutcome.RUNNING:
            raise ThreadEndedError("Process is no longer running")
        return process, registered

    def _locate(self, thread: ThreadHandle) -> tuple[TaskProcess, ProcessThread]:
        """按句柄的标识找到所属进程与线程，并要求登记的正是这一个句柄。

        句柄只由本类在登记线程时创建，所属进程已不在进程表中说明进程已经关闭，
        线程随之结束，以 ``ThreadEndedError`` 拒绝。进程仍在而线程未登记、或登记
        的不是这一个句柄（例如按标识另造的句柄），属于接线缺陷，以
        ``ScopeRequiredError`` 拒绝。
        """
        process = self._table.get(thread.process_id)
        if process is None:
            raise ThreadEndedError("Process has closed")
        registered = process.threads.get(thread.thread_id)
        if registered is None or registered.handle is not thread:
            raise ScopeRequiredError("执行线程未登记在所属进程的线程表中")
        return process, registered


__all__ = [
    "MAIN_THREAD_ID",
    "ProcessThread",
    "ProcessThreadTable",
    "ProcessThreads",
    "ThreadState",
]
