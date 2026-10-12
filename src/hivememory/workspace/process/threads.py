"""任务进程的线程管理：主线程与 CALL 子线程的派生、登记和统一撤销。"""

from hivememory.core.access import ThreadContextLifecycle, WorkspaceAccessContext
from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models.agent import AgentProfile
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts.threads import ExecutionThread
from hivememory.workspace.process.table import ProcessOutcome, ProcessPhase, ProcessTable
from hivememory.workspace.process.task_process import TaskProcess
from hivememory.workspace.process.working_set import ProcessWorkingSet


class ProcessThreads:
    """共享线程管理器；全部运行状态只登记在所属进程的工作集。"""

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

    def derive_main_thread(self, process: TaskProcess) -> WorkspaceAccessContext:
        """进入 Actor 阶段时派生并登记唯一主线程。"""
        if process.working_set.thread_contexts:
            raise ScopeRequiredError("进程已经登记执行线程")
        access = self._lifecycle.derive_thread_context(process.record.access)
        process.working_set.thread_contexts["main"] = access
        return access

    def derive_child_thread(
        self, *, access: WorkspaceAccessContext, process_id: str, profile: AgentProfile
    ) -> ExecutionThread:
        """只接受 Actor 阶段主线程派生；同步完成全部检查与登记。

        context 按对象身份比较，不读取密封内容。即使 Profile 解析曾经
        等待，回到这里时也必须重新满足阶段与主线程存活条件。
        """
        process = self._table.get(process_id)
        if (
            process is None
            or process.record.phase is not ProcessPhase.ACTOR
            or process.record.outcome is not ProcessOutcome.RUNNING
            or process.working_set.thread_contexts.get("main") is not access
        ):
            raise ScopeRequiredError("只有 Actor 阶段的本进程主线程可以派生子线程")
        self._authorizer.validate_thread_context(access)
        if not profile.agent_id:
            raise ValueError("CALL 目标 Profile 缺少 agent_id")
        working_set = process.working_set
        thread_id = f"child-{len(working_set.thread_contexts)}"
        child = self._lifecycle.derive_child_thread_context(
            process.record.access, agent_id=profile.agent_id, thread_id=thread_id
        )
        try:
            operations = self._authorizer.get_stage_operations(
                child, process.request.target_workspace
            )
        except BaseException:
            # 快照读取失败也不留下有效凭据或登记；异常沿原类型传播。
            self._lifecycle.revoke_thread_context(child)
            raise
        working_set.thread_contexts[thread_id] = child
        return ExecutionThread(agent_profile=profile, allowed_operations=operations, access=child)

    def revoke_threads(self, working_set: ProcessWorkingSet) -> None:
        """CPU 结束与关闭兜底共用同步撤销，不等待任何外部操作。"""
        for access in working_set.take_thread_contexts():
            self._lifecycle.revoke_thread_context(access)
