"""能力层交给任务进程的线程端口：解析执行线程、取得派发话题，以及 CALL 子线程的派生与结束。

端口只在 workspace 内部使用：操作入口与 actor API 依赖它，进程线程管理实现
它，组合根注入。放在能力层与进程之外，使能力层不必导入进程子包。
"""

from dataclasses import dataclass
from typing import Protocol

from hivememory.core.access import WorkingActor
from hivememory.core.models.agent import AgentProfile
from hivememory.workspace.contracts.threads import ExecutionThread, ThreadHandle


@dataclass(frozen=True, slots=True)
class ResolvedThread:
    """线程表对一个存活线程的一次解析结果：进程关联、线程标识与工作 actor。

    只在 workspace 内部、一次操作的调用链内使用，不进入 CPU 可见的对象、
    记录或事件。``actor`` 是线程记录持有的工作 actor，由 actor API 交给
    授权者逐次授权。派发话题只有写入用到，经 :meth:`ExecutionThreadTable.dispatch_topic`
    另行取得。
    """

    process_id: str
    thread_id: str
    actor: WorkingActor


class ExecutionThreadTable(Protocol):
    """执行线程的唯一入口：解析、派生与结束都只经过所属进程的线程表。"""

    def resolve(self, thread: ThreadHandle) -> ResolvedThread:
        """同步判定线程存活并返回解析结果。

        线程已结束、进程已收到停止或已经关闭，以 ``ThreadEndedError`` 拒绝；
        句柄不是登记的那一个，属于接线缺陷，以 ``ScopeRequiredError`` 拒绝。
        """
        ...

    async def dispatch_topic(self, thread: ThreadHandle) -> str:
        """取得线程所属进程这一轮的派发话题；还没有时为这一轮确定话题。

        先判定线程存活，拒绝方式与 :meth:`resolve` 相同。派发话题由进程的
        主线程与 CALL 子线程共享，同一进程只确定一次；actor 请求不能指定它。
        确定话题失败时错误交回这次调用，下一次调用重新确定。
        """
        ...

    def derive_child(self, *, parent: ThreadHandle, profile: AgentProfile) -> ExecutionThread:
        """重新判定调用方存活且是主线程，再为图纸的 Agent 做准入、读取操作快照并登记。"""
        ...

    def end_child(self, thread: ThreadHandle) -> None:
        """同步结束一个子线程；已结束的线程是空操作，主线程以接线缺陷拒绝。"""
        ...


__all__ = ["ExecutionThreadTable", "ResolvedThread"]
