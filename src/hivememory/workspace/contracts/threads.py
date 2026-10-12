"""执行线程契约：任务进程派生的执行线程，以及能力层交给进程的派生端口。"""

from dataclasses import dataclass
from typing import Protocol

from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.models.agent import AgentProfile


@dataclass(frozen=True)
class ExecutionThread:
    """任务进程派生的执行线程：以哪份图纸运行、能做什么，以及它的线程 context。

    只经操作入口交给 CPU 驱动，由驱动把 ``access`` 绑定为该线程的提交函数；
    不进入 Alice 运行层、输入清单或任何记录。
    """

    agent_profile: AgentProfile
    allowed_operations: frozenset[WorkspaceOperation]
    access: WorkspaceAccessContext


class ChildThreadSpawner(Protocol):
    """actor API 交给进程的派生请求；进程关联由授权者读取。"""

    def derive_child_thread(
        self, *, access: WorkspaceAccessContext, process_id: str, profile: AgentProfile
    ) -> ExecutionThread:
        """确认主线程仍在 Actor 阶段，再以图纸的 Agent 派生、登记并读取其操作快照。"""
        ...
