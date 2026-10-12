"""能力层与任务进程共享的子线程派生端口，不暴露进程实现。"""

from dataclasses import dataclass
from typing import Protocol

from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation


@dataclass(frozen=True)
class DerivedThread:
    """进程派生结果；context 只交给 CPU 驱动绑定提交函数。"""

    access: WorkspaceAccessContext
    allowed_operations: frozenset[WorkspaceOperation]


class ChildThreadSpawner(Protocol):
    """actor API 交给进程的派生请求；进程关联由授权者读取。"""

    def derive_child_thread(
        self, *, access: WorkspaceAccessContext, process_id: str, agent_id: str
    ) -> DerivedThread:
        """确认主线程仍在 Actor 阶段，再派生、登记并读取被调用方快照。"""
        ...
