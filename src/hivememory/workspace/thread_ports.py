"""能力层交给任务进程的线程端口：CALL 子线程派生与写入意图的派发话题读取。

两个端口只在 workspace 内部使用：actor API 依赖它们，进程线程管理实现它们，
组合根注入。放在能力层与进程之外，使能力层不必导入进程子包。
"""

from typing import Protocol

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models.agent import AgentProfile
from hivememory.workspace.contracts.threads import ExecutionThread


class ChildThreadSpawner(Protocol):
    """actor API 交给进程的派生请求；进程关联由授权者读取。"""

    def derive_child_thread(
        self, *, access: WorkspaceAccessContext, process_id: str, profile: AgentProfile
    ) -> ExecutionThread:
        """确认主线程仍在 Actor 阶段，再以图纸的 Agent 派生、登记并读取其操作快照。"""
        ...


class IntentDispatchTopicReader(Protocol):
    """能力层读取本进程的派发话题；只接受仍在 Actor 阶段的已登记线程。"""

    def get_dispatch_topic(self, *, access: WorkspaceAccessContext, process_id: str) -> str:
        """同步校验进程与线程关联，返回 CPU 分配时冻结的真实话题 ID。"""
        ...


__all__ = ["ChildThreadSpawner", "IntentDispatchTopicReader"]
