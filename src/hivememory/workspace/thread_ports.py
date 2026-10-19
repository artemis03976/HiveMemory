"""能力层交给任务进程的线程端口：一次解析执行线程，以及 CALL 子线程的派生。

端口只在 workspace 内部使用：操作入口与 actor API 依赖它，进程线程管理实现
它，组合根注入。放在能力层与进程之外，使能力层不必导入进程子包。
"""

from dataclasses import dataclass
from typing import Protocol

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models.agent import AgentProfile
from hivememory.workspace.contracts.threads import ExecutionThread


@dataclass(frozen=True, slots=True)
class ResolvedThread:
    """线程表对一个存活线程的一次解析结果：进程关联、线程标识与派发话题。

    只在 workspace 内部、一次操作的调用链内使用，不进入 CPU 可见的对象、
    记录或事件；身份仍只在线程 context 中，由授权者逐次授权时读取。
    """

    process_id: str
    thread_id: str
    # CPU 分配冻结的真实话题，主线程与 CALL 子线程共享，actor 请求不能指定。
    dispatch_topic_id: str


class ExecutionThreadTable(Protocol):
    """执行线程的唯一入口：解析与派生都只经过所属进程的线程表。"""

    def resolve(self, access: WorkspaceAccessContext) -> ResolvedThread:
        """同步判定线程存活并返回解析结果。

        线程已结束（凭据已撤销、线程已结束或进程已收到停止）以
        ``ThreadEndedError`` 拒绝；context 不对、线程或进程未登记属于接线
        缺陷，以 ``ScopeRequiredError`` 拒绝。
        """
        ...

    def derive_child(
        self, *, parent: WorkspaceAccessContext, profile: AgentProfile
    ) -> ExecutionThread:
        """重新判定调用方存活且是主线程，再以图纸的 Agent 派生、读取操作快照并登记。"""
        ...


__all__ = ["ExecutionThreadTable", "ResolvedThread"]
