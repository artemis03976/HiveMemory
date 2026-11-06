"""执行线程契约：线程描述交给 CPU，线程句柄只经操作入口交给 CPU 驱动。

执行线程是任务进程在 Actor 阶段内的执行主体，回答“这一步是谁在执行”。
主线程与 CALL 子线程以同一种形态交给 CPU：线程描述回答“谁来处理”，输入
清单回答“处理什么”。
"""

from dataclasses import dataclass

from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import ExecutionLabels
from hivememory.core.models.agent import AgentProfile


@dataclass(frozen=True)
class ThreadDescriptor:
    """执行线程的描述：标识、观测标签、图纸与操作快照，CPU 可见。

    由线程表在派生时生成。``labels`` 取线程自己的身份：主线程为注册标签，
    子线程为被调用方与注册时的 workspace。``allowed_operations`` 是派生时按
    该线程的工作 actor 读取的 actor 操作快照。标签与快照只用于观测、提示词与
    执行策略，不是授权依据；描述不含工作 actor。
    """

    thread_id: str
    labels: ExecutionLabels
    agent_profile: AgentProfile
    allowed_operations: frozenset[WorkspaceOperation]


@dataclass(frozen=True, eq=False)
class ThreadHandle:
    """执行线程的不透明句柄：只用于定位线程，不含身份。

    由线程表在登记线程时创建，按对象身份比较：线程表按 ``process_id`` 与
    ``thread_id`` 找到线程后，还要求登记的正是这一个句柄；标识相同的另一个
    句柄以接线缺陷拒绝。CPU 一侧因此不能凭标识构造句柄，冒用另一个线程
    （例如子线程冒用主线程）的身份与白名单提交。
    """

    process_id: str
    thread_id: str


@dataclass(frozen=True)
class ExecutionThread:
    """任务进程派生的执行线程：线程描述与它的线程句柄。

    主线程与子线程同用。只交给 CPU 驱动，由驱动把 ``handle`` 绑定为该线程的
    提交函数；句柄不进入 CPU 运行层、输入清单或任何记录。
    """

    descriptor: ThreadDescriptor
    handle: ThreadHandle


__all__ = ["ExecutionThread", "ThreadDescriptor", "ThreadHandle"]
