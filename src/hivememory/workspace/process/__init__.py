"""任务进程：进程表（注册与 stop 控制面）、进程工作集与 chat 任务进程的四阶段编排骨架。

进程表是进程内共享设施：以 ``process_id`` 为唯一标识登记任务进程，只向
同 owner/workspace 的控制请求暴露进程记录；编排服务经全局总线的公开路由
驱动 Gateway → Patchouli prepare → CPU 分配 → Alice run → finalize，不
直接持有任何子系统引用。CPU 分配（Profile 解析、附件租借、编译与输入
清单组装）由进程在 prepare 之后、进入 Alice 之前完成，进程工作集持有
本轮附件租借并随进程关闭统一释放。workspace 中 ``process`` 以外的模块
不得导入本子包。
"""

from hivememory.workspace.process.service import (
    NonStreamingChatAgentOutcome,
    NonStreamingChatCommandOutcome,
    NonStreamingChatResult,
    TaskProcessService,
)
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessStatusSnapshot,
    ProcessTable,
    StopResult,
)
from hivememory.workspace.process.working_set import ProcessWorkingSet

__all__ = [
    "CancelResult",
    "NonStreamingChatAgentOutcome",
    "NonStreamingChatCommandOutcome",
    "NonStreamingChatResult",
    "ProcessOutcome",
    "ProcessPhase",
    "ProcessRecord",
    "ProcessStatusSnapshot",
    "ProcessTable",
    "ProcessWorkingSet",
    "StopResult",
    "TaskProcessService",
]
