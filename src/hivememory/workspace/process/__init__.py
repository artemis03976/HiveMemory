"""任务进程：进程表（注册与 stop 控制面）与 chat 任务进程的四阶段编排骨架。

进程表是进程内共享设施：以 ``process_id`` 为唯一标识登记任务进程，只向
同 owner/workspace 的控制请求暴露进程记录；编排服务经全局总线的公开路由
驱动 Gateway → Patchouli prepare → Alice run → finalize，不直接持有任何
子系统引用。workspace 中 ``process`` 以外的模块不得导入本子包。
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
    "StopResult",
    "TaskProcessService",
]
