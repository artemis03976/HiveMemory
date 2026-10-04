"""任务进程：进程表（唯一的进程注册表）、进程工作集与任务进程的四阶段编排骨架。

进程表是进程内共享设施，也是唯一的进程注册表：以 ``process_id`` 为键
登记任务进程，进程记录经进程取得；控制请求经注册入口的进程控制授权。每次任务请求创建一个
``TaskProcess``（``task_process``），经全局总线的公开路由驱动 Gateway →
Patchouli prepare → CPU 分配（``allocation``）→ Actor 执行（经组合根
注入的 CPU 端口调用 CPU）→ finalize，流式与非流式交付共用同一条骨架
（``outputs``）；``chat.run.*`` 观测事件由领域 emitter（``events``）投影。
进程工作集持有本轮附件租借并随进程关闭统一释放。workspace 中 ``process``
以外的模块不得导入本子包。
"""

from hivememory.workspace.process.outputs import (
    NonStreamingAgentOutcome,
    NonStreamingCommandOutcome,
    NonStreamingResult,
)
from hivememory.workspace.process.service import TaskProcessService
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
    "NonStreamingAgentOutcome",
    "NonStreamingCommandOutcome",
    "NonStreamingResult",
    "ProcessOutcome",
    "ProcessPhase",
    "ProcessRecord",
    "ProcessStatusSnapshot",
    "ProcessTable",
    "ProcessWorkingSet",
    "StopResult",
    "TaskProcessService",
]
