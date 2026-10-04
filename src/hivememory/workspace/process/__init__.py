"""任务进程：注册入口、进程表（唯一的进程注册表）、进程状态容器与四阶段编排骨架。

三个角色按“谁持有什么”划分：注册入口（``service`` 的
``TaskProcessService``）只管理进程的生命周期；每次任务请求创建一个状态
容器 ``TaskProcess``（``task_process``），只持有本进程的进程记录、任务参数
与工作集；所有进程共用的执行器 ``TaskProcessRunner``（``runner``）持有
编排依赖，经全局总线的公开路由驱动 Gateway → Patchouli prepare → CPU
分配（``allocation``）→ Actor 执行（经组合根注入的 CPU 端口调用 CPU）→
finalize，流式与非流式交付共用同一条骨架（``outputs``）。进程表是进程内
共享设施，以 ``process_id`` 为键登记任务进程，进程记录经进程取得；控制
请求经注册入口的进程控制授权。``chat.run.*`` 观测事件由领域 emitter
（``events``）投影。工作集登记本轮附件租借与 CPU 输出流，随进程关闭由
取得它们的一方释放。workspace 中 ``process`` 以外的模块不得导入本子包。
"""

from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.outputs import (
    NonStreamingAgentOutcome,
    NonStreamingCommandOutcome,
    NonStreamingResult,
)
from hivememory.workspace.process.runner import TaskProcessRunner
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
    "CPUAllocator",
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
    "TaskProcessRunner",
    "TaskProcessService",
]
