"""任务进程：注册入口、进程表（唯一的进程注册表）、进程状态容器与四阶段编排骨架。

三个角色按“谁持有什么”划分：注册入口（``service`` 的
``TaskProcessService``）只管理进程的生命周期，并为每个进程创建它自己的
进程 task；每次任务请求创建一个状态容器 ``TaskProcess``（``task_process``），
只持有本进程的进程记录、任务参数、工作集、线程表、进程 task 与输出通道；
所有进程共用的执行器 ``TaskProcessRunner``（``runner``）持有编排依赖，在
进程 task 中经全局总线的公开路由驱动 Gateway → Patchouli prepare → CPU 分配
（``allocation``）→ Actor 执行（经组合根注入的 CPU 端口调用 CPU）→
finalize，阶段产出写入输出通道，流式交付订阅通道、非流式交付等待终态
（``outputs``）。进程表是进程内共享设施，以 ``process_id`` 为键登记任务
进程，进程记录经进程取得；控制请求经注册入口的进程控制授权。
``chat.run.*`` 观测事件由领域 emitter（``events``）投影。工作集登记关闭时
仍需释放或补偿的资源，随进程关闭由取得它们的一方释放；Actor 阶段的执行线程
登记在线程表中，派生、结束与解析只经线程管理（``threads``）。workspace 中
``process`` 以外的模块不得导入本子包。
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
    "ProcessTable",
    "ProcessWorkingSet",
    "StopResult",
    "TaskProcessRunner",
    "TaskProcessService",
]
