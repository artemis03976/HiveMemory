"""任务进程 — 一次进程的状态容器。

任务进程是运行时的状态容器：只持有本进程独有的东西
——进程记录（控制面：访问 context、观测标签、阶段与终态、绑定了本进程
关联字段的事件通道）、任务参数、工作集（待释放与
待补偿的资源）、线程表（Actor 阶段内的执行线程）、``trace_id``、进程 task
与输出通道。跨进程共享的编排依赖
（总线、CPU 端口、CPU 分配器、操作授权者、Gateway 超时配置）只由四阶段
骨架（``workspace.process.runner`` 的 ``TaskProcessRunner``）持有；进程
task 的创建、进程的登记、注销与 context 失效由注册入口
（``workspace.process.service``）负责。进程记录与 ``TaskProcess`` 都不
离开注册入口。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from hivememory.core.models import AttachmentSelectionRequest, WorkspaceIdentity
from hivememory.workspace.process.outputs import ProcessOutputChannel
from hivememory.workspace.process.table import ProcessRecord
from hivememory.workspace.process.threads import ProcessThreadTable
from hivememory.workspace.process.working_set import ProcessWorkingSet


@dataclass(frozen=True, kw_only=True)
class ProcessRequest:
    """一次任务进程的任务参数（注册完成后冻结，不含身份凭据）。

    ``message`` 是交给 Gateway 分析的指令文本：主动请求是用户本次发出的
    消息。``target_workspace`` 是注册时通过认证的请求进入 workspace：它是
    各阶段授权点显式接收的目标 workspace——授权点以下只流动
    操作授权者组装并经此目标校验的 ``IdentityScope``。
    """

    message: str
    target_workspace: WorkspaceIdentity
    session_id: str
    enable_memory_retrieval: bool = True
    generation_options: dict[str, Any] | None = None
    attachments: tuple[AttachmentSelectionRequest, ...] = ()


@dataclass(eq=False, kw_only=True)
class TaskProcess:
    """一次任务进程：进程记录、任务参数、工作集、线程表、进程 task 与输出通道的容器。

    由注册入口创建并登记到进程表，按对象身份识别（进程表注销与进程句柄
    都比较对象身份）。``task`` 是进程自己的执行流，由注册入口在交付方首次
    运行进程时创建：阶段推进、关闭与注销都在其中完成，不依赖交付方读取；
    进程从未运行就被关闭时，它只执行关闭流程。``task`` 为 ``None`` 表示
    进程尚未启动。``outputs`` 是进程写入阶段产出的通道，
    只允许一个订阅者，结束标记在进程关闭完成之后写入。``threads`` 登记本
    进程的全部执行线程，只经线程管理（``workspace.process.threads`` 的
    ``ProcessThreads``）读写。
    """

    record: ProcessRecord
    request: ProcessRequest
    trace_id: str
    working_set: ProcessWorkingSet = field(default_factory=ProcessWorkingSet)
    threads: ProcessThreadTable = field(default_factory=ProcessThreadTable)
    outputs: ProcessOutputChannel = field(default_factory=ProcessOutputChannel)
    task: asyncio.Task[None] | None = None


__all__ = ["ProcessRequest", "TaskProcess"]
