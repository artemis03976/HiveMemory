"""任务进程 — 一次进程的状态容器。

任务进程是运行时的状态容器（任务进程 Idea 1.2）：只持有本进程独有的东西
——进程记录（控制面：访问 context、阶段与终态、绑定了本进程观测标签的
事件通道，CPU 输入清单沿用其中的只读标签）、任务参数、工作集（阶段产出与
待释放资源）、``trace_id`` 与
驱动本进程的 owner task。跨进程共享的编排依赖（总线、CPU 端口、CPU
分配器、操作授权者、Gateway 超时配置）只由四阶段骨架
（``workspace.process.runner`` 的 ``TaskProcessRunner``）持有；进程的登记、
注销与 context 失效由注册入口（``workspace.process.service``）负责。进程
记录与 ``TaskProcess`` 都不离开注册入口。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from hivememory.core.models import AttachmentSelectionRequest, WorkspaceIdentity
from hivememory.workspace.process.table import ProcessRecord
from hivememory.workspace.process.working_set import ProcessWorkingSet


@dataclass(frozen=True, kw_only=True)
class ProcessRequest:
    """一次任务进程的任务参数（注册完成后冻结，不含身份凭据）。

    ``message`` 是交给 Gateway 分析的指令文本：主动请求是用户本次发出的
    消息。``target_workspace`` 是注册时通过认证的请求进入 workspace：它是
    各阶段授权点显式接收的目标 workspace（I-4、I-8）——授权点以下只流动
    操作授权者组装并经此目标校验的 ``IdentityScope``。
    """

    message: str
    target_workspace: WorkspaceIdentity
    enable_memory_retrieval: bool = True
    generation_options: dict[str, Any] | None = None
    attachments: tuple[AttachmentSelectionRequest, ...] = ()


@dataclass(eq=False, kw_only=True)
class TaskProcess:
    """一次任务进程：进程记录、任务参数与工作集的容器。

    由注册入口创建并登记到进程表，按对象身份识别（进程表注销与进程句柄
    都比较对象身份）。``owner_task`` 是驱动本进程编排的 task，由骨架在
    运行开始时写入；关闭流程据此区分“交付方提前关闭”与“驱动方正在被
    取消”。
    """

    record: ProcessRecord
    request: ProcessRequest
    trace_id: str
    working_set: ProcessWorkingSet = field(default_factory=ProcessWorkingSet)
    owner_task: asyncio.Task[Any] | None = None


__all__ = ["ProcessRequest", "TaskProcess"]
