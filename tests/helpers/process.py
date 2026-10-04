"""任务进程测试的装配辅助。

与生产组合根同构：编排依赖（总线、CPU 端口、CPU 分配器、阶段授权、
Gateway 超时）只交给执行器 ``TaskProcessRunner``，注册入口
``TaskProcessService`` 只持有执行器与生命周期依赖（认证网关、进程控制
授权、事件发布器）。
"""

from __future__ import annotations

from typing import Any

from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts import CPUPort
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.service import TaskProcessService


def make_task_process_service(
    global_bus: Any,
    *,
    cpu: CPUPort,
    access_gateway: ActorAuthenticationGateway,
    operation_authorizer: WorkspaceOperationAuthorizer,
    event_publisher: RuntimeEventPublisher | None = None,
    asset_reader: WorkspaceAssetReaderPort | None = None,
    attachment_compiler_config: AttachmentCompilerConfig | None = None,
) -> TaskProcessService:
    """按生产装配方式构建注册入口：CPU 分配器 → 执行器 → 注册入口。"""
    allocator = CPUAllocator(
        global_bus,
        operation_authorizer=operation_authorizer,
        asset_reader=asset_reader,
        attachment_compiler_config=attachment_compiler_config,
    )
    runner = TaskProcessRunner(
        global_bus,
        cpu=cpu,
        allocator=allocator,
        operation_authorizer=operation_authorizer,
    )
    return TaskProcessService(
        runner,
        access_gateway=access_gateway,
        operation_authorizer=operation_authorizer,
        event_publisher=event_publisher,
    )


__all__ = ["make_task_process_service"]
