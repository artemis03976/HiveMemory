"""任务进程测试的装配辅助。

与生产组合根同构：编排依赖（总线、CPU 端口、CPU 分配器、阶段授权、
执行凭据表、Gateway 超时）只交给执行器 ``TaskProcessRunner``；CPU
持有真实操作入口，以每次执行收到的凭据绑定提交函数。注册入口
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
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import CPUPort, OperationEntry
from hivememory.workspace.credentials import ExecutionCredentialRegistry
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.cpu import ScriptedCPU


def make_task_process_service(
    global_bus: Any,
    *,
    cpu: CPUPort,
    access_gateway: ActorAuthenticationGateway,
    operation_authorizer: WorkspaceOperationAuthorizer,
    event_publisher: RuntimeEventPublisher | None = None,
    asset_reader: WorkspaceAssetReaderPort | None = None,
    attachment_compiler_config: AttachmentCompilerConfig | None = None,
    workspace_runtime: WorkspaceRuntime | None = None,
    credential_registry: ExecutionCredentialRegistry | None = None,
    operation_entry: OperationEntry | None = None,
) -> TaskProcessService:
    """按生产顺序装配能力与共享凭据表，再构建 CPU 分配器、执行器与注册入口。"""
    runtime = workspace_runtime or WorkspaceRuntime(
        backing=BusCanonicalReadBackend(global_bus), atom_capacity=64, profile_capacity=32
    )
    memory = MemoryApplicationService(
        global_bus,
        operation_authorizer=operation_authorizer,
        memory_reader=runtime.aliases,
    )
    credentials = credential_registry or ExecutionCredentialRegistry()
    agent = AgentApplicationService(
        global_bus,
        operation_authorizer=operation_authorizer,
        profile_reader=runtime.profiles,
    )
    entry = operation_entry or WorkspaceOperationEntry(
        memory, agent=agent, credential_registry=credentials, intent_registry=runtime.intents
    )
    if isinstance(cpu, ScriptedCPU):
        cpu.bind_operation_entry(entry)
    allocator = CPUAllocator(
        global_bus,
        operation_authorizer=operation_authorizer,
        agent_service=agent,
        asset_reader=asset_reader,
        attachment_compiler_config=attachment_compiler_config,
    )
    runner = TaskProcessRunner(
        global_bus,
        cpu=cpu,
        allocator=allocator,
        operation_authorizer=operation_authorizer,
        credential_registry=credentials,
        intent_registry=runtime.intents,
    )
    return TaskProcessService(
        runner,
        access_gateway=access_gateway,
        operation_authorizer=operation_authorizer,
        event_publisher=event_publisher,
    )


__all__ = ["make_task_process_service"]
