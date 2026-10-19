"""任务进程测试的装配辅助。

与生产组合根同构：编排依赖（总线、CPU 端口、CPU 分配器、阶段授权、
线程 context 生命周期端口、Gateway 超时）只交给执行器 ``TaskProcessRunner``；CPU
持有真实操作入口，以每次执行收到的线程 context 绑定提交函数。注册入口
``TaskProcessService`` 只持有执行器与生命周期依赖（认证网关、进程控制
授权、事件发布器）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import CPUPort, OperationEntry
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.service import TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.process.threads import ProcessThreads
from hivememory.workspace.runtime import WorkspaceRuntime
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.cpu import ScriptedCPU
from tests.helpers.intent_dispatch import install_intent_admission


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
    operation_entry: OperationEntry | None = None,
    process_table: ProcessTable | None = None,
    sessions: ConversationSessionService | None = None,
    gateway_session_turn_limit: int = 3,
    shutdown_wait_seconds: float = 10.0,
) -> TaskProcessService:
    """按生产顺序装配能力与操作入口，再构建 CPU 分配器、执行器与注册入口。"""
    if (
        isinstance(global_bus, GlobalSystemBus)
        and PatchouliRoutes.MEMORY_INTENT_SUBMIT not in global_bus.list_routes()
    ):
        install_intent_admission(global_bus)
    runtime = workspace_runtime or WorkspaceRuntime(
        backing=BusCanonicalReadBackend(global_bus), atom_capacity=64, profile_capacity=32
    )
    table = process_table if process_table is not None else ProcessTable()
    threads = ProcessThreads(table, lifecycle=access_gateway, authorizer=operation_authorizer)
    memory = ActorMemoryService(
        global_bus,
        operation_authorizer=operation_authorizer,
        memory_reader=runtime.aliases,
        dispatch_topics=threads,
    )
    agent = ActorAgentService(
        operation_authorizer=operation_authorizer,
        profile_reader=runtime.profiles,
        thread_spawner=threads,
    )
    entry = operation_entry or WorkspaceOperationEntry(
        memory,
        agent=agent,
        operation_authorizer=operation_authorizer,
    )
    if isinstance(cpu, ScriptedCPU):
        cpu.bind_operation_entry(entry)
    allocator = CPUAllocator(
        operation_authorizer=operation_authorizer,
        profile_reader=runtime.profiles,
        asset_reader=asset_reader or InMemoryWorkspaceAssetStore(),
        attachment_compiler_config=attachment_compiler_config,
    )
    sessions = sessions or ConversationSessionService(WorkspaceSessionConfig())
    runner = TaskProcessRunner(
        global_bus,
        cpu=cpu,
        sessions=sessions,
        allocator=allocator,
        operation_authorizer=operation_authorizer,
        thread_contexts=threads,
        gateway_session_turn_limit=gateway_session_turn_limit,
    )
    return TaskProcessService(
        runner,
        process_table=table,
        sessions=sessions,
        access_gateway=access_gateway,
        operation_authorizer=operation_authorizer,
        event_publisher=event_publisher,
        shutdown_wait_seconds=shutdown_wait_seconds,
    )


def process_state(table: ProcessTable, process_id: str) -> tuple[str, str, str | None] | None:
    """读取已登记进程的 (阶段, 结局, 停止原因)；未登记或已注销时为 ``None``。"""
    process = table.get(process_id)
    if process is None:
        return None
    record = process.record
    return record.phase.value, record.outcome.value, record.stop_reason


async def wait_for_process_exit(
    table: ProcessTable, process_id: str, *, timeout: float = 2.0
) -> None:
    """等待已启动进程的进程 task 结束，不请求停止；进程已注销时立即返回。"""
    process = table.get(process_id)
    if process is None:
        return
    if process.task is None:
        raise AssertionError(f"进程尚未启动: {process_id}")
    await asyncio.wait_for(asyncio.wait((process.task,)), timeout)


__all__ = ["make_task_process_service", "process_state", "wait_for_process_exit"]
