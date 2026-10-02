"""
TaskProcessService — 任务请求的唯一注册入口与进程控制面

位于 workspace：为每次任务请求创建一个 :class:`TaskProcess`（四阶段编排
骨架见 ``workspace.process.task_process``），把它的阶段产出交付为流式
事件或非流式结果，并经进程表（``workspace.process.table``）提供 stop 与
状态查询。对子系统的一切调用都经全局总线的公开路由完成；Actor 执行经
组合根注入的 CPU 端口（``workspace.contracts`` 的 ``CPUPort``）完成，
本服务不持有任何具体 CPU 的引用。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Coroutine
from contextlib import aclosing
from typing import Any, Literal, overload

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import (
    AttachmentSelectionRequest,
    IdentityScope,
    require_identity_scope,
)
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUPort,
)
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.events import TaskProcessEventEmitter
from hivememory.workspace.process.outputs import (
    CommandCompleted,
    NonStreamingAgentOutcome,
    NonStreamingCommandOutcome,
    NonStreamingResult,
    ProcessFailed,
    RunCancelled,
    RunCompleted,
    RunFailed,
    stream_events,
)
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessStatusSnapshot,
    ProcessTable,
)
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess


class TaskProcessService:
    """任务进程服务 — 任务请求的唯一注册入口与进程控制面。

    对子系统的一切调用都经全局总线的公开路由完成，不直接持有任何子系统
    引用。身份入口约定（v0.6.2 收敛）：本服务只接受调用方在 server 边界
    冻结的 ``IdentityScope``，不再解析裸 ``user_id``。任务进程是 Agent
    action，必须由具体 Agent 执行；actor 为保留 ``system`` 值的 scope
    会在入口被拒绝。

    CPU 分配与 Actor 执行所需能力由组合根注入：``cpu`` 是 CPU 端口
    （由 CPU 的提供方实现，当前为 Alice）；``asset_reader`` 是进程级唯一
    WorkspaceAssetStore 的只读 reader 端口（附件租借在此 acquire，随进程
    关闭统一 release）；两个编译配置段驱动进程侧的记忆/附件编译。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        event_publisher: RuntimeEventPublisher | None = None,
        gateway_request_timeout_ms: int = 8000,
        *,
        cpu: CPUPort,
        asset_reader: WorkspaceAssetReaderPort | None = None,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        attachment_compiler_config: AttachmentCompilerConfig | None = None,
    ) -> None:
        self._bus = global_bus
        self._process_table = ProcessTable()
        self._events = TaskProcessEventEmitter(
            event_publisher or RuntimeEventPublisher(NullRuntimeEventSink())
        )
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._cpu = cpu
        self._allocator = CPUAllocator(
            global_bus,
            asset_reader=asset_reader,
            memory_compiler_config=memory_compiler_config,
            attachment_compiler_config=attachment_compiler_config,
        )

    # ========== 注册入口 ==========

    @overload
    def run_process(
        self,
        message: str,
        *,
        identity_scope: IdentityScope,
        process_id: str,
        stream: Literal[True] = True,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    @overload
    def run_process(
        self,
        message: str,
        *,
        identity_scope: IdentityScope,
        process_id: str,
        stream: Literal[False],
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> Coroutine[Any, Any, NonStreamingResult]: ...

    def run_process(
        self,
        message: str,
        *,
        identity_scope: IdentityScope,
        process_id: str,
        stream: bool = True,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None] | Coroutine[Any, Any, NonStreamingResult]:
        """任务请求的唯一注册入口：使用 server 边界冻结的完整 Workspace scope。

        ``message`` 是交给 Gateway 分析的指令文本。``stream=True``（默认）
        返回流式事件的异步生成器；``stream=False``
        返回可 await 的非流式结果。两种形态运行同一条进程骨架，身份校验在
        开始迭代或 await 时才执行。``process_id`` 由 server 入口在进入本服务
        前生成并冻结（Q-16）；``attachments`` 只透传用户选择，ref/READY/版本
        校验发生在进程的 CPU 分配边界。
        """
        request = ProcessRequest(
            message=message,
            identity_scope=identity_scope,
            process_id=process_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
            attachments=tuple(attachments or ()),
        )
        if stream:
            return self._deliver_stream(request)
        return self._deliver_once(request)

    async def _deliver_stream(
        self,
        request: ProcessRequest,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式交付：阶段产出逐项投影为流式事件；编排异常翻译为 error 事件。"""
        process = self._open_process(request, stream=True)
        async with aclosing(process.run()) as outputs:
            async for output in outputs:
                if isinstance(output, ProcessFailed):
                    yield _stream_error(output.error)
                    continue
                for event in stream_events(output, process_id=request.process_id):
                    yield event

    async def _deliver_once(self, request: ProcessRequest) -> NonStreamingResult:
        """非流式交付：只取终态产出；编排异常在进程关闭后原样上抛。"""
        process = self._open_process(request, stream=False)
        result: NonStreamingResult | None = None
        async with aclosing(process.run()) as outputs:
            async for output in outputs:
                match output:
                    case CommandCompleted(command_result=command_result):
                        result = NonStreamingCommandOutcome(command_execution_result=command_result)
                    case RunCompleted(execution_result=execution_result) | RunFailed(
                        execution_result=execution_result
                    ):
                        result = NonStreamingAgentOutcome(execution_result=execution_result)
                    case RunCancelled(execution_result=execution_result):
                        result = NonStreamingAgentOutcome(
                            execution_result=_cancelled_execution_result(execution_result)
                        )
                    case ProcessFailed(error=error):
                        raise error
        if result is None:
            raise RuntimeError("任务进程结束时没有终态产出")
        return result

    def _open_process(self, request: ProcessRequest, *, stream: bool) -> TaskProcess:
        identity_scope = require_identity_scope(request.identity_scope)
        self._reject_system_actor(identity_scope.actor_identity.agent_id)
        return TaskProcess(
            request,
            stream=stream,
            global_bus=self._bus,
            process_table=self._process_table,
            allocator=self._allocator,
            cpu=self._cpu,
            events=self._events,
            gateway_request_timeout_ms=self._gateway_request_timeout_ms,
        )

    # ========== 进程控制 ==========

    def cancel_process(
        self,
        process_id: str,
        *,
        identity_scope: IdentityScope,
        reason: str = "user_requested",
    ) -> CancelResult:
        """幂等取消入口：请求方 scope 只做 owner/workspace 校验。

        取消与事件发布一律使用进程创建时冻结在进程表里的原始 scope；
        请求方当前选择（尤其是 agent 维度）不得重新构造出可能不同的身份
        坐标，因此跨 user/workspace 的取消只会得到 ``not_found``。
        """
        identity_scope = require_identity_scope(identity_scope)
        result = self._process_table.cancel(
            process_id,
            identity_scope,
            reason=reason,
        )
        run = self._process_table.get(process_id, identity_scope)
        # 事件承载进程创建时冻结的身份坐标；请求方 scope 仅用于上面的校验。
        frozen_scope = run.identity_scope if run is not None else identity_scope
        self._events.cancel_requested(
            result,
            workspace_id=frozen_scope.workspace_identity.workspace_id,
        )
        if run is not None:
            self._events.for_process(run).status()
        return result

    def process_status(
        self,
        process_id: str,
        *,
        identity_scope: IdentityScope,
    ) -> ProcessStatusSnapshot | None:
        """返回 scoped 进程状态；错误 scope 与不存在统一为 ``None``。"""
        return self._process_table.status(
            process_id,
            require_identity_scope(identity_scope),
        )

    # ========== 内部辅助 ==========

    @staticmethod
    def _reject_system_actor(agent_id: str) -> None:
        """任务进程必须由具体 Agent 执行；保留 ``system`` actor 在此显式失败。"""
        if agent_id == SYSTEM_AGENT_ID:
            raise WorkspaceDomainError(
                "任务进程不能使用保留 system actor：必须指定具体执行 Agent",
                details={"agent_id": agent_id},
            )


def _stream_error(error: Exception) -> dict[str, Any]:
    """编排异常的 SSE 翻译：Workspace 领域错误携带安全文案，其余统一为系统错误。"""
    if isinstance(error, WorkspaceDomainError):
        return {"event": "error", "data": {"message": str(error), "code": error.code}}
    return {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}


def _cancelled_execution_result(result: CPUExecutionResult | None) -> CPUExecutionResult:
    """把 Actor 自报的执行结果归一为取消终态；无结果时构造纯取消结果。"""
    if result is None:
        return CPUExecutionResult(status=CPUExecutionStatus.CANCELLED)
    return result.model_copy(update={"status": CPUExecutionStatus.CANCELLED.value})


__all__ = [
    "NonStreamingAgentOutcome",
    "NonStreamingCommandOutcome",
    "NonStreamingResult",
    "TaskProcessService",
]
