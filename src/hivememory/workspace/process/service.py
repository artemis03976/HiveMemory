"""
TaskProcessService — 任务请求的唯一注册入口与进程控制面

位于 workspace：完成任务请求的两阶段认证（注册步骤），签发即绑定并立即
创建进程、登记到进程表，然后把已注册的 :class:`TaskProcess`（四阶段编排
骨架见 ``workspace.process.task_process``）交付为流式事件或非流式结果，
并经进程表提供 stop 与状态查询。对子系统的一切调用都经全局总线的公开
路由完成；Actor 执行经组合根注入的 CPU 端口（``workspace.contracts`` 的
``CPUPort``）完成，本服务不持有任何具体 CPU 的引用。

注册入口的生命周期职责（A1 访问边界返工第 4.4 节）：未通过两阶段认证不
创建进程；注册成功即登记，进程以任何结局关闭后由本入口使 context 失效
并从进程表注销——``TaskProcess`` 只释放自身资源。注册成功但流一直没有
开始时（例如关停信号在注册期间到达），调用方经 :meth:`close_process`
触发同一关闭路径。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Coroutine
from contextlib import aclosing
from typing import Any, Literal, overload

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.trace_context import generate_trace_id
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.access import (
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import (
    ActorIdentity,
    AttachmentSelectionRequest,
    WorkspaceIdentity,
)
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.workspace.access import WorkspaceAccessGuard
from hivememory.workspace.authentication import ActorAuthenticationGateway
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
    ProcessRecord,
    ProcessStatusSnapshot,
    ProcessTable,
)
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess


class TaskProcessService:
    """任务进程服务 — 任务请求的唯一注册入口与进程控制面。

    对子系统的一切调用都经全局总线的公开路由完成，不直接持有任何子系统
    引用。注册入口完成两阶段认证（v0.7.0 A1 访问边界返工第 4.4 节）：
    调用方只提交 actor 声明与请求进入的 workspace（认证前不组装
    ``IdentityScope``），认证失败直接抛出 ``AdmissionDeniedError``，不
    创建、不登记进程。任务进程是 Agent action，必须由具体 Agent 执行；
    actor 为保留 ``system`` 值的声明在认证前被拒绝。

    CPU 分配与 Actor 执行所需能力由组合根注入：``cpu`` 是 CPU 端口
    （由 CPU 的提供方实现，当前为 Alice）；``asset_reader`` 是进程级唯一
    WorkspaceAssetStore 的只读 reader 端口（附件租借在此 acquire，随进程
    关闭统一释放）；两个编译配置段驱动进程侧的记忆/附件编译；
    ``access_gateway`` 与 ``access_guard`` 使用组合根装配的同一实例——
    注册经网关认证签发即绑定本进程，阶段 operation 授权在进程内执行，
    进程以任何结局关闭时由本入口使绑定 context 失效。
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
        access_gateway: ActorAuthenticationGateway,
        access_guard: WorkspaceAccessGuard,
    ) -> None:
        self._bus = global_bus
        self._process_table = ProcessTable()
        self._events = TaskProcessEventEmitter(
            event_publisher or RuntimeEventPublisher(NullRuntimeEventSink())
        )
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._cpu = cpu
        self._access_gateway = access_gateway
        self._access_guard = access_guard
        self._allocator = CPUAllocator(
            global_bus,
            asset_reader=asset_reader,
            memory_compiler_config=memory_compiler_config,
            attachment_compiler_config=attachment_compiler_config,
            access_guard=access_guard,
        )

    # ========== 注册入口 ==========

    async def register_process(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        process_id: str,
        message: str,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> TaskProcess:
        """注册步骤：认证、签发即绑定、创建进程并登记（流式响应开始前完成）。

        ``actor`` / ``workspace`` 是 server 入口解析的声明（不是已组装的
        ``IdentityScope``）；``process_id`` 由 server 入口在进入本服务前
        生成并冻结（Q-16），认证签发的 context 以它为运行绑定。任一认证
        检查失败直接抛出（HTTP 入口据此返回 403），此时不创建、不登记
        进程。注册成功返回已登记的进程句柄；``attachments`` 只透传用户
        选择，ref/READY/版本校验发生在进程的 CPU 分配边界。
        """
        # 1. 拒绝 system 作为任务请求的 actor：任务进程必须由具体 Agent 执行。
        self._reject_system_actor(actor.agent_id)
        # 2. 经认证网关完成两阶段认证，签发即绑定本进程（I-3）。
        access = await self._access_gateway.authenticate(
            adapter=adapter,
            principal=principal,
            actor=actor,
            workspace=workspace,
            binding=RunBinding.for_task_process(process_id),
        )
        # 3-4. 创建进程记录写入 context，绑定观测标签，并登记到进程表。
        record = ProcessRecord(process_id=process_id, access=access)
        request = ProcessRequest(
            message=message,
            workspace=workspace,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
            attachments=tuple(attachments or ()),
        )
        # 用通过认证的声明绑定运行时事件的观测标签（workspace_id /
        # agent_id 只在创建时绑定一次）。
        trace_id = generate_trace_id("task")
        record.events = self._events.for_process(
            record,
            workspace_id=workspace.workspace_id,
            agent_id=actor.agent_id,
            trace_id=trace_id,
        )
        process = TaskProcess(
            record=record,
            request=request,
            global_bus=self._bus,
            allocator=self._allocator,
            cpu=self._cpu,
            events=record.events,
            gateway_request_timeout_ms=self._gateway_request_timeout_ms,
            access_guard=self._access_guard,
            trace_id=trace_id,
        )
        self._process_table.register(record)
        record.events.created()
        # 5. 返回进程句柄；运行由调用方经 run_process 发起。
        return process

    @overload
    def run_process(
        self,
        process: TaskProcess,
        *,
        stream: Literal[True] = True,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    @overload
    def run_process(
        self,
        process: TaskProcess,
        *,
        stream: Literal[False],
    ) -> Coroutine[Any, Any, NonStreamingResult]: ...

    def run_process(
        self,
        process: TaskProcess,
        *,
        stream: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None] | Coroutine[Any, Any, NonStreamingResult]:
        """运行一个已注册的进程：``stream=True``（默认）返回流式事件的
        异步生成器，``stream=False`` 返回可 await 的非流式结果。

        两种形态只负责运行，认证与登记已在 :meth:`register_process` 完成；
        交付结束（含提前关闭）后由本服务收口：使绑定 context 失效并从
        进程表注销。
        """
        if stream:
            return self._deliver_stream(process)
        return self._deliver_once(process)

    async def close_process(self, process: TaskProcess) -> None:
        """注册入口的关闭路径：释放进程资源，使绑定 context 失效并注销。

        进程正常交付结束、以任何结局终态化、或注册成功但流一直没有开始
        （例如关停信号在注册期间到达），都经本方法收口；幂等，重复调用
        是空操作。context 失效与进程注销放在内层 ``finally``：关闭子流被
        取消中断时收口仍要完成，不留已失效却仍登记的进程。
        """
        record = process.record
        try:
            await process.close()
        finally:
            # 绑定 context 随进程关闭失效（P-6）：无论 completed、失败、
            # 取消还是断流，进程结束后该凭据不再可用。
            self._access_guard.invalidate(record.access)
            self._process_table.close(record)

    async def _deliver_stream(
        self,
        process: TaskProcess,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式交付：阶段产出逐项投影为流式事件；编排异常翻译为 error 事件。"""
        try:
            async with aclosing(process.run(stream=True)) as outputs:
                async for output in outputs:
                    if isinstance(output, ProcessFailed):
                        yield _stream_error(output.error)
                        continue
                    for event in stream_events(output, process_id=process.record.process_id):
                        yield event
        finally:
            await self.close_process(process)

    async def _deliver_once(self, process: TaskProcess) -> NonStreamingResult:
        """非流式交付：只取终态产出；编排异常在进程关闭后原样上抛。"""
        try:
            result: NonStreamingResult | None = None
            async with aclosing(process.run(stream=False)) as outputs:
                async for output in outputs:
                    match output:
                        case CommandCompleted(command_result=command_result):
                            result = NonStreamingCommandOutcome(
                                command_execution_result=command_result
                            )
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
        finally:
            await self.close_process(process)

    # ========== 进程控制 ==========

    def cancel_process(
        self,
        process_id: str,
        *,
        access: WorkspaceAccessContext,
        reason: str = "user_requested",
    ) -> CancelResult:
        """幂等取消入口：比对请求方 context 与进程记录中的 context（P-7）。

        取消是进程控制操作，不新增 operation：请求方经 guard 的进程控制
        授权与目标进程比对驻留坐标，不匹配时与不存在统一按 ``not_found``
        呈现，不泄露进程是否存在。判定顺序：进程查无（``not_found``）先于
        请求方 context 校验；进程存在而请求方 context 无效按接线缺陷以
        ``ScopeRequiredError`` 拒绝（生产入口经网关签发后不应出现）。取消
        与事件发布一律使用进程创建时绑定的观测标签，请求方当前声明不得
        重新构造身份坐标。
        """
        record = self._process_table.get(process_id)
        if record is None or not self._access_guard.authorize_process_control(
            access, record.access
        ):
            result = CancelResult(
                process_id=process_id,
                cancelled=False,
                status="not_found",
                reason=reason,
            )
            # 请求方 context 的驻留 workspace 只是观测标签；经 guard 诊断
            # 查询取回，不作为身份兑现。
            summary = self._access_guard.describe(access)
            self._events.cancel_requested(
                result,
                workspace_id=summary.workspace_id if summary else None,
            )
            return result

        stop = record.request_stop(reason)
        result = CancelResult(
            process_id=record.process_id,
            cancelled=stop.accepted,
            status=record.outcome.value,
            reason=stop.reason,
        )
        if record.events is not None:
            # 注册入口登记的进程都有绑定的事件发布器；事件本身是 best-effort
            # 观测，未绑定（不属于本服务登记流程）时跳过发布。
            record.events.cancel_requested(result)
            record.events.status()
        return result

    def process_status(
        self,
        process_id: str,
        *,
        access: WorkspaceAccessContext,
    ) -> ProcessStatusSnapshot | None:
        """返回 scoped 进程状态；不可控与不存在统一为 ``None``。"""
        record = self._process_table.get(process_id)
        if record is None or not self._access_guard.authorize_process_control(
            access, record.access
        ):
            return None
        return ProcessStatusSnapshot(
            process_id=record.process_id,
            phase=record.phase.value,
            status=record.outcome.value,
            reason=record.stop_reason,
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
