"""
TaskProcessService — 任务请求的唯一注册入口与进程控制面

位于 workspace：完成任务请求的两阶段认证（注册步骤），签发即绑定并立即
创建进程、登记到进程表，返回不透明的进程句柄；已注册的
:class:`TaskProcess`（状态容器）经句柄交给组合根注入的执行器
:class:`TaskProcessRunner`（四阶段编排骨架）运行，交付为流式事件或非流式
结果，并经进程表提供取消与状态查询。本入口只管理任务进程的生命周期
（任务进程 Idea Q-3）：不持有总线、CPU 端口、CPU 分配器、asset reader
与编译配置等编排依赖，它们只由执行器持有。

注册入口的生命周期职责（A1 访问边界返工第 4.4 节）：未通过两阶段认证不
创建进程；注册成功即登记，进程以任何结局关闭后由本入口使 context 失效
并从进程表注销——执行器的关闭流程只释放进程自身的资源。进程表是唯一的进程
注册表，登记 ``process_id → TaskProcess``。进程记录与 ``TaskProcess`` 都不
离开本入口：入口 adapter（server）只持有 :class:`ProcessHandle`；句柄按
对象身份判定有效，持有有效句柄即为该进程生命周期的所有者，运行与关闭以
句柄为参数，取消以句柄或 ``process_id`` 加请求级 context 为依据（I-8 的
2026-10-04 补充）。
注册成功但流一直没有开始时（例如关停信号在注册期间到达），调用方经
:meth:`close_process` 触发同一关闭路径。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Coroutine
from contextlib import aclosing
from typing import Any, Literal, overload

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.trace_context import generate_trace_id
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
    ExecutionLabels,
    WorkspaceIdentity,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
)
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
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessRecord,
    ProcessStatusSnapshot,
    ProcessTable,
)
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess


class ProcessHandle:
    """已注册任务进程的不透明句柄（I-8 的补充）。

    句柄是入口 adapter 与已注册进程之间唯一的引用：只暴露 ``process_id``，
    不暴露进程记录、其中的访问 context 或 ``TaskProcess``。句柄只由注册
    入口签发，按对象身份判定有效（I-8 的 2026-10-04 补充）：它私下记着
    签发时对应的进程，注册入口解析时要求进程表中登记的正是这一个。按
    ``process_id`` 重新构造的对象、进程关闭后的旧句柄都不是有效句柄；
    入口 adapter 见不到进程对象，因此造不出有效句柄。这维护的是可信进程
    内的调用纪律，与访问 context 的信任模型一致；句柄不离开进程，不提供
    序列化。
    """

    __slots__ = ("_process", "_process_id")

    def __init__(self, process: TaskProcess) -> None:
        self._process = process
        self._process_id = process.record.process_id

    @property
    def process_id(self) -> str:
        """句柄对应进程的唯一标识（Q-16）。"""
        return self._process_id

    def __repr__(self) -> str:
        return f"ProcessHandle(process_id={self._process_id!r})"


class TaskProcessService:
    """任务进程服务 — 任务请求的唯一注册入口与进程控制面。

    注册入口完成两阶段认证（v0.7.0 A1 访问边界返工第 4.4 节）：调用方只
    提交 actor 声明与请求进入的 workspace（认证前不组装 ``IdentityScope``），
    认证失败直接抛出 ``AdmissionDeniedError``，不创建、不登记进程。任务
    进程是 Agent action，必须由具体 Agent 执行；actor 为保留 ``system`` 值
    的声明在认证前被拒绝。

    本入口只持有生命周期依赖：``runner`` 是组合根构建的执行器（四阶段编排
    骨架，持有总线、CPU 端口、CPU 分配器等编排依赖）；本服务是进程
    context 的运行持有者——注册经 ``access_gateway`` 认证签发即绑定本进程，
    进程以任何结局关闭时由本入口经认证网关使绑定 context 失效；
    ``operation_authorizer`` 只用于进程控制授权（阶段授权由执行器执行）；
    ``event_publisher`` 驱动 ``chat.run.*`` 观测事件。
    """

    def __init__(
        self,
        runner: TaskProcessRunner,
        *,
        access_gateway: ActorAuthenticationGateway,
        operation_authorizer: WorkspaceOperationAuthorizer,
        event_publisher: RuntimeEventPublisher | None = None,
    ) -> None:
        self._runner = runner
        # 唯一的进程注册表：process_id → 已注册的 TaskProcess（不外泄）。
        self._process_table = ProcessTable()
        self._events = TaskProcessEventEmitter(
            event_publisher or RuntimeEventPublisher(NullRuntimeEventSink())
        )
        self._access_gateway = access_gateway
        self._authorizer = operation_authorizer

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
    ) -> ProcessHandle:
        """注册步骤：认证、签发即绑定、创建进程并登记（流式响应开始前完成）。

        ``actor`` / ``workspace`` 是 server 入口解析的声明（不是已组装的
        ``IdentityScope``）；``process_id`` 由 server 入口在进入本服务前
        生成并冻结（Q-16），认证签发的 context 以它为运行绑定。任一认证
        检查失败直接抛出（HTTP 入口据此返回 403），此时不创建、不登记
        进程；认证通过后、登记完成前失败（如 ``process_id`` 重复）时使
        已签发的 context 失效，不留悬挂凭据。注册成功返回不透明的进程
        句柄；``attachments`` 只透传用户选择，ref/READY/版本校验发生在
        进程的 CPU 分配边界。
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
        try:
            # 3-4. 创建进程记录写入 context，绑定观测标签，并登记到进程表。
            # 标签只在认证成功后创建一次，事件通道与 CPU 输入清单共用。
            labels = ExecutionLabels(agent_id=actor.agent_id, workspace_id=workspace.workspace_id)
            trace_id = generate_trace_id("task")
            record = ProcessRecord(
                process_id=process_id,
                access=access,
                events=self._events.for_process(
                    process_id=process_id,
                    labels=labels,
                    trace_id=trace_id,
                ),
            )
            request = ProcessRequest(
                message=message,
                target_workspace=workspace,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
                attachments=tuple(attachments or ()),
            )
            process = TaskProcess(record=record, request=request, trace_id=trace_id)
            self._process_table.register(process)
            record.events.created(record)
        except BaseException:
            # 登记前失败：使已签发的 context 失效，不留悬挂凭据（P-6）。
            self._access_gateway.invalidate_context(access)
            raise
        # 5. 签发不透明的进程句柄；运行由调用方经 run_process 发起。
        return ProcessHandle(process)

    @overload
    def run_process(
        self,
        handle: ProcessHandle,
        *,
        stream: Literal[True] = True,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    @overload
    def run_process(
        self,
        handle: ProcessHandle,
        *,
        stream: Literal[False],
    ) -> Coroutine[Any, Any, NonStreamingResult]: ...

    def run_process(
        self,
        handle: ProcessHandle,
        *,
        stream: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None] | Coroutine[Any, Any, NonStreamingResult]:
        """运行句柄对应的已注册进程：``stream=True``（默认）返回流式事件的
        异步生成器，``stream=False`` 返回可 await 的非流式结果。

        两种形态只负责运行，认证与登记已在 :meth:`register_process` 完成；
        交付结束（含提前关闭）后由本服务收口：使绑定 context 失效并从
        进程表注销。
        """
        process = self._resolve(handle)
        if stream:
            return self._deliver_stream(process)
        return self._deliver_once(process)

    async def close_process(self, handle: ProcessHandle) -> None:
        """注册入口的关闭路径：经执行器释放进程资源，使绑定 context 失效并注销。

        进程正常交付结束、以任何结局终态化、或注册成功但流一直没有开始
        （例如关停信号在注册期间到达），都经本路径收口；幂等，重复调用
        或句柄已失效（进程已关闭、句柄不是本入口签发的）是空操作。
        """
        process = self._lookup(handle)
        if process is None:
            return
        await self._close(process)

    async def _close(self, process: TaskProcess) -> None:
        """关闭一个已登记的进程（幂等）。

        context 失效与进程注销放在内层 ``finally``：关闭子流被取消中断时
        收口仍要完成，不留已失效却仍登记的进程。
        """
        record = process.record
        try:
            await self._runner.close(process)
        finally:
            # 绑定 context 随进程关闭失效（P-6）：无论 completed、失败、
            # 取消还是断流，进程结束后该凭据不再可用。
            self._access_gateway.invalidate_context(record.access)
            self._process_table.close(process)

    async def _deliver_stream(
        self,
        process: TaskProcess,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式交付：阶段产出逐项投影为流式事件；编排异常翻译为 error 事件。"""
        try:
            async with aclosing(self._runner.run(process, stream=True)) as outputs:
                async for output in outputs:
                    if isinstance(output, ProcessFailed):
                        yield _stream_error(output.error)
                        continue
                    for event in stream_events(output, process_id=process.record.process_id):
                        yield event
        finally:
            await self._close(process)

    async def _deliver_once(self, process: TaskProcess) -> NonStreamingResult:
        """非流式交付：只取终态产出；编排异常在进程关闭后原样上抛。"""
        try:
            result: NonStreamingResult | None = None
            async with aclosing(self._runner.run(process, stream=False)) as outputs:
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
            await self._close(process)

    # ========== 进程控制 ==========

    @overload
    def cancel_process(
        self,
        target: ProcessHandle,
        *,
        reason: str = "user_requested",
    ) -> CancelResult: ...

    @overload
    def cancel_process(
        self,
        target: str,
        *,
        access: WorkspaceAccessContext,
        reason: str = "user_requested",
    ) -> CancelResult: ...

    def cancel_process(
        self,
        target: ProcessHandle | str,
        *,
        access: WorkspaceAccessContext | None = None,
        reason: str = "user_requested",
    ) -> CancelResult:
        """唯一的幂等取消方法：取消的依据作为参数（I-8 的 2026-10-04 补充）。

        - 传入句柄：调用方是进程的所有者（例如 chat 路由在客户端断开时），
          不经进程控制授权，不接受另传的 ``access``；句柄已失效时返回
          ``not_found``，不发布事件。
        - 传入 ``process_id`` 与请求级 ``access``：控制请求（``/chat/stop``），
          经操作授权者的进程控制授权比对请求方与进程记录的驻留坐标（P-7）；
          不匹配时与进程不存在一样返回 ``not_found``，不泄露进程是否存在，
          并发布带请求方观测标签的事件。判定顺序：进程查无先于请求方
          context 校验；进程存在而请求方 context 无效按接线缺陷以
          ``ScopeRequiredError`` 拒绝（生产入口经网关签发后不应出现）。

        取消不新增 operation。两种依据解析出进程后，stop 记录、终态判定
        与运行时事件只有一份实现，事件一律使用进程注册时绑定的观测标签。
        """
        if isinstance(target, ProcessHandle):
            if access is not None:
                raise TypeError("句柄形式的取消不接受 access：持有句柄即为进程的所有者")
            process = self._lookup(target)
            if process is None:
                return _not_found(target.process_id, reason)
            return self._request_stop(process, reason)

        if not isinstance(target, str):
            raise TypeError("cancel_process 的 target 必须是 ProcessHandle 或 process_id")
        if access is None:
            raise TypeError("按 process_id 取消必须提供请求级 access")
        process = self._process_table.get(target)
        if process is None or not self._authorizer.authorize_process_control(
            access, process.record.access
        ):
            result = _not_found(target, reason)
            # 请求方 context 的驻留 workspace 只是观测标签；经认证网关的
            # 诊断查询取回，不作为授权依据。
            summary = self._access_gateway.describe_context(access)
            self._events.cancel_requested(
                result,
                workspace_id=summary.workspace_id if summary else None,
            )
            return result
        return self._request_stop(process, reason)

    def process_status(
        self,
        process_id: str,
        *,
        access: WorkspaceAccessContext,
    ) -> ProcessStatusSnapshot | None:
        """返回 scoped 进程状态；不可控与不存在统一为 ``None``。"""
        process = self._process_table.get(process_id)
        if process is None or not self._authorizer.authorize_process_control(
            access, process.record.access
        ):
            return None
        record = process.record
        return ProcessStatusSnapshot(
            process_id=record.process_id,
            phase=record.phase.value,
            status=record.outcome.value,
            reason=record.stop_reason,
        )

    # ========== 内部辅助 ==========

    def _resolve(self, handle: ProcessHandle) -> TaskProcess:
        """把句柄解析为已登记的进程；句柄无效时显式失败（``process_handle_unknown``）。"""
        process = self._lookup(handle)
        if process is None:
            raise WorkspaceDomainError(
                "进程句柄无效：不是本入口签发的句柄，或对应进程已关闭",
                details={"reason": "process_handle_unknown", "process_id": handle.process_id},
            )
        return process

    def _lookup(self, handle: ProcessHandle) -> TaskProcess | None:
        """按对象身份解析句柄：进程表中登记的必须正是句柄签发时对应的进程。"""
        if not isinstance(handle, ProcessHandle):
            raise TypeError("handle 必须是注册入口签发的 ProcessHandle")
        process = self._process_table.get(handle.process_id)
        if process is None or process is not handle._process:
            return None
        return process

    @staticmethod
    def _request_stop(process: TaskProcess, reason: str) -> CancelResult:
        """记录 stop 并发布取消事件：两种取消依据共用的唯一实现。"""
        record = process.record
        stop = record.request_stop(reason)
        result = CancelResult(
            process_id=record.process_id,
            cancelled=stop.accepted,
            status=record.outcome.value,
            reason=stop.reason,
        )
        record.events.cancel_requested(result)
        record.events.status(record)
        return result

    @staticmethod
    def _reject_system_actor(agent_id: str) -> None:
        """任务进程必须由具体 Agent 执行；保留 ``system`` actor 在此显式失败。"""
        if agent_id == SYSTEM_AGENT_ID:
            raise WorkspaceDomainError(
                "任务进程不能使用保留 system actor：必须指定具体执行 Agent",
                details={"agent_id": agent_id},
            )


def _not_found(process_id: str, reason: str) -> CancelResult:
    """取消找不到可控进程时的统一结果。"""
    return CancelResult(
        process_id=process_id,
        cancelled=False,
        status="not_found",
        reason=reason,
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
    "ProcessHandle",
    "TaskProcessService",
]
