"""
TaskProcessService — 任务请求的唯一注册入口与进程控制面

位于 workspace：完成任务请求的两阶段认证（注册步骤），签发即绑定并立即
创建进程、登记到进程表，返回不透明的进程句柄；交付方首次运行进程时，
本入口为它创建进程 task，在其中由组合根注入的执行器
:class:`TaskProcessRunner`（四阶段编排骨架）推进各阶段。交付方只订阅
进程的输出通道（流式），或等待进程 task 交出终态（非流式），并经进程表
提供取消。本入口只管理任务进程的生命周期：不持有总线、CPU 端口、CPU
分配器、asset reader 与编译配置等编排依赖，它们只由执行器持有。

注册入口的生命周期职责：未通过两阶段认证不创建进程；注册成功即登记；
进程只有一条退出路径——进程 task 在执行器完成关闭流程后，由本入口使
context 失效、从进程表注销并释放会话认领，最后结束输出通道。交付方离开
只退订并以交付方给出的原因请求停止，不在自己的取消上下文中执行关闭。
进程表是唯一的进程注册表，登记 ``process_id → TaskProcess``。进程记录与
``TaskProcess`` 都不离开本入口：入口 adapter（server）只持有
:class:`ProcessHandle`；句柄按对象身份判定有效，持有有效句柄即为该进程
生命周期的所有者，运行与关闭以句柄为参数，取消以句柄或 ``process_id``
加请求级 context 为依据。注册成功但交付一直没有开始时（例如关停信号在
注册期间到达），调用方经 :meth:`close_process` 关闭进程。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Coroutine
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
from hivememory.workspace.process.events import TaskProcessEventEmitter
from hivememory.workspace.process.outputs import (
    NonStreamingAgentOutcome,
    NonStreamingCommandOutcome,
    NonStreamingResult,
    ProcessFailed,
    TerminalOutput,
    non_streaming_result,
    stream_events,
)
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.table import (
    STREAM_CLOSED_STOP_REASON,
    SYSTEM_SHUTDOWN_STOP_REASON,
    CancelResult,
    ProcessRecord,
    ProcessTable,
)
from hivememory.workspace.process.task_process import ProcessRequest, TaskProcess
from hivememory.workspace.sessions import ConversationSessionService

logger = logging.getLogger(__name__)


class ProcessHandle:
    """已注册任务进程的不透明句柄。

    句柄是入口 adapter 与已注册进程之间唯一的引用：只暴露 ``process_id`` 与 ``session_id``，
    不暴露进程记录、其中的访问 context 或 ``TaskProcess``。句柄只由注册
    入口签发，按对象身份判定有效：它私下记着
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
    def session_id(self) -> str:
        """本进程认领的会话 ID；不暴露访问凭据。"""
        return self._process.request.session_id

    @property
    def process_id(self) -> str:
        """句柄对应进程的唯一标识。"""
        return self._process_id

    def __repr__(self) -> str:
        return f"ProcessHandle(process_id={self._process_id!r})"


class TaskProcessService:
    """任务进程服务 — 任务请求的唯一注册入口与进程控制面。

    注册入口完成两阶段认证：调用方只
    提交 actor 声明与请求进入的 workspace（认证前不组装 ``IdentityScope``），
    认证失败直接抛出 ``AdmissionDeniedError``，不创建、不登记进程。任务
    进程是 Agent action，必须由具体 Agent 执行；actor 为保留 ``system`` 值
    的声明在认证前被拒绝。

    本入口只持有生命周期依赖：``runner`` 是组合根构建的执行器（四阶段编排
    骨架，持有总线、CPU 端口、CPU 分配器等编排依赖）；本服务是进程
    context 的运行持有者——注册经 ``access_gateway`` 认证签发即绑定本进程，
    进程以任何结局关闭时由本入口经认证网关使绑定 context 失效；
    ``operation_authorizer`` 只用于进程控制授权（阶段授权由执行器执行）；
    ``event_publisher`` 驱动 ``chat.run.*`` 观测事件；``shutdown_wait_seconds``
    是 System 关闭时请求停止后等待进程 task 结束的上限，超时后硬取消。
    """

    def __init__(
        self,
        runner: TaskProcessRunner,
        *,
        process_table: ProcessTable,
        sessions: ConversationSessionService,
        access_gateway: ActorAuthenticationGateway,
        operation_authorizer: WorkspaceOperationAuthorizer,
        event_publisher: RuntimeEventPublisher | None = None,
        shutdown_wait_seconds: float = 10.0,
    ) -> None:
        self._sessions = sessions
        self._shutdown_wait_seconds = shutdown_wait_seconds
        self._runner = runner
        # 唯一的进程注册表：process_id → 已注册的 TaskProcess（不外泄）。
        self._process_table = process_table
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
        session_id: str | None = None,
    ) -> ProcessHandle:
        """注册步骤：认证、签发即绑定、创建进程并登记（流式响应开始前完成）。

        ``actor`` / ``workspace`` 是 server 入口解析的声明（不是已组装的
        ``IdentityScope``）；``process_id`` 由 server 入口在进入本服务前
        生成并冻结，认证签发的 context 以它为运行绑定。任一认证
        检查失败直接抛出（HTTP 入口据此返回 403），此时不创建、不登记
        进程；认证通过后、登记完成前失败（如 ``process_id`` 重复）时使
        已签发的 context 失效，不留悬挂凭据。注册成功返回不透明的进程
        句柄；``attachments`` 只透传用户选择，ref/READY/版本校验发生在
        进程的 CPU 分配边界。
        """
        # 1. 拒绝 system 作为任务请求的 actor：任务进程必须由具体 Agent 执行。
        self._reject_system_actor(actor.agent_id)
        # 2. 经认证网关完成两阶段认证，签发即绑定本进程。
        # 注册入口只服务进程内入口（server 的 chat），没有接入凭据。
        access = await self._access_gateway.authenticate(
            adapter=adapter,
            principal=principal,
            credential=None,
            actor=actor,
            workspace=workspace,
            binding=RunBinding.for_task_process(process_id),
        )
        claimed = False
        created = False
        try:
            # 认证成功后同步认领；后续登记失败时还原会话与凭据。
            # 进程表登记时还会再查一次；这里先于会话检查，使重复的 process_id
            # 不创建或认领会话，也不被会话忙碌、容量等错误掩盖。
            if self._process_table.get(process_id) is not None:
                raise WorkspaceDomainError(
                    "process_id 已被注册，拒绝覆盖现有任务进程",
                    details={"process_id": process_id},
                )
            if session_id is None:
                session_id = self._sessions.create(workspace).session_id
                created = True
            self._sessions.claim(workspace, session_id, process_id)
            claimed = True
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
                session_id=session_id,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
                attachments=tuple(attachments or ()),
            )
            process = TaskProcess(record=record, request=request, trace_id=trace_id)
            self._process_table.register(process)
            record.events.created(record)
        except BaseException:
            # 登记前失败：使已签发的 context 失效，不留悬挂凭据。
            self._access_gateway.revoke_context(access)
            if claimed and session_id is not None:
                self._sessions.release(workspace, session_id, process_id)
            if created and session_id is not None:
                self._sessions.delete(workspace, session_id)
            raise
        # 5. 签发不透明的进程句柄；运行由调用方经 run_process 发起。
        return ProcessHandle(process)

    @overload
    def run_process(
        self,
        handle: ProcessHandle,
        *,
        stream: Literal[True] = True,
        detach_reason: str = STREAM_CLOSED_STOP_REASON,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    @overload
    def run_process(
        self,
        handle: ProcessHandle,
        *,
        stream: Literal[False],
        detach_reason: str = STREAM_CLOSED_STOP_REASON,
    ) -> Coroutine[Any, Any, NonStreamingResult]: ...

    def run_process(
        self,
        handle: ProcessHandle,
        *,
        stream: bool = True,
        detach_reason: str = STREAM_CLOSED_STOP_REASON,
    ) -> AsyncGenerator[dict[str, Any], None] | Coroutine[Any, Any, NonStreamingResult]:
        """运行句柄对应的已注册进程：``stream=True``（默认）返回流式事件的
        异步生成器，``stream=False`` 返回可 await 的非流式结果。

        认证与登记已在 :meth:`register_process` 完成；首次迭代生成器或
        await 结果时才创建进程 task，每个进程只运行一次。``detach_reason``
        是交付方离开（生成器被关闭或取消、等待被取消）时请求停止的原因，
        HTTP 交付在客户端断连时传入 ``client_disconnected``。
        """
        process = self._resolve(handle)
        if stream:
            return self._deliver_stream(process, detach_reason)
        return self._deliver_once(process, detach_reason)

    async def shutdown(self) -> None:
        """停止全部进程，等待进程 task 结束，超时后硬取消；返回前全部进程已关闭。

        从未运行的进程直接关闭；已启动的进程以 ``system_shutdown`` 请求停止。
        系统关闭不是用户的停止，已产生的部分输出不写入会话、不提交；已进入
        finalize 的进程拒绝停止，照常等待提交完成。超过
        ``shutdown_wait_seconds`` 仍未结束的进程 task 被取消并等待结束：进程
        照常在自己的 task 中收尾，finalize 中的提交仍要等它结束。会话设施
        只能在本方法之后清空。
        """
        tasks: list[asyncio.Task[TerminalOutput | None]] = []
        for process in self._process_table.snapshot():
            if process.task is None:
                tasks.append(self._close_unstarted(process, SYSTEM_SHUTDOWN_STOP_REASON))
                continue
            process.record.request_stop(SYSTEM_SHUTDOWN_STOP_REASON)
            tasks.append(process.task)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self._shutdown_wait_seconds)
        for task in pending:
            logger.warning("任务进程未在关闭等待内结束，硬取消: %s", task.get_name())
            task.cancel()
        if pending:
            await asyncio.wait(pending)

    async def close_process(self, handle: ProcessHandle) -> None:
        """注册入口的关闭路径：关闭句柄对应的进程，并等待进程 task 结束。

        进程从未运行（例如关停信号在注册期间到达）时，创建只执行关闭流程的
        进程 task；进程仍在运行时以 ``stream_closed`` 请求停止。等待可以被
        调用方取消，取消不影响进程自身的收尾。句柄已失效（进程已关闭、句柄
        不是本入口签发的）时是空操作。
        """
        process = self._lookup(handle)
        if process is None:
            return
        task = process.task
        if task is None:
            task = self._close_unstarted(process, STREAM_CLOSED_STOP_REASON)
        elif not process.record.is_terminal:
            self._request_stop(process, STREAM_CLOSED_STOP_REASON)
        await asyncio.wait((task,))

    def _start(
        self, process: TaskProcess, *, subscribe: bool
    ) -> asyncio.Task[TerminalOutput | None]:
        """首次运行时创建进程 task；每个进程只运行一次，只允许一个交付方。

        交付对象可能在进程关闭之后才开始迭代，因此这里重新确认进程仍在登记。
        ``subscribe`` 为假时输出通道从一开始就处于无订阅者状态。
        """
        if self._process_table.get(process.record.process_id) is not process:
            raise _unknown_handle_error(process.record.process_id)
        if process.task is not None:
            raise RuntimeError("任务进程已经开始运行或正在关闭：每个进程只允许一个交付方")
        if not subscribe:
            process.outputs.detach()
        process.task = asyncio.create_task(
            self._drive(process, run=True), name=f"task-process:{process.record.process_id}"
        )
        return process.task

    def _close_unstarted(
        self, process: TaskProcess, reason: str
    ) -> asyncio.Task[TerminalOutput | None]:
        """关闭从未运行的进程：记下停止原因，由只执行关闭流程的进程 task 收尾。"""
        process.record.request_stop(reason)
        process.task = asyncio.create_task(
            self._drive(process, run=False), name=f"task-process:{process.record.process_id}"
        )
        return process.task

    async def _drive(self, process: TaskProcess, *, run: bool) -> TerminalOutput | None:
        """进程 task 的驱动协程：唯一的退出路径。

        ``run`` 为真时由执行器推进各阶段并在收尾中完成关闭流程，返回终态
        产出；为假时进程从未运行，只执行关闭流程。无论以终态还是硬取消
        结束，``finally`` 都撤销进程 context、从进程表注销、释放会话认领，
        最后结束输出通道：订阅者读到结束标记时，进程已经关闭。
        """
        try:
            if run:
                return await self._runner.run(process)
            await self._runner.close(process)
            return None
        finally:
            try:
                self._release(process)
            finally:
                process.outputs.end()

    def _release(self, process: TaskProcess) -> None:
        """进程关闭后的收口：撤销进程 context、注销进程，再释放会话认领。"""
        record = process.record
        # 绑定 context 随进程关闭失效：无论 completed、失败、取消还是
        # 硬取消，进程结束后该凭据不再可用。
        self._access_gateway.revoke_context(record.access)
        self._process_table.close(process)
        # 只释放本进程的认领，会话不会被误交给后继进程。
        self._sessions.release(
            process.request.target_workspace, process.request.session_id, record.process_id
        )

    async def _deliver_stream(
        self,
        process: TaskProcess,
        detach_reason: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式交付：订阅输出通道，把阶段产出投影为流式事件；编排异常翻译为 error 事件。

        生成器被关闭或被取消时只退订，进程尚未记录终态时以 ``detach_reason``
        请求停止；交付方不等待进程 task，也不执行任何关闭步骤。正常结束时
        结束标记在进程关闭之后才到达，流随之结束。
        """
        self._start(process, subscribe=True)
        try:
            while (output := await process.outputs.receive()) is not None:
                if isinstance(output, ProcessFailed):
                    yield _stream_error(output.error)
                    continue
                for event in stream_events(
                    output,
                    process_id=process.record.process_id,
                    session_id=process.request.session_id,
                ):
                    yield event
        finally:
            self._detach(process, detach_reason)

    async def _deliver_once(self, process: TaskProcess, detach_reason: str) -> NonStreamingResult:
        """非流式交付：不订阅输出通道，等待进程 task 交出终态产出。

        进程 task 结束时进程已经关闭；编排异常随后原样上抛。调用方在等待中
        被取消时，以 ``detach_reason`` 请求停止后传播取消，进程在自己的 task
        中收尾。进程 task 被系统关闭硬取消时没有终态产出，取消同样传给调用方。
        """
        task = self._start(process, subscribe=False)
        try:
            terminal = await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                self._detach(process, detach_reason)
            raise
        if terminal is None:
            raise RuntimeError("任务进程结束时没有终态产出")
        return non_streaming_result(terminal, session_id=process.request.session_id)

    def _detach(self, process: TaskProcess, reason: str) -> None:
        """交付方离开：退订输出通道；进程尚未记录终态时，以交付方给出的原因请求停止。"""
        process.outputs.detach()
        if not process.record.is_terminal:
            self._request_stop(process, reason)

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
        """唯一的幂等取消方法：取消的依据作为参数。

        - 传入句柄：调用方是进程的所有者（交付方离开时的停止也经同一实现），
          不经进程控制授权，不接受另传的 ``access``；句柄已失效时返回
          ``not_found``，不发布事件。
        - 传入 ``process_id`` 与请求级 ``access``：控制请求（``/chat/stop``），
          经操作授权者的进程控制授权比对请求方与进程记录的驻留坐标；
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

    # ========== 内部辅助 ==========

    def _resolve(self, handle: ProcessHandle) -> TaskProcess:
        """把句柄解析为已登记的进程；句柄无效时显式失败（``process_handle_unknown``）。"""
        process = self._lookup(handle)
        if process is None:
            raise _unknown_handle_error(handle.process_id)
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


def _unknown_handle_error(process_id: str) -> WorkspaceDomainError:
    """句柄无效：不是本入口签发的句柄，或对应进程已关闭。"""
    return WorkspaceDomainError(
        "进程句柄无效：不是本入口签发的句柄，或对应进程已关闭",
        details={"reason": "process_handle_unknown", "process_id": process_id},
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
        return {
            "event": "error",
            "data": {"message": "请求未能完成，请检查权限、资源和会话状态", "code": error.code},
        }
    return {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}


__all__ = [
    "NonStreamingAgentOutcome",
    "NonStreamingCommandOutcome",
    "NonStreamingResult",
    "ProcessHandle",
    "TaskProcessService",
]
