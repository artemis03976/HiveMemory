"""任务进程执行器 — 所有进程共用的四阶段编排骨架。

骨架依次驱动 Gateway 分析 → Patchouli prepare → CPU 分配 → Actor 执行 →
有输出时封口交互记录并 finalize，对子系统的一切调用都经全局总线
的公开路由完成，Actor 执行经组合根注入的 CPU 端口（``workspace.contracts``
的 :class:`CPUPort`）完成。骨架只产出类型化的阶段产出（见
``workspace.process.outputs``），流式与非流式交付共用同一条阶段顺序、
同一组取消响应点与同一个关闭流程；两者的执行差异只有 CPU 以流式还是
非流式产出（流式逐条转交交互事件）。

执行器跨进程共享、不持有任何进程的状态：它只持有编排依赖，每次运行与
关闭都以 :class:`TaskProcess` 为参数，进程的全部状态（进程记录、任务参数、
工作集）都在进程上。执行器经进程记录使用绑定的访问 context 与事件发布器；
进程的登记、注销与 context 失效由注册入口（``workspace.process.service``）
负责。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.trace_context import (
    reset_trace_context,
    set_trace_context,
)
from hivememory.core.errors import WorkspaceDomainError, WorkspaceMismatchError
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from hivememory.core.protocol.gateway import GatewayDecision, GatewayIngressMode, GatewaySessionTurn
from hivememory.core.protocol.models import InteractionPayload
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUInputManifest,
    CPUPort,
)
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.command_terminal import command_terminal
from hivememory.workspace.process.outputs import (
    ActorEvent,
    CommandCompleted,
    Finalizing,
    InputsAllocated,
    ProcessFailed,
    ProcessOutput,
    ProcessStarted,
    RunCancelled,
    RunCompleted,
    RunFailed,
)
from hivememory.workspace.process.sealing import seal_interaction
from hivememory.workspace.process.table import (
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
)
from hivememory.workspace.process.task_process import TaskProcess
from hivememory.workspace.process.threads import ProcessThreads
from hivememory.workspace.process.working_set import ProcessWorkingSet
from hivememory.workspace.sessions import ConversationSessionService

logger = logging.getLogger(__name__)


class _ProcessCancelled(Exception):  # noqa: N818 - 设计要求使用私有领域分支名
    """进程编排内部的用户 stop 分支。"""

    def __init__(self, phase: ProcessPhase, reason: str) -> None:
        super().__init__(f"{phase.value} cancelled: {reason}")
        self.phase = phase
        self.reason = reason


async def _run_interruptible(
    control: ProcessRecord,
    phase: ProcessPhase,
    operation_factory: Callable[[], Awaitable[Any]],
) -> Any:
    """用进程编排自有 child task 包装一个可中断阶段。"""
    owner_task = asyncio.current_task()
    if owner_task is None:
        raise RuntimeError("_run_interruptible 必须运行在 asyncio task 中")
    entry_cancelling = owner_task.cancelling()

    if control.outcome is ProcessOutcome.STOP_REQUESTED:
        raise _ProcessCancelled(phase, control.stop_reason or "user_requested")

    async def invoke() -> Any:
        return await operation_factory()

    task = asyncio.create_task(invoke())
    control.bind_phase(phase, task)
    try:
        result = await task
        if control.outcome is ProcessOutcome.STOP_REQUESTED:
            raise _ProcessCancelled(
                phase,
                control.stop_reason or "user_requested",
            )
        return result
    except asyncio.CancelledError:
        if owner_task.cancelling() > entry_cancelling:
            raise
        if control.outcome is ProcessOutcome.STOP_REQUESTED and control.active_task is task:
            raise _ProcessCancelled(
                phase,
                control.stop_reason or "user_requested",
            ) from None
        raise
    finally:
        control.unbind_phase(task)


def _require_prepared_workspace(
    prepared: PreparedAgentRun,
    belong_to: WorkspaceIdentity,
) -> None:
    """拒绝 prepare 返回与任务目标 Workspace 不一致的结果。"""
    if prepared.belong_to != belong_to:
        raise WorkspaceMismatchError(
            "PreparedAgentRun 与任务目标 Workspace 不一致",
            details={
                "requested_workspace": belong_to.workspace_id,
                "prepared_workspace": prepared.belong_to.workspace_id,
            },
        )


class TaskProcessRunner:
    """任务进程执行器：四阶段编排骨架与关闭流程，跨进程共享。

    :meth:`run` 是唯一的编排骨架，:meth:`close` 是唯一的关闭流程；两者都
    以进程为参数，执行器自身不保存任何进程的状态。``run`` 的 ``stream``
    参数只决定 CPU 以流式还是非流式产出。执行器与 ``CPUAllocator`` 由组合根
    构建并注入注册入口。

    阶段调用由进程编排负责，每次调用前校验准入与任务目标 Workspace，
    不检查 actor 操作白名单。授权返回的 ``IdentityScope`` 保留进程 actor，
    使预检索仍按该 actor 的资源可见性过滤；scope 不进入 CPU 输入清单。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        cpu: CPUPort,
        sessions: ConversationSessionService,
        allocator: CPUAllocator,
        operation_authorizer: WorkspaceOperationAuthorizer,
        thread_contexts: ProcessThreads,
        gateway_request_timeout_ms: int = 8000,
        gateway_session_turn_limit: int = 3,
    ) -> None:
        self._sessions = sessions
        self._bus = global_bus
        self._cpu = cpu
        self._allocator = allocator
        self._authorizer = operation_authorizer
        self._thread_contexts = thread_contexts
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._gateway_session_turn_limit = gateway_session_turn_limit

    # ========== 编排骨架 ==========

    async def run(
        self,
        process: TaskProcess,
        *,
        stream: bool,
    ) -> AsyncGenerator[ProcessOutput, None]:
        """按四阶段顺序产出阶段产出，结束时（含提前关闭）执行 :meth:`close`。

        进程每记录一次终态（``record.mark_*``）就紧接着交出终态产出，中间
        没有 ``await``：终态是否已经交出因此由进程记录的终态推出。
        """
        record = process.record
        request = process.request
        working_set = process.working_set
        events = record.events
        process.owner_task = asyncio.current_task()
        trace_tokens = None
        try:
            trace_tokens = set_trace_context(
                process.trace_id,
                "TaskProcess.Stream" if stream else "TaskProcess.NonStreaming",
                "foreground",
            )
            yield ProcessStarted()

            # ---- Gateway：可被 stop 中断 ----
            record.enter_phase(ProcessPhase.GATEWAY)
            events.status(record)
            # 阶段授权只校验准入与目标；Gateway 沿用进程 actor 的 scope
            # 读取话题，actor 是否持有 READ 不影响这次系统编排。
            gateway_scope = self._authorize(process)
            gateway_result = await _run_interruptible(
                record,
                ProcessPhase.GATEWAY,
                lambda: self._bus.request(
                    GatewayPublicRoutes.PROCESS,
                    message=request.message,
                    identity_scope=gateway_scope,
                    ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
                    session_turns=tuple(
                        GatewaySessionTurn(
                            user_message=t.payload.user_message,
                            assistant_text=t.payload.assistant_final_text or "",
                        )
                        for t in self._sessions.get(
                            request.target_workspace, request.session_id
                        ).turns[-self._gateway_session_turn_limit :]
                    ),
                    request_timeout_ms=self._gateway_request_timeout_ms,
                ),
            )
            if gateway_result.kind == "command":
                # 命令只解析不执行：解析结果在这里转换为命令终态，命令不可用
                # 是命令自身的终态，进程仍按 completed 结局收口。
                command_result = command_terminal(gateway_result.command_parse_result)
                record.mark_completed()
                events.command_completed(record, command_id=command_result.command_id)
                yield CommandCompleted(command_result)
                return

            # ---- prepare 与 CPU 分配：不可中断，进入 Actor 前统一检查 stop ----
            prepared, manifest = await self._prepare_and_allocate(process, gateway_result.decision)
            yield InputsAllocated(prepared=prepared, manifest=manifest)

            # ---- Actor 执行：可被 stop 中断；流式逐条转交交互事件 ----
            record.enter_phase(ProcessPhase.ACTOR)
            events.status(record)
            # CPU 只持有派生线程 context；进程 context 留给全部阶段的授权。
            access = self._thread_contexts.derive_main_thread(process)
            # Actor 阶段只剩一个循环：流式与非流式都经 CPU 端口逐项拉取
            # （非流式只拉取一次），交互事件产出为 ActorEvent、终态结果作为
            # 执行结果。每次拉取都经 _run_interruptible 包装，停止请求的
            # 响应点不变；输出流登记进工作集，由关闭流程兜底关闭（含断流与
            # 取消路径）。
            try:
                cpu_run = self._cpu.execute(
                    manifest,
                    access=access,
                    generation_options=request.generation_options,
                    stream=stream,
                )
                cpu_output = cpu_run.output
                working_set.cpu_output = cpu_output
                execution_result: CPUExecutionResult | None = None
                try:
                    while True:
                        try:
                            item = await _run_interruptible(
                                record, ProcessPhase.ACTOR, lambda: anext(cpu_output)
                            )
                        except StopAsyncIteration:
                            break
                        if isinstance(item, CPUExecutionResult):
                            execution_result = item
                            break
                        yield ActorEvent(item)
                except _ProcessCancelled:
                    # 停止可能发生在两次拉取之间；先关闭流，CPU 才会同步回填结果槽。
                    await self._close_cpu_output(working_set)
                    if record.discards_output:
                        raise
                    # 沿用槽中的结局：CPU 在停止生效前已经完成时，结果仍是 completed。
                    execution_result = cpu_run.result_slot.result
                    if execution_result is None:
                        raise
                finally:
                    if execution_result is None:
                        # 断连或异常先撤销，不能在等待 CPU 关闭时保留线程权限。
                        self._thread_contexts.revoke_threads(working_set)
                    await self._close_cpu_output(working_set)
            finally:
                # 输出流关闭与线程撤销先于封口，不让 finalize 持有 actor 权限。
                self._thread_contexts.revoke_threads(working_set)
            # 迭代器在没有终态结果时结束属于端口语义错误，按进程失败处理。
            if execution_result is None:
                raise RuntimeError("CPU 输出流在没有终态执行结果的情况下结束")

            # 交付方离开或系统关闭时丢弃本轮；用户停止晚于 CPU 终态时沿用 CPU
            # 的结局，不把完整回复改记为 cancelled。
            if record.discards_output:
                raise _ProcessCancelled(record.phase, record.stop_reason or "stream_closed")
            working_set.execution_result = execution_result
            has_output = execution_result.has_output
            if has_output:
                # 封口、会话追加、进入 finalize 与登记提交任务之间没有 await/yield。
                # finalizing 一旦交给客户端，提交任务就已由进程持有。
                payload = seal_interaction(
                    user_message=request.message,
                    gateway_decision=gateway_result.decision,
                    assistant_final_text=execution_result.final_text,
                    turn_events=execution_result.turn_events,
                    model_used=execution_result.model_used,
                    used_attachments=working_set.used_attachments,
                    outcome=CPUExecutionStatus(execution_result.status).value,
                )
                self._sessions.append(
                    request.target_workspace,
                    request.session_id,
                    process_id=record.process_id,
                    agent_id=record.events.labels.agent_id,
                    payload=payload,
                    topic_id=prepared.topic_id,
                )
                if not record.try_enter_finalizing():
                    raise _ProcessCancelled(record.phase, record.stop_reason or "user_requested")
                working_set.finalization_task = asyncio.create_task(
                    self._finalize(process, prepared, payload)
                )
                events.status(record)
                yield Finalizing()
                await self._join_finalization(working_set.finalization_task)

            topic_id = prepared.topic_id if has_output else None
            self._mark_execution_terminal(process, execution_result)
            if execution_result.status == CPUExecutionStatus.CANCELLED.value:
                yield RunCancelled(
                    reason=record.stop_reason or "user_requested",
                    execution_result=execution_result,
                    topic_id=topic_id,
                )
            elif execution_result.status == CPUExecutionStatus.FAILED.value:
                yield RunFailed(execution_result, topic_id=topic_id)
            else:
                yield RunCompleted(execution_result=execution_result, topic_id=topic_id)
        except _ProcessCancelled as cancelled:
            record.mark_cancelled()
            events.cancelled(record, phase=cancelled.phase)
            yield RunCancelled(reason=record.stop_reason or "user_requested")
        except Exception as exc:
            if isinstance(exc, WorkspaceDomainError):
                logger.warning("任务进程领域错误: %s", exc.code)
            else:
                logger.exception("任务进程异常")
            record.mark_failed()
            events.failed(record, exc)
            yield ProcessFailed(exc)
        finally:
            try:
                await self.close(process)
            finally:
                if trace_tokens is not None:
                    reset_trace_context(trace_tokens)

    async def _finalize(
        self, process: TaskProcess, prepared: PreparedAgentRun, payload: InteractionPayload
    ) -> None:
        """把已封口交互交回 Patchouli；任务归进程持有，scope 仅在调用链内存在。"""
        await self._bus.request(
            PatchouliRoutes.FINALIZE_AGENT_RUN,
            prepared_run=prepared,
            payload=payload,
            identity_scope=self._authorize(process),
        )
        process.working_set.hand_off_prepared()

    @staticmethod
    async def _join_finalization(task: asyncio.Task[None]) -> None:
        """屏蔽交付方取消直到提交结束，再传播取消；不留下脱离进程的后台任务。"""
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
            except Exception:
                break
        if cancelled:
            # 取出异常以免未观察告警；关闭流程仍会按同一任务的结果收尾。
            task.exception()
            raise asyncio.CancelledError
        task.result()

    @staticmethod
    def _mark_execution_terminal(process: TaskProcess, result: CPUExecutionResult) -> None:
        """仅在提交完成后发布对应终态，重复关闭不重复发布。"""
        record = process.record
        if record.is_terminal:
            return
        if result.status == CPUExecutionStatus.CANCELLED.value:
            record.mark_cancelled()
            record.events.cancelled(record)
        elif result.status == CPUExecutionStatus.FAILED.value:
            record.mark_failed()
            record.events.failed(record, failure_reason=result.failure_reason)
        else:
            record.mark_completed()
            record.events.completed(record)

    async def _prepare_and_allocate(
        self,
        process: TaskProcess,
        decision: GatewayDecision,
    ) -> tuple[PreparedAgentRun, CPUInputManifest]:
        """Profile 解析、Patchouli prepare 与 CPU 分配，最后检查一次停止请求。

        Profile 解析与附件租借由 CPUAllocator 在副作用前执行阶段授权；
        prepare 同样只检查准入与目标。finalize 在调用前再次校验，不要求
        actor 持有交互提交操作。
        """
        record = process.record
        request = process.request
        working_set = process.working_set
        record.enter_phase(ProcessPhase.PREPARE)
        # Profile 在 prepare 之前解析：缺少图纸的请求不会留下预建话题或 LRU 结算的副作用。
        agent_profile = await self._allocator.resolve_agent_profile(
            access=record.access,
            target_workspace=request.target_workspace,
        )
        # prepare 做话题准备与检索，资源可见性仍使用进程 actor。
        prepare_scope = self._authorize(process)
        prepared: PreparedAgentRun = await self._bus.request(
            PatchouliRoutes.PREPARE_AGENT_RUN,
            identity_scope=prepare_scope,
            interaction_id=record.process_id,
            gateway_decision=decision,
            enable_memory_retrieval=request.enable_memory_retrieval,
        )
        # 先写入工作集再校验：关闭流程仍按本进程目标授权；资源 owner
        # 拒绝清理越域结果，避免补偿路径删除其他 Workspace 的话题。
        working_set.prepared = prepared
        record.events.bind_topic(prepared.topic_id)
        _require_prepared_workspace(prepared, request.target_workspace)

        # CPU 分配的其余部分（附件租借与编译、清单组装）仍在 PREPARE 阶段内完成。
        manifest = self._allocator.allocate(
            working_set,
            process_id=record.process_id,
            labels=record.events.labels,
            user_message=request.message,
            agent_profile=agent_profile,
            selections=list(request.attachments),
            session_history=self._sessions.get(request.target_workspace, request.session_id).turns,
            access=record.access,
            target_workspace=request.target_workspace,
        )

        # 取消检查：只在进入 Actor 之前检查一次。prepare 与分配
        # 期间收到的 stop 请求都在此生效，已取得的租借由关闭流程释放。
        if record.outcome is ProcessOutcome.STOP_REQUESTED:
            raise _ProcessCancelled(ProcessPhase.PREPARE, record.stop_reason or "user_requested")
        return prepared, manifest

    def _authorize(self, process: TaskProcess) -> IdentityScope:
        """以进程记录绑定的 context 对任务目标 workspace 执行阶段授权。

        返回操作授权者组装的可信 scope，供紧随的阶段路由使用；授权失败
        沿异常路径按进程失败收口。
        """
        return self._authorizer.authorize_stage(
            process.record.access,
            process.request.target_workspace,
        )

    # ========== 关闭流程 ==========

    async def close(self, process: TaskProcess) -> None:
        """进程关闭：终态兜底、释放租借、关闭 CPU 输出流、补偿 prepare。

        无论完成、取消、失败、断流还是分配失败，都经此关闭。租借释放必须先于
        任何 await 同步执行：owner task 在关闭子流或 cleanup 期间被取消时，
        释放不能依赖这些 await 完成。context 失效与进程注销由注册入口在
        本流程之后执行。本方法可重复调用——
        run() 的收尾与注册入口的关闭路径都会调用它：终态兜底只在进程尚无
        终态时执行，工作集中的每项资源只取出一次，已经释放的不会再次释放。
        """
        record = process.record
        working_set = process.working_set
        # 分支：交付方提前关闭（如客户端断流），且此前没有交出终态。
        owner_task = process.owner_task
        owner_is_cancelling = owner_task is not None and owner_task.cancelling() > 0
        if (
            working_set.finalization_task is None
            and not record.is_terminal
            and not owner_is_cancelling
        ):
            if record.outcome is ProcessOutcome.RUNNING:
                record.request_stop("stream_closed")
            record.mark_cancelled()
            record.events.closed_before_terminal(record)

        # 撤销线程 context 与租借释放都先于任何 await：即使后续
        # CPU 输出流关闭被取消，也不能留下有效线程 context 或未释放租借。
        self._thread_contexts.revoke_threads(working_set)
        # 附件文本在 CPU 分配时已编译进清单，Actor 执行不再读取租借内容。
        self._allocator.release(working_set)
        await self._close_cpu_output(working_set)
        if working_set.finalization_task is not None and not record.is_terminal:
            try:
                await self._join_finalization(working_set.finalization_task)
            except Exception as exc:
                if not record.is_terminal:
                    record.mark_failed()
                    record.events.failed(record, exc)
                logger.warning("关闭期间 finalize 失败", exc_info=True)
            else:
                if working_set.execution_result is not None:
                    self._mark_execution_terminal(process, working_set.execution_result)
        # 只要 prepare 成功但 finalize 未成功，就清理可能的新建空 Topic。
        prepared = working_set.take_prepared_for_cleanup()
        if prepared is not None:
            try:
                await self._bus.request(
                    PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN,
                    prepared_run=prepared,
                    # 补偿仍校验准入与任务目标，context 此时尚未失效。
                    identity_scope=self._authorize(process),
                )
            except Exception:
                logger.warning("清理 prepared run 失败", exc_info=True)

    @staticmethod
    async def _close_cpu_output(working_set: ProcessWorkingSet) -> None:
        """关闭工作集登记的 CPU 输出流（只取出一次），关闭失败只记录警告。"""
        cpu_output = working_set.take_cpu_output()
        if cpu_output is None:
            return
        try:
            await cpu_output.aclose()
        except Exception:
            logger.warning("关闭 CPU 输出流失败", exc_info=True)


__all__ = ["TaskProcessRunner"]
