"""任务进程 — 一次进程的状态容器与四阶段编排骨架。

骨架依次驱动 Gateway 分析 → Patchouli prepare → CPU 分配 → Actor 执行 →
（仅 completed）封口交互记录并 finalize，对子系统的一切调用都经全局总线
的公开路由完成，Actor 执行经组合根注入的 CPU 端口（``workspace.contracts``
的 :class:`CPUPort`）完成。骨架只产出类型化的阶段产出（见
``workspace.process.outputs``），流式与非流式交付共用同一条阶段顺序、
同一组取消响应点与同一个关闭流程；两者的执行差异只有 CPU 以流式还是
非流式产出（流式逐条转交交互事件），以及 finalize 之后读取话题池
（只服务于流式 done 事件）。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import WorkspaceDomainError, WorkspaceMismatchError
from hivememory.core.models import AttachmentSelectionRequest, IdentityScope
from hivememory.core.protocol.gateway import GatewayDecision, GatewayIngressMode
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUInputManifest,
    CPUOutput,
    CPUPort,
)
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.command_terminal import command_terminal
from hivememory.workspace.process.events import TaskProcessEventEmitter
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
    TerminalOutput,
)
from hivememory.workspace.process.sealing import seal_interaction
from hivememory.workspace.process.table import (
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessTable,
)

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


def _require_prepared_scope(
    prepared: PreparedAgentRun,
    identity_scope: IdentityScope,
) -> None:
    """拒绝 prepare 返回与进程表不一致的请求级 scope。"""
    if prepared.identity_scope != identity_scope:
        raise WorkspaceMismatchError(
            "PreparedAgentRun 与进程记录的身份作用域不一致",
            details={
                "requested_workspace": identity_scope.workspace_identity.workspace_id,
                "prepared_workspace": prepared.identity_scope.workspace_identity.workspace_id,
            },
        )


@dataclass(frozen=True, kw_only=True)
class ProcessRequest:
    """一次任务请求的入口参数（``process_id`` 由 server 入口冻结）。

    ``message`` 是交给 Gateway 分析的指令文本：主动请求是用户本次发出的消息。
    """

    message: str
    identity_scope: IdentityScope
    process_id: str
    enable_memory_retrieval: bool = True
    generation_options: dict[str, Any] | None = None
    attachments: tuple[AttachmentSelectionRequest, ...] = ()


class TaskProcess:
    """一次任务进程：进程记录、工作集与事件投影的容器。

    :meth:`run` 是唯一的编排骨架，:meth:`close` 是唯一的关闭流程；实例只
    运行一次。``stream`` 只决定 CPU 以流式还是非流式产出。
    """

    def __init__(
        self,
        request: ProcessRequest,
        *,
        stream: bool,
        global_bus: GlobalSystemBus,
        process_table: ProcessTable,
        allocator: CPUAllocator,
        cpu: CPUPort,
        events: TaskProcessEventEmitter,
        gateway_request_timeout_ms: int,
    ) -> None:
        self._request = request
        self._stream = stream
        self._bus = global_bus
        self._process_table = process_table
        self._allocator = allocator
        self._cpu = cpu
        self._gateway_request_timeout_ms = gateway_request_timeout_ms

        self._record = ProcessRecord(
            identity_scope=request.identity_scope,
            process_id=request.process_id,
        )
        self._working_set = allocator.new_working_set()
        self._trace_id = generate_trace_id("task")
        self._events = events.for_process(self._record, trace_id=self._trace_id)

        self._trace_tokens: Any = None
        self._owner_task: asyncio.Task[Any] | None = None
        self._cpu_output: AsyncGenerator[CPUOutput, None] | None = None
        # 终态产出是否已经交出；关闭时据此判断是否需要按断流收口。
        self._terminal_published = False
        # finalize 成功后 Patchouli 已接管本轮交互，不再清理 prepared run。
        self._prepared_finalized = False

    # ========== 编排骨架 ==========

    async def run(self) -> AsyncGenerator[ProcessOutput, None]:
        """按四阶段顺序产出阶段产出，结束时（含提前关闭）执行 :meth:`close`。"""
        record = self._record
        request = self._request
        self._owner_task = asyncio.current_task()
        try:
            self._trace_tokens = set_trace_context(
                self._trace_id,
                "TaskProcess.Stream" if self._stream else "TaskProcess.NonStreaming",
                "foreground",
            )
            self._process_table.register(record)
            self._events.created()
            yield ProcessStarted()

            # ---- Gateway：可被 stop 中断 ----
            record.enter_phase(ProcessPhase.GATEWAY)
            self._events.status()
            gateway_result = await _run_interruptible(
                record,
                ProcessPhase.GATEWAY,
                lambda: self._bus.request(
                    GlobalRoutes.GATEWAY_PROCESS,
                    message=request.message,
                    identity_scope=request.identity_scope,
                    ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
                    request_timeout_ms=self._gateway_request_timeout_ms,
                ),
            )
            if gateway_result.kind == "command":
                # 命令只解析不执行：解析结果在这里转换为命令终态，命令不可用
                # 是命令自身的终态，进程仍按 completed 结局收口。
                command_result = command_terminal(gateway_result.command_parse_result)
                record.mark_completed()
                self._events.command_completed(command_id=command_result.command_id)
                yield self._terminal(CommandCompleted(command_result))
                return

            # ---- prepare 与 CPU 分配：不可中断，进入 Actor 前统一检查 stop ----
            prepared, manifest = await self._prepare_and_allocate(gateway_result.decision)
            yield InputsAllocated(prepared=prepared, manifest=manifest)

            # ---- Actor 执行：可被 stop 中断；流式逐条转交交互事件 ----
            record.enter_phase(ProcessPhase.ACTOR)
            self._events.status()
            # Actor 阶段只剩一个循环：流式与非流式都经 CPU 端口逐项拉取
            # （非流式只拉取一次），交互事件产出为 ActorEvent、终态结果作为
            # 执行结果。每次拉取都经 _run_interruptible 包装，停止请求的
            # 响应点不变；迭代器交给关闭流程统一关闭（含断流与取消路径）。
            cpu_output = self._cpu.execute(
                manifest,
                generation_options=request.generation_options,
                stream=self._stream,
            )
            self._cpu_output = cpu_output
            execution_result: CPUExecutionResult | None = None
            while True:
                try:
                    item = await _run_interruptible(
                        record,
                        ProcessPhase.ACTOR,
                        lambda: anext(cpu_output),
                    )
                except StopAsyncIteration:
                    break
                if isinstance(item, CPUExecutionResult):
                    # 终态结果恰好出现一次且是最后一项；拿到即结束拉取。
                    execution_result = item
                    break
                yield ActorEvent(item)
            # 拿到终态结果后立即关闭 CPU 输出流，让 CPU 在 finalize 之前释放
            # 自己的资源（finalize 要等交互被应用）；关闭流程中的关闭仅作兜底。
            await self._close_cpu_output()
            # 迭代器在没有终态结果时结束属于端口语义错误，按进程失败处理。
            if execution_result is None:
                raise RuntimeError("CPU 输出流在没有终态执行结果的情况下结束")

            if execution_result.status == CPUExecutionStatus.CANCELLED.value:
                record.mark_cancelled()
                self._events.cancelled()
                yield self._terminal(
                    RunCancelled(
                        reason=record.stop_reason or "user_requested",
                        execution_result=execution_result,
                    )
                )
                return
            if execution_result.status == CPUExecutionStatus.FAILED.value:
                record.mark_failed()
                self._events.failed()
                yield self._terminal(RunFailed(execution_result))
                return

            # ---- finalize（仅 completed）：进入后拒绝取消 ----
            if not record.try_enter_finalizing():
                raise _ProcessCancelled(record.phase, record.stop_reason or "user_requested")
            self._events.status()
            yield Finalizing()
            # 进程在调用 finalize 前封口交互记录（Q-14）：这是骨架唯一从
            # CPU 执行结果提取字段组装交互输入的地方；组装失败沿异常路径
            # 按进程失败处理，走现有关闭流程。
            payload = seal_interaction(
                user_message=request.message,
                gateway_decision=gateway_result.decision,
                assistant_final_text=execution_result.final_text,
                turn_events=execution_result.turn_events,
                model_used=execution_result.model_used,
                materialize_tasks=execution_result.materialize_tasks,
                # 附件编译冻结的实际使用引用（被预算跳过的附件不在其中）。
                used_attachments=self._working_set.used_attachments,
            )
            memory_tasks = await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=prepared,
                payload=payload,
            )
            self._prepared_finalized = True
            memory_task_ids = [memory_task.task_id for memory_task in (memory_tasks or [])]
            # 结算后的话题池只服务于流式 done 事件的前端刷新。
            pool_topics = await self._list_final_pool_topics(prepared) if self._stream else []

            record.mark_completed()
            self._events.completed(memory_task_ids=memory_task_ids)
            yield self._terminal(
                RunCompleted(
                    execution_result=execution_result,
                    memory_task_ids=memory_task_ids,
                    pool_topics=pool_topics,
                )
            )
        except _ProcessCancelled as cancelled:
            record.mark_cancelled()
            self._events.cancelled(phase=cancelled.phase)
            yield self._terminal(RunCancelled(reason=record.stop_reason or "user_requested"))
        except Exception as exc:
            if isinstance(exc, WorkspaceDomainError):
                logger.warning("任务进程领域错误: %s", exc.code)
            else:
                logger.exception("任务进程异常")
            record.mark_failed()
            self._events.failed(exc)
            yield self._terminal(ProcessFailed(exc))
        finally:
            await self.close()

    async def _prepare_and_allocate(
        self,
        decision: GatewayDecision,
    ) -> tuple[PreparedAgentRun, CPUInputManifest]:
        """Profile 解析、Patchouli prepare 与 CPU 分配，最后检查一次停止请求。"""
        record = self._record
        request = self._request
        record.enter_phase(ProcessPhase.PREPARE)
        # Profile 暂时先于 prepare 解析（中间态），原因见 CPUAllocator.resolve_agent_profile。
        agent_profile = await self._allocator.resolve_agent_profile(request.identity_scope)
        prepared: PreparedAgentRun = await self._bus.request(
            GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
            identity_scope=request.identity_scope,
            interaction_id=request.process_id,
            gateway_decision=decision,
            enable_memory_retrieval=request.enable_memory_retrieval,
        )
        # 先写入工作集再校验：scope 不一致时关闭流程仍需把它交回 cleanup，
        # 以补偿 prepare 可能已经预建的 Topic。
        self._working_set.prepared = prepared
        self._events.bind_topic(prepared.topic_id)
        _require_prepared_scope(prepared, request.identity_scope)

        # CPU 分配的其余部分（附件租借与编译、清单组装）仍在 PREPARE 阶段内完成。
        manifest = self._allocator.allocate(
            self._working_set,
            process_id=record.process_id,
            identity_scope=request.identity_scope,
            user_message=request.message,
            agent_profile=agent_profile,
            selections=list(request.attachments),
        )

        # 取消检查（Q-15）：只在进入 Actor 之前检查一次。prepare 与分配
        # 期间收到的 stop 请求都在此生效，已取得的租借由关闭流程释放。
        if record.outcome is ProcessOutcome.STOP_REQUESTED:
            raise _ProcessCancelled(ProcessPhase.PREPARE, record.stop_reason or "user_requested")
        return prepared, manifest

    def _terminal(self, output: TerminalOutput) -> TerminalOutput:
        self._terminal_published = True
        return output

    # ========== 关闭流程 ==========

    async def close(self) -> None:
        """进程关闭：终态兜底、释放租借、关闭 CPU 输出流、补偿 prepare、注销进程。

        无论完成、取消、失败、断流还是分配失败，都经此关闭。租借释放必须先于
        任何 await 同步执行：owner task 在关闭子流或 cleanup 期间被取消时，
        释放不能依赖这些 await 完成；进程记录的注销放在内层 ``finally``。
        """
        record = self._record
        # 分支：交付方提前关闭（如客户端断流），且此前没有交出终态。
        owner_is_cancelling = self._owner_task is not None and self._owner_task.cancelling() > 0
        if not self._terminal_published and not owner_is_cancelling:
            if record.outcome is ProcessOutcome.RUNNING:
                record.request_stop("stream_closed")
            record.mark_cancelled()
            self._events.closed_before_terminal()
            self._terminal_published = True

        # 附件文本在 CPU 分配时已编译进清单，Actor 执行不再读取租借内容。
        self._working_set.release()
        try:
            await self._close_cpu_output()
            # 只要 prepare 成功但 finalize 未成功，就清理可能的新建空 Topic。
            if self._working_set.prepared is not None and not self._prepared_finalized:
                try:
                    await self._bus.request(
                        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                        prepared_run=self._working_set.prepared,
                    )
                except Exception:
                    logger.warning("清理 prepared run 失败", exc_info=True)
        finally:
            self._process_table.close(record)
            if self._trace_tokens is not None:
                reset_trace_context(self._trace_tokens)

    async def _close_cpu_output(self) -> None:
        """幂等关闭 CPU 输出流：先摘除引用，关闭失败只记录警告。"""
        cpu_output, self._cpu_output = self._cpu_output, None
        if cpu_output is None:
            return
        try:
            await cpu_output.aclose()
        except Exception:
            logger.warning("关闭 CPU 输出流失败", exc_info=True)

    async def _list_final_pool_topics(
        self,
        prepared_run: PreparedAgentRun,
    ) -> list[dict[str, Any]]:
        try:
            topics = await self._bus.request(
                GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
                identity_scope=prepared_run.identity_scope,
                include_empty=True,
            )
        except Exception:
            logger.warning("Failed to load final topic pool after finalize.", exc_info=True)
            return []
        return [topic.model_dump(mode="json") for topic in (topics or [])]


__all__ = ["ProcessRequest", "TaskProcess"]
