"""任务进程执行器 — 所有进程共用的四阶段编排骨架。

骨架依次驱动 Gateway 分析 → Patchouli prepare → CPU 分配 → Actor 执行 →
有输出时封口交互记录并 finalize，对子系统的一切调用都经全局总线
的公开路由完成，Actor 执行经组合根注入的 CPU 端口（``workspace.contracts``
的 :class:`CPUPort`）完成。骨架运行在进程自己的 task 中，只把类型化的阶段
产出写入进程的输出通道（见 ``workspace.process.outputs``）：CPU 逐项推送
交互事件，订阅通道的流式交付逐条转交。进入 Gateway 之前，骨架先请求
Patchouli 在后台预取推理模型。

执行器跨进程共享、不持有任何进程的状态：它只持有编排依赖，每次运行与
关闭都以 :class:`TaskProcess` 为参数，进程的全部状态（进程记录、任务参数、
工作集、输出通道）都在进程上。执行器经进程记录使用绑定的访问 context 与
事件发布器；进程 task 的创建、进程的登记、注销与 context 失效由注册入口
（``workspace.process.service``）负责。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.cancellation import join_shielded
from hivememory.components.trace_context import (
    reset_trace_context,
    set_trace_context,
)
from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import AgentProfile, IdentityScope
from hivememory.core.protocol.gateway import GatewayDecision, GatewayIngressMode, GatewaySessionTurn
from hivememory.core.protocol.models import InteractionPayload
from hivememory.core.protocol.sealing import seal_interaction
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
    ProcessStarted,
    RunCancelled,
    RunCompleted,
    RunFailed,
    TerminalOutput,
)
from hivememory.workspace.process.table import (
    CLIENT_DISCONNECTED_STOP_REASON,
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
)
from hivememory.workspace.process.task_process import TaskProcess
from hivememory.workspace.process.threads import ProcessThreads
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
    """用进程编排自有 child task 包装一个可中断阶段。

    停止请求只取消这个 child task，在此翻译为 :class:`_ProcessCancelled`；
    进程 task 自身被取消（系统关闭的硬取消）时按进入时记录的
    ``cancelling()`` 识别，原样向上传播。child task 可能把取消转换为返回值
    （CPU 在最外层这样做），这时等待方的取消请求会随之丢失，因此 child task
    正常结束后同样按 ``cancelling()`` 判断，重新抛出硬取消。
    """
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
        if owner_task.cancelling() > entry_cancelling:
            raise asyncio.CancelledError
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


@dataclass
class _ReturnedResult:
    """Actor 子 task 与阶段函数共享的局部状态：CPU 返回的终态结果。

    子 task 结束后，停止（晚于 CPU 返回才被识别）与硬取消都会以异常离开
    ``_run_interruptible``，子 task 的返回值随之丢失；阶段函数改从这里取回。
    """

    result: CPUExecutionResult | None = None


class TaskProcessRunner:
    """任务进程执行器：四阶段编排骨架与关闭流程，跨进程共享。

    :meth:`run` 是唯一的编排骨架，:meth:`close` 是唯一的关闭流程；两者都
    以进程为参数、都运行在进程 task 中，执行器自身不保存任何进程的状态。
    执行器与 ``CPUAllocator`` 由组合根构建并注入注册入口。

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
        threads: ProcessThreads,
        gateway_request_timeout_ms: int = 8000,
        gateway_session_turn_limit: int = 3,
    ) -> None:
        self._sessions = sessions
        self._bus = global_bus
        self._cpu = cpu
        self._allocator = allocator
        self._authorizer = operation_authorizer
        self._threads = threads
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._gateway_session_turn_limit = gateway_session_turn_limit

    # ========== 编排骨架 ==========

    async def run(self, process: TaskProcess) -> None:
        """在进程 task 中按四阶段顺序推进，把终态产出写入输出通道；结束时执行 :meth:`close`。

        停止与普通异常都收口为终态产出，只有硬取消以 ``CancelledError`` 结束；
        无论哪种结局，关闭流程都在 ``finally`` 中执行一次。
        """
        trace_tokens = set_trace_context(process.trace_id, "TaskProcess", "foreground")
        try:
            await process.outputs.send(await self._execute(process))
        finally:
            try:
                await self.close(process)
            finally:
                reset_trace_context(trace_tokens)

    async def _execute(self, process: TaskProcess) -> TerminalOutput:
        """推进各阶段，把停止与普通异常收口为终态产出；硬取消原样传播。

        进程每记录一次终态（``record.mark_*``）都紧接着返回终态产出，返回
        之后直到写入通道都没有会挂起的 ``await``：终态是否已经写入因此由
        进程记录的终态推出。
        """
        record = process.record
        events = record.events
        try:
            return await self._run_stages(process)
        except _ProcessCancelled as cancelled:
            record.mark_cancelled()
            events.cancelled(record, phase=cancelled.phase)
            return RunCancelled(reason=record.stop_reason or "user_requested")
        except Exception as exc:
            if isinstance(exc, WorkspaceDomainError):
                logger.warning("任务进程领域错误: %s", exc.code)
            else:
                logger.exception("任务进程异常")
            record.mark_failed()
            events.failed(record, exc)
            return ProcessFailed(exc)

    async def _run_stages(self, process: TaskProcess) -> TerminalOutput:
        """四阶段的主路径：停止以 :class:`_ProcessCancelled` 结束，异常原样抛出。"""
        record = process.record
        request = process.request
        events = record.events
        await process.outputs.send(ProcessStarted())

        # ---- Gateway：可被 stop 中断 ----
        record.enter_phase(ProcessPhase.GATEWAY)
        events.status(record)
        # 阶段授权只校验准入与目标；Gateway 沿用进程 actor 的 scope
        # 读取话题，actor 是否持有 READ 不影响这次系统编排。
        gateway_scope = self._authorize(process)
        await self._prefetch_models()
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
                        assistant_text=t.payload.reply_text,
                        topic_id=t.topic_id,
                    )
                    for t in self._sessions.get(request.target_workspace, request.session_id).turns[
                        -self._gateway_session_turn_limit :
                    ]
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
            return CommandCompleted(command_result)

        # ---- prepare 与 CPU 分配：不可中断，进入 Actor 前统一检查 stop ----
        prepared, agent_profile, manifest = await self._prepare_and_allocate(
            process, gateway_result.decision
        )
        await process.outputs.send(InputsAllocated(prepared=prepared))

        # ---- Actor 执行：可被 stop 中断；交互事件经输出通道逐条转交 ----
        execution_result = await self._run_actor(process, manifest, agent_profile)
        # 交付方离开或系统关闭时丢弃本轮；用户停止晚于 CPU 终态时沿用 CPU
        # 的结局，不把完整回复改记为 cancelled。
        if record.discards_output:
            raise _ProcessCancelled(
                record.phase, record.stop_reason or CLIENT_DISCONNECTED_STOP_REASON
            )
        has_output = execution_result.has_output
        if has_output:
            await self._seal_and_finalize(process, prepared, execution_result)

        topic_id = prepared.topic_id if has_output else None
        self._mark_execution_terminal(process, execution_result)
        if execution_result.status is CPUExecutionStatus.CANCELLED:
            return RunCancelled(
                reason=record.stop_reason or "user_requested",
                execution_result=execution_result,
                topic_id=topic_id,
            )
        if execution_result.status is CPUExecutionStatus.FAILED:
            return RunFailed(execution_result, topic_id=topic_id)
        return RunCompleted(execution_result=execution_result, topic_id=topic_id)

    async def _prefetch_models(self) -> None:
        """在 Gateway 之前请求 Patchouli 在后台加载推理模型，让加载与 Gateway 的 LLM 调用并行。

        在阶段授权之后发出，准入已失效的进程不触发加载。路由只开始加载、立即返回；
        Gateway 判定不需要检索时模型同样会加载，由空闲卸载回收。预取失败只记录日志，
        不影响任务进程。
        """
        try:
            await self._bus.request(PatchouliRoutes.PREFETCH_MODELS)
        except Exception as exc:
            logger.warning("推理模型预取请求失败，任务进程照常继续: %s", exc)

    async def _run_actor(
        self, process: TaskProcess, manifest: CPUInputManifest, agent_profile: AgentProfile
    ) -> CPUExecutionResult:
        """Actor 阶段：派生主线程，在一个可中断子 task 中等待 CPU，返回本轮采用的结果。

        子 task 直接等待 CPU 端口的 ``execute``，交互事件经 sink 写入输出
        通道；停止取消子 task 时，无论 CPU 在等模型还是阻塞在通道写入上，
        都立即生效。结果的取舍：CPU 以返回值表达停止后的结局，返回之后进程
        以它为准，不再按停止标志改写为取消；CPU 原样传播取消、没有结果时按
        无输出的取消传播。

        主线程以 prepare 之前解析的图纸派生，派生时以主线程 context 读取操作
        快照；准入此时已被撤销则派生失败，进程按失败收口。子线程随 CALL 返回
        由 CPU 结束；全部线程只在本阶段的 ``finally`` 中兜底结束一次（关闭
        流程只是再兜底），结束即撤销线程 context；停止被接受时线程已随之结束，
        迟到的提交在此之前就被拒绝。检查点也只在这里接受一次：结束线程之后、
        判断丢弃与有无输出之前。普通执行异常没有可接受的结果。进程 task 被
        硬取消而 CPU 把取消转换为返回值时，先结束线程、接受其中已完成的
        检查点，再重新抛出取消。
        """
        record = process.record
        record.enter_phase(ProcessPhase.ACTOR)
        record.events.status(record)
        # CPU 只持有主线程的执行线程；进程 context 留给全部阶段的授权。
        thread = self._threads.derive_main(process, agent_profile)
        returned = _ReturnedResult()

        async def publish(event: dict[str, Any]) -> None:
            await process.outputs.send(ActorEvent(event))

        async def drive() -> None:
            returned.result = await self._cpu.execute(
                manifest,
                thread=thread,
                generation_options=process.request.generation_options,
                events=publish,
            )

        try:
            await _run_interruptible(record, ProcessPhase.ACTOR, drive)
        except _ProcessCancelled:
            # 停止：CPU 返回的结局优先（停止晚于 CPU 终态时仍是 CPU 的结局）；
            # CPU 没有返回结果时按无输出取消。
            if returned.result is None:
                raise
            return returned.result
        finally:
            # 先结束全部线程（撤销线程 context），再接受检查点；两者都先于
            # 封口、finalize 与终态交付，硬取消时先于重新抛出取消。
            self._threads.end_all(process)
            self._accept_checkpoint(process, returned.result)
        # drive 正常结束时已经写入 CPU 返回的结果。
        assert returned.result is not None
        return returned.result

    async def _seal_and_finalize(
        self,
        process: TaskProcess,
        prepared: PreparedAgentRun,
        execution_result: CPUExecutionResult,
    ) -> None:
        """封口交互记录并同步写入会话，再把提交交给 Patchouli 并等待它完成。

        封口、会话追加、进入 finalize 与创建提交任务之间没有 await：提交任务
        在写入 ``Finalizing`` 之前创建，交付方一旦收到 finalizing，提交就已由
        进程持有。交付方离开打断不了提交；提交以 ``join_shielded`` 等待，只有
        硬取消会遇到它，而硬取消也要等提交结束，提交成功时照常记录 CPU 的
        结局，再传播取消。
        """
        record = process.record
        request = process.request
        payload = seal_interaction(
            turn_events=execution_result.turn_events,
            model_used=execution_result.model_used,
            used_attachments=process.working_set.used_attachments,
            outcome=execution_result.status.value,
            final_text=execution_result.final_text,
        )
        self._sessions.append(
            request.target_workspace,
            request.session_id,
            process_id=record.process_id,
            agent_id=record.labels.agent_id,
            payload=payload,
            topic_id=prepared.topic_id,
        )
        if not record.try_enter_finalizing():
            raise _ProcessCancelled(record.phase, record.stop_reason or "user_requested")
        finalization = asyncio.create_task(self._finalize(process, prepared, payload))
        record.events.status(record)
        await process.outputs.send(Finalizing())
        try:
            await join_shielded(finalization)
        except asyncio.CancelledError:
            if not finalization.cancelled():
                if finalization.exception() is None:
                    self._mark_execution_terminal(process, execution_result)
                else:
                    logger.warning("硬取消期间 finalize 失败", exc_info=finalization.exception())
            raise

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
    def _mark_execution_terminal(process: TaskProcess, result: CPUExecutionResult) -> None:
        """仅在提交完成后发布对应终态；进程已有终态时不重复发布。"""
        record = process.record
        if record.is_terminal:
            return
        if result.status is CPUExecutionStatus.CANCELLED:
            record.mark_cancelled()
            record.events.cancelled(record)
        elif result.status is CPUExecutionStatus.FAILED:
            record.mark_failed()
            record.events.failed(record, failure_reason=result.failure_reason)
        else:
            record.mark_completed()
            record.events.completed(record)

    async def _prepare_and_allocate(
        self,
        process: TaskProcess,
        decision: GatewayDecision,
    ) -> tuple[PreparedAgentRun, AgentProfile, CPUInputManifest]:
        """Profile 解析、Patchouli prepare 与 CPU 分配，最后检查一次停止请求。

        Profile 解析与附件租借由 CPUAllocator 在副作用前执行阶段授权；
        prepare 同样只检查准入与目标。finalize 在调用前再次校验，不要求
        actor 持有交互提交操作。解析出的图纸随主线程交给 CPU，不进入输入清单。
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
        working_set.prepared = prepared
        record.events.bind_topic(prepared.topic_id)

        # CPU 分配的其余部分（附件租借与编译、清单组装）仍在 PREPARE 阶段内完成。
        session = self._sessions.get(request.target_workspace, request.session_id)
        manifest = self._allocator.allocate(
            working_set,
            process_id=record.process_id,
            labels=record.labels,
            user_message=request.message,
            agent_profile=agent_profile,
            selections=list(request.attachments),
            session_history=session.turns,
            compaction_checkpoint=session.compaction_checkpoint,
            access=record.access,
            target_workspace=request.target_workspace,
        )

        # 取消检查：只在进入 Actor 之前检查一次。prepare 与分配
        # 期间收到的 stop 请求都在此生效，已取得的租借由关闭流程释放。
        if record.outcome is ProcessOutcome.STOP_REQUESTED:
            raise _ProcessCancelled(ProcessPhase.PREPARE, record.stop_reason or "user_requested")
        return prepared, agent_profile, manifest

    def _accept_checkpoint(self, process: TaskProcess, result: CPUExecutionResult | None) -> None:
        """同步接受 CPU 的历史视图；领域拒绝仅记录告警，不影响本轮收尾。

        检查点只覆盖已封口的旧轮次，接受不依赖本轮有输出或交付方仍在线；
        只由 Actor 阶段调用一次。
        """
        if result is None or result.compaction_checkpoint is None:
            return
        try:
            self._sessions.accept_checkpoint(
                process.request.target_workspace,
                process.request.session_id,
                process_id=process.record.process_id,
                checkpoint=result.compaction_checkpoint,
            )
        except WorkspaceDomainError:
            logger.warning(
                "拒绝会话压缩检查点：process_id=%s session_id=%s",
                process.record.process_id,
                process.request.session_id,
                exc_info=True,
            )

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
        """进程关闭：终态兜底、结束线程、释放租借、补偿 prepare。

        只在进程 task 中执行：:meth:`run` 的收尾，或进程从未运行就被关闭时
        由注册入口的进程 task 直接调用。终态兜底只在进程仍无终态时执行，即
        硬取消与“从未运行就被关闭”两种情形。结束线程与租借释放都先于任何
        await 同步执行：硬取消打断后续 cleanup 时，也不留下有效的线程凭据或
        未释放的租借；线程正常由 Actor 阶段结束，这里只是兜底，重复结束是
        空操作。context 失效与进程注销由注册入口在本流程之后执行。工作集中的
        每项资源只取出一次，本方法即使重复执行也不会重复释放。
        """
        record = process.record
        working_set = process.working_set
        if not record.is_terminal:
            record.mark_cancelled()
            record.events.closed_before_terminal(record)

        self._threads.end_all(process)
        # 附件文本在 CPU 分配时已编译进清单，Actor 执行不再读取租借内容。
        self._allocator.release(working_set)
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


__all__ = ["TaskProcessRunner"]
