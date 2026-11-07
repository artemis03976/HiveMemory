"""任务进程执行器 — 所有进程共用的四阶段编排骨架。

骨架依次驱动 Gateway 分析 → PREPARE（派生主线程、预检索、已有话题的确认与
CPU 分配）→ Actor 执行 → 有输出时封口交互记录并提交，对子系统的调用经 root
API、操作入口或全局总线的公开路由完成，Actor 执行经组合根注入的 CPU 端口
（``workspace.contracts`` 的 :class:`CPUPort`）完成。骨架运行在进程自己的
task 中，只把类型化的阶段产出写入进程的输出通道（见 ``workspace.process.outputs``）：
CPU 逐项推送交互事件，订阅通道的流式交付逐条转交。进入 Gateway 之前，骨架
先请求 Patchouli 在后台预取推理模型。

执行器跨进程共享、不持有任何进程的状态：它只持有编排依赖，每次运行与
关闭都以 :class:`TaskProcess` 为参数，进程的全部状态（进程记录、任务参数、
工作集、线程表、输出通道）都在进程上。执行器经进程记录使用注册时认证的两个
工作 actor 与事件发布器；进程 task 的创建、进程的登记与注销由注册入口
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
from hivememory.core.errors import OperationDeniedError, ThreadEndedError, WorkspaceDomainError
from hivememory.core.models import IdentityScope, MemoryAtom
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayIngressMode,
    GatewaySessionTurn,
    RetrievalMode,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.core.protocol.sealing import seal_interaction
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
from hivememory.patchouli.contracts.interaction_submission import InteractionSubmitResult
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.root.interaction import RootInteractionService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUInputManifest,
    CPUPort,
)
from hivememory.workspace.contracts.errors import ActiveInteractionFinalizationError
from hivememory.workspace.contracts.operations import OperationEntry, RetrieveRequest
from hivememory.workspace.contracts.threads import ExecutionThread
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

_NEW_TOPIC = "NEW_TOPIC"


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

    阶段调用按“谁在执行”授权：Gateway、Profile、附件、存储健康、话题确认与
    交互提交是系统级操作，以进程记录的系统 root 工作 actor 执行，Gateway 由
    执行器在调用前 root 授权，其余经 root API 在方法入口授权；预检索是执行者的
    actor 操作，经操作入口以主线程提交，由 actor API 授权，检索按执行者的可见性
    进行，检索命中随检索记录。交互中的 Agent（执行者）作为参与者显式交给提交。
    scope 只交给紧随的阶段调用，不进入 CPU 输入清单。
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
        operations: OperationEntry,
        interactions: RootInteractionService,
        memories: RootMemoryService,
        gateway_request_timeout_ms: int = 8000,
        gateway_session_turn_limit: int = 3,
    ) -> None:
        self._sessions = sessions
        self._bus = global_bus
        self._cpu = cpu
        self._allocator = allocator
        self._authorizer = operation_authorizer
        self._threads = threads
        self._operations = operations
        self._interactions = interactions
        self._memories = memories
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
                logger.warning(
                    "任务进程领域错误: process_id=%s code=%s details=%s",
                    record.process_id,
                    exc.code,
                    exc.details,
                )
            else:
                logger.exception("任务进程异常: process_id=%s", record.process_id)
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
        # Gateway 面对的话题属于 Workspace，与 Agent 无关：由系统以 root
        # 授权执行，执行者是否持有 READ 不影响这次系统编排。
        gateway_scope = self._root_scope(process)
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

        # ---- PREPARE：不可中断，进入 Actor 前统一检查 stop ----
        thread, memories, manifest = await self._prepare(process, gateway_result.decision)
        await process.outputs.send(InputsAllocated(memories=memories))

        # ---- Actor 执行：可被 stop 中断；交互事件经输出通道逐条转交 ----
        execution_result = await self._run_actor(process, manifest, thread)
        # 交付方离开或系统关闭时丢弃本轮；用户停止晚于 CPU 终态时沿用 CPU
        # 的结局，不把完整回复改记为 cancelled。
        if record.discards_output:
            raise _ProcessCancelled(
                record.phase, record.stop_reason or CLIENT_DISCONNECTED_STOP_REASON
            )
        topic_id: str | None = None
        if execution_result.has_output:
            topic_id = await self._seal_and_submit(process, execution_result)

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

        在 Gateway 的 root 授权之后发出，准入已失效的进程不触发加载。路由只开始加载、立即返回；
        Gateway 判定不需要检索时模型同样会加载，由空闲卸载回收。预取失败只记录日志，
        不影响任务进程。
        """
        try:
            await self._bus.request(PatchouliRoutes.PREFETCH_MODELS)
        except Exception as exc:
            logger.warning("推理模型预取请求失败，任务进程照常继续: %s", exc)

    async def _prepare(
        self,
        process: TaskProcess,
        decision: GatewayDecision,
    ) -> tuple[ExecutionThread, tuple[MemoryAtom, ...], CPUInputManifest]:
        """PREPARE：解析 Profile、派生主线程、预检索、确认已有话题、读取存储健康与 CPU 分配。

        Profile 解析与附件租借是用户的选择，由 CPUAllocator 在副作用前以系统的
        root 工作 actor 授权。主线程在 Profile 解析之后派生，预检索与已有话题的
        确认都以它提交：停止被接受后主线程随之结束，两者都以 ``ThreadEndedError``
        被拒绝，在此按 PREPARE 阶段的取消收口。Gateway 指向新话题时这里不建立
        话题，话题在第一次写入或交互应用时才建立。阶段内其余时刻收到的停止
        在进入 Actor 之前统一检查。
        """
        record = process.record
        request = process.request
        working_set = process.working_set
        record.enter_phase(ProcessPhase.PREPARE)
        working_set.gateway_decision = decision
        # 加载哪份图纸取自认证得到的执行者，不取自请求输入。
        agent_profile = await self._allocator.resolve_agent_profile(
            agent_id=record.actor.actor.agent_id,
            system_actor=record.system_actor,
            target_workspace=request.target_workspace,
        )
        thread = self._threads.derive_main(process, agent_profile)
        try:
            memories = await self._pre_retrieve(process, decision, thread)
            if decision.target_topic_id != _NEW_TOPIC:
                # 已有话题：确认它存在并刷新驻留，派发话题从 Actor 开始即可用。
                await self._threads.dispatch_topic(thread.handle)
        except ThreadEndedError:
            raise _ProcessCancelled(
                ProcessPhase.PREPARE, record.stop_reason or "user_requested"
            ) from None
        storage_available = await self._memories.storage_available(
            target_workspace=request.target_workspace,
            actor=record.system_actor,
        )

        session = self._sessions.get(request.target_workspace, request.session_id)
        manifest = self._allocator.allocate(
            working_set,
            process_id=record.process_id,
            labels=record.labels,
            user_message=request.message,
            agent_profile=agent_profile,
            memories=memories,
            storage_available=storage_available,
            selections=list(request.attachments),
            session_history=session.turns,
            compaction_checkpoint=session.compaction_checkpoint,
            system_actor=record.system_actor,
            target_workspace=request.target_workspace,
        )

        # 取消检查：只在进入 Actor 之前检查一次。PREPARE 期间收到的 stop 请求
        # 都在此生效，已取得的租借由关闭流程释放。
        if record.outcome is ProcessOutcome.STOP_REQUESTED:
            raise _ProcessCancelled(ProcessPhase.PREPARE, record.stop_reason or "user_requested")
        return thread, memories, manifest

    async def _pre_retrieve(
        self,
        process: TaskProcess,
        decision: GatewayDecision,
        thread: ExecutionThread,
    ) -> tuple[MemoryAtom, ...]:
        """预检索：执行者的 actor 操作，按 Gateway 的检索计划经操作入口以主线程提交。

        请求关闭了记忆检索、检索计划为跳过或要求零条时不检索。检索与 Alice 的
        SEARCH 走同一条路：actor API 以执行者授权 ``memory.search``，读取视图
        逐条做归属校验与交付授权。执行者没有 ``memory.search`` 时授权以
        ``operation_not_allowed`` 拒绝，这里把它译为不检索，对话照常进行；只译
        这一种拒绝，准入失效等其他授权失败照常传播，进程按失败收口。
        """
        plan = decision.retrieval_plan
        if (
            not process.request.enable_memory_retrieval
            or plan.mode is RetrievalMode.SKIP
            or plan.top_k == 0
        ):
            return ()
        retrieval = RetrieveRequest(
            semantic_query=decision.rewritten_query,
            keywords=tuple(decision.search_keywords),
            top_k=plan.top_k,
        )
        try:
            return tuple(await self._operations.execute(retrieval, thread=thread.handle))
        except OperationDeniedError as error:
            if error.details.get("reason") != "operation_not_allowed":
                raise
            return ()

    async def _run_actor(
        self, process: TaskProcess, manifest: CPUInputManifest, thread: ExecutionThread
    ) -> CPUExecutionResult:
        """Actor 阶段：把主线程交给 CPU，在一个可中断子 task 中等待它，返回本轮采用的结果。

        子 task 直接等待 CPU 端口的 ``execute``，交互事件经 sink 写入输出
        通道；停止取消子 task 时，无论 CPU 在等模型还是阻塞在通道写入上，
        都立即生效。结果的取舍：CPU 以返回值表达停止后的结局，返回之后进程
        以它为准，不再按停止标志改写为取消；CPU 原样传播取消、没有结果时按
        无输出的取消传播。

        主线程在 PREPARE 派生。子线程随 CALL 返回由 CPU 结束；全部线程只在本
        阶段的 ``finally`` 中兜底结束一次（关闭流程只是再兜底）；停止被接受时
        线程已随之结束，迟到的提交在此之前就被拒绝。检查点也只在这里接受一次：
        结束线程之后、判断丢弃与有无输出之前。普通执行异常没有可接受的结果。
        进程 task 被硬取消而 CPU 把取消转换为返回值时，先结束线程、接受其中
        已完成的检查点，再重新抛出取消。
        """
        record = process.record
        record.enter_phase(ProcessPhase.ACTOR)
        record.events.status(record)
        returned = _ReturnedResult()

        async def publish(event: dict[str, Any]) -> None:
            await process.outputs.send(ActorEvent(event))

        async def drive() -> None:
            returned.result = await self._cpu.execute(
                manifest,
                thread=thread,
                model_selection=process.request.model_selection,
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
            # 先结束全部线程，再接受检查点；两者都先于封口、提交与
            # 终态交付，硬取消时先于重新抛出取消。
            self._threads.end_all(process)
            self._accept_checkpoint(process, returned.result)
        # drive 正常结束时已经写入 CPU 返回的结果。
        assert returned.result is not None
        return returned.result

    async def _seal_and_submit(
        self,
        process: TaskProcess,
        execution_result: CPUExecutionResult,
    ) -> str | None:
        """封口交互记录，进入 FINALIZE 后由进程持有的提交任务提交，返回实际所在的话题。

        封口、进入 FINALIZE 与创建提交任务之间没有 await：提交任务在写入
        ``Finalizing`` 之前创建，交付方一旦收到 finalizing，提交就已由进程
        持有。交付方离开打断不了提交；提交以 ``join_shielded`` 等待，只有
        硬取消会遇到它，而硬取消也要等提交结束，提交成功时照常记录 CPU 的
        结局，再传播取消。
        """
        record = process.record
        payload = seal_interaction(
            turn_events=execution_result.turn_events,
            model_used=execution_result.model_used,
            used_attachments=process.working_set.used_attachments,
            outcome=execution_result.status.value,
            final_text=execution_result.final_text,
        )
        if not record.try_enter_finalizing():
            raise _ProcessCancelled(record.phase, record.stop_reason or "user_requested")
        submission = asyncio.create_task(self._submit(process, payload))
        record.events.status(record)
        await process.outputs.send(Finalizing())
        try:
            return await join_shielded(submission)
        except asyncio.CancelledError:
            if not submission.cancelled():
                if submission.exception() is None:
                    self._mark_execution_terminal(process, execution_result)
                else:
                    logger.warning(
                        "硬取消期间交互提交失败: process_id=%s",
                        record.process_id,
                        exc_info=submission.exception(),
                    )
            raise

    async def _submit(
        self,
        process: TaskProcess,
        payload: InteractionPayload,
    ) -> str | None:
        """提交任务：经 root API 提交交互并等待应用，再写入会话。

        先等正在进行的话题确定结束：有派发话题时提交到它，否则提交 Gateway 的
        新话题决定，话题在交互应用时建立。交互提交是系统级操作，以系统的
        root 工作 actor 执行；本轮的执行 Agent 作为参与者显式传递，记为交互的
        发起者。会话原文与记忆提交相互独立：提交成败都写入会话，话题取交互
        实际所在的话题，没有应用成功时为空；没有应用成功时随后以
        ``ActiveInteractionFinalizationError`` 失败，进程按失败收口。
        """
        record = process.record
        request = process.request
        working_set = process.working_set
        await working_set.settle_topic_determination()

        decision = working_set.gateway_decision
        assert decision is not None
        topic: str | GatewayDecision = working_set.dispatch_topic_id or decision
        result: InteractionSubmitResult | None = None
        try:
            result = await self._interactions.submit_interaction(
                target_workspace=request.target_workspace,
                participant=record.actor.actor,
                payload=payload,
                interaction_id=record.process_id,
                origin="active_chat",
                topic=topic,
                actor=record.system_actor,
            )
        except InteractionAdmissionUnavailableError as error:
            raise ActiveInteractionFinalizationError(
                interaction_id=record.process_id,
                stage="interaction_admission",
                reason=error.details.get("reason", "admission_unavailable"),
            ) from error
        finally:
            applied_topic_id = (
                result.topic_id if result is not None and result.state == "succeeded" else None
            )
            self._sessions.append(
                request.target_workspace,
                request.session_id,
                process_id=record.process_id,
                agent_id=record.labels.agent_id,
                payload=payload,
                topic_id=applied_topic_id,
            )
        if result.state != "succeeded":
            # 不设时限的等待只会因队列停止而停在应用之前。
            raise ActiveInteractionFinalizationError(
                interaction_id=record.process_id,
                stage="interaction_apply",
                reason="queue_stopped" if result.state == "pending" else "apply_failed",
                error_class=result.error_class,
            )
        if working_set.dispatch_topic_id is None and result.topic_id is not None:
            record.events.bind_topic(result.topic_id)
        return result.topic_id

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

    def _root_scope(self, process: TaskProcess) -> IdentityScope:
        """以进程记录的系统 root 工作 actor 对任务目标 workspace 执行 root 授权。

        返回操作授权者组装的可信 scope，供紧随的系统阶段路由使用；授权失败
        沿异常路径按进程失败收口。
        """
        return self._authorizer.authorize_root(
            process.record.system_actor,
            process.request.target_workspace,
        )

    # ========== 关闭流程 ==========

    async def close(self, process: TaskProcess) -> None:
        """进程关闭：终态兜底、结束线程、释放租借、等待正在进行的话题确定。

        只在进程 task 中执行：:meth:`run` 的收尾，或进程从未运行就被关闭时
        由注册入口的进程 task 直接调用。终态兜底只在进程仍无终态时执行，即
        硬取消与“从未运行就被关闭”两种情形。结束线程与租借释放都先于任何
        await 同步执行：硬取消打断后续等待时，也不留下仍在运行的线程或
        未释放的租借；线程正常由 Actor 阶段结束，这里只是兜底，重复结束是
        空操作。进程注销由注册入口在本流程之后执行。工作集中的每项资源只
        取出一次，本方法即使重复执行也不会重复释放。没有任何补偿：话题只在
        写入或交互应用时建立，建立后没有接收交互的话题由空闲结算或 LRU 回收。
        """
        record = process.record
        working_set = process.working_set
        if not record.is_terminal:
            record.mark_cancelled()
            record.events.closed_before_terminal(record)

        self._threads.end_all(process)
        # 附件文本在 CPU 分配时已编译进清单，Actor 执行不再读取租借内容。
        self._allocator.release(working_set)
        # 没有提交的结局（停止、无输出、丢弃）下，写入启动的话题确定可能仍在进行。
        await working_set.settle_topic_determination()


__all__ = ["TaskProcessRunner"]
