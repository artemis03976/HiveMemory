"""
TaskProcessService — chat 任务进程的注册入口与四阶段编排骨架

位于 workspace：登记任务进程并编排 Gateway 分析 → Patchouli prepare →
CPU 分配 → Alice run → finalize 四个阶段，与进程表
（``workspace.process.table``）协作 stop 控制面；对子系统的一切调用都经
全局总线的公开路由完成。CPU 分配（Profile 解析、附件租借、记忆/附件
编译与输入清单组装）由进程在进入 Alice 之前完成；其中 Profile 解析
暂时提前到 prepare 之前，原因见 ``_resolve_agent_profile``。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RuntimeEventSink,
)
from hivememory.components.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.contracts.runtime_events import (
    RuntimeEvent,
    RuntimeEventType,
)
from hivememory.core.errors import (
    AssetOperationConflictError,
    WorkspaceDomainError,
    WorkspaceMismatchError,
)
from hivememory.core.models import (
    AgentProfile,
    AttachmentSelectionRequest,
    IdentityScope,
    MemoryAtom,
    ResolvedAgentProfile,
    require_identity_scope,
)
from hivememory.core.models.workspace_asset import (
    RepresentationLease,
)
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    GatewayIngressMode,
)
from hivememory.core.protocol.models import AgentRunResult, AgentRunStatus
from hivememory.engines.attachment_compiler import AttachmentCompiler
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUInputManifest
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessStatusSnapshot,
    ProcessTable,
)
from hivememory.workspace.process.working_set import ProcessWorkingSet

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
    prepared: Any,
    identity_scope: IdentityScope,
) -> None:
    """拒绝 prepare 返回与进程表不一致的请求级 scope。"""
    prepared_scope = getattr(prepared, "identity_scope", None)
    if not isinstance(prepared_scope, IdentityScope) or prepared_scope != identity_scope:
        raise WorkspaceMismatchError(
            "PreparedAgentRun 与进程记录的身份作用域不一致",
            details={
                "requested_workspace": identity_scope.workspace_identity.workspace_id,
                "prepared_workspace": (
                    prepared_scope.workspace_identity.workspace_id
                    if isinstance(prepared_scope, IdentityScope)
                    else None
                ),
            },
        )


@dataclass(frozen=True, kw_only=True)
class NonStreamingChatCommandOutcome:
    """非流式聊天的系统指令终态。"""

    kind: Literal["command"] = "command"
    command_execution_result: CommandExecutionResult


@dataclass(frozen=True, kw_only=True)
class NonStreamingChatAgentOutcome:
    """非流式聊天的 Agent 运行终态。"""

    kind: Literal["agent"] = "agent"
    agent_run_result: AgentRunResult


type NonStreamingChatResult = (NonStreamingChatCommandOutcome | NonStreamingChatAgentOutcome)


class TaskProcessService:
    """任务进程服务 — chat 任务类型的注册入口与四阶段编排骨架。

    对子系统的一切调用都经全局总线的公开路由完成，不直接持有任何子系统
    引用。身份入口约定（v0.6.2 收敛）：本服务只接受调用方在 server 边界
    冻结的 ``IdentityScope``，不再解析裸 ``user_id``。Chat 是 Agent
    action，必须由具体 Agent 执行；actor 为保留 ``system`` 值的 scope
    会在入口被拒绝。

    CPU 分配所需能力由组合根注入：``asset_reader`` 是进程级唯一
    WorkspaceAssetStore 的只读 reader 端口（附件租借在此 acquire，随进程
    关闭统一 release）；两个编译配置段驱动进程侧的记忆/附件编译。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        runtime_events: RuntimeEventSink | None = None,
        gateway_request_timeout_ms: int = 8000,
        *,
        asset_reader: WorkspaceAssetReaderPort | None = None,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        attachment_compiler_config: AttachmentCompilerConfig | None = None,
    ) -> None:
        self._bus = global_bus
        self._process_table = ProcessTable()
        self._events = runtime_events or NullRuntimeEventSink()
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._asset_reader = asset_reader
        # 记忆/附件编译使用与拆分前 Patchouli prepare 相同的引擎与配置段。
        self._memory_compiler_config = memory_compiler_config or MemoryCompilerConfig()
        self._memory_compiler = MemoryCompiler()
        self._attachment_compiler = AttachmentCompiler(
            attachment_compiler_config or AttachmentCompilerConfig(),
        )

    # ========== 非流式主链路 ==========

    async def chat_scoped(
        self,
        user_message: str,
        *,
        identity_scope: IdentityScope,
        process_id: str,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> NonStreamingChatResult:
        """非流式 Chat 公共入口：使用 server 边界冻结的完整 Workspace scope。

        ``process_id`` 由 server 入口在进入本服务前生成并冻结（Q-16）；
        ``attachments`` 只透传用户选择，ref/READY/版本校验发生在进程的
        CPU 分配边界（经注入的 reader port 读取 Store），本层不接触
        Patchouli 内部实现。
        """
        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        self._reject_system_actor(identity.agent_id)
        agent_id = identity.agent_id
        trace_id = generate_trace_id("chat")
        tokens = set_trace_context(trace_id, "TaskProcess.Chat", "foreground")
        run = ProcessRecord(
            identity_scope=identity_scope,
            process_id=process_id,
        )
        working_set = ProcessWorkingSet(asset_reader=self._asset_reader)
        prepared: PreparedAgentRun | None = None
        prepared_finalized = False
        try:
            self._process_table.register(run)
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_CREATED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
            )
            run.enter_phase(ProcessPhase.GATEWAY)
            self._emit_process_status(run, trace_id=trace_id, agent_id=agent_id)
            gateway_result = await _run_interruptible(
                run,
                ProcessPhase.GATEWAY,
                lambda: self._bus.request(
                    GlobalRoutes.GATEWAY_PROCESS,
                    message=user_message,
                    identity_scope=identity_scope,
                    ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
                    request_timeout_ms=self._gateway_request_timeout_ms,
                ),
            )

            if gateway_result.kind == "command":
                run.mark_completed()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_COMPLETED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                )
                return NonStreamingChatCommandOutcome(
                    command_execution_result=(gateway_result.command_execution_result)
                )

            run.enter_phase(ProcessPhase.PREPARE)
            agent_profile = await self._resolve_agent_profile(identity_scope)
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                identity_scope=identity_scope,
                interaction_id=process_id,
                gateway_decision=gateway_result.decision,
                enable_memory_retrieval=enable_memory_retrieval,
            )
            # 先写入工作集再校验：scope 不一致时 finally 仍需把它交回 cleanup，
            # 以补偿 prepare 可能已经预建的 Topic。
            working_set.prepared = prepared
            _require_prepared_scope(prepared, identity_scope)

            # CPU 分配的其余部分（附件租借与编译、清单组装）仍在 PREPARE 阶段内完成。
            self._allocate_cpu_inputs(
                run,
                working_set,
                identity_scope=identity_scope,
                agent_profile=agent_profile,
                selections=attachments or [],
            )

            # 取消检查（Q-15）：只在进入 Alice 之前检查一次。prepare 与分配
            # 期间收到的 stop 请求都在此生效，已取得的租借由 finally 释放。
            if run.outcome is ProcessOutcome.STOP_REQUESTED:
                raise _ProcessCancelled(
                    ProcessPhase.PREPARE,
                    run.stop_reason or "user_requested",
                )

            run.enter_phase(ProcessPhase.ALICE)
            self._emit_process_status(run, trace_id=trace_id, agent_id=agent_id)
            loop_result: AgentRunResult = await _run_interruptible(
                run,
                ProcessPhase.ALICE,
                lambda: self._bus.request(
                    GlobalRoutes.ALICE_RUN_AGENT,
                    input_manifest=working_set.input_manifest,
                    generation_options=generation_options,
                ),
            )

            if loop_result.status == AgentRunStatus.CANCELLED.value:
                run.mark_cancelled()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                )
                return NonStreamingChatAgentOutcome(
                    agent_run_result=self._cancelled_agent_result(loop_result)
                )
            if loop_result.status == AgentRunStatus.FAILED.value:
                run.mark_failed()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_FAILED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                    severity="error",
                )
                return NonStreamingChatAgentOutcome(agent_run_result=loop_result)

            if not run.try_enter_finalizing():
                raise _ProcessCancelled(
                    run.phase,
                    run.stop_reason or "user_requested",
                )
            self._emit_process_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=working_set.prepared,
                loop_result=loop_result,
                used_attachments=working_set.used_attachments,
            )
            prepared_finalized = True

            run.mark_completed()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            return NonStreamingChatAgentOutcome(agent_run_result=loop_result)
        except _ProcessCancelled as cancelled:
            run.mark_cancelled()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_CANCELLED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id if prepared is not None else None,
                data={"phase": cancelled.phase.value},
            )
            return NonStreamingChatAgentOutcome(agent_run_result=self._cancelled_agent_result())
        except Exception:
            run.mark_failed()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_FAILED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id if prepared is not None else None,
                severity="error",
            )
            logger.exception("TaskProcessService.chat 异常")
            raise
        finally:
            # 统一释放：完成、取消、失败与分配失败各条路径都在这里释放附件
            # 租借（幂等）。必须先于下面的 await 同步执行：owner task 在
            # cleanup 期间被取消时，CancelledError 不会被 except Exception
            # 捕获，释放不能依赖这些 await 完成。
            working_set.release()
            try:
                if working_set.prepared is not None and not prepared_finalized:
                    try:
                        await self._bus.request(
                            GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                            prepared_run=working_set.prepared,
                        )
                    except Exception:
                        logger.warning("清理 prepared run 失败", exc_info=True)
            finally:
                self._process_table.close(run)
                reset_trace_context(tokens)

    # ========== 流式主链路 ==========

    async def chat_stream_scoped(
        self,
        user_message: str,
        *,
        identity_scope: IdentityScope,
        process_id: str,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        attachments: list[AttachmentSelectionRequest] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式 Chat 公共入口：使用 server 边界冻结的完整 Workspace scope。

        ``process_id`` 由 server 入口在进入本服务前生成并冻结（Q-16）；
        ``attachments`` 只透传用户选择，ref/READY/版本校验发生在进程的
        CPU 分配边界（经注入的 reader port 读取 Store）。

        编排骨架: process_id 事件 -> gateway -> prepare -> CPU 分配
                  -> 前导事件 -> run_agent_stream -> [finalize if not cancelled] -> done
        """
        trace_id = generate_trace_id("stream")
        tokens = None

        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        self._reject_system_actor(identity.agent_id)
        agent_id = identity.agent_id
        run = ProcessRecord(
            identity_scope=identity_scope,
            process_id=process_id,
        )
        working_set = ProcessWorkingSet(asset_reader=self._asset_reader)
        prepared: PreparedAgentRun | None = None
        stream = None
        # 只记录 chat 终态是否已经对外发布；finally 依赖它判断是否需要断流兜底。
        terminal_state: Literal["completed", "cancelled", "failed"] | None = None
        # finalize 成功后 Patchouli 已接管本轮交互，不再清理 prepared run。
        prepared_finalized = False
        owner_task = asyncio.current_task()
        try:
            tokens = set_trace_context(trace_id, "TaskProcess.Stream", "foreground")
            self._process_table.register(run)
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_CREATED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
            )
            yield {"event": "process_id", "data": {"process_id": run.process_id}}

            run.enter_phase(ProcessPhase.GATEWAY)
            self._emit_process_status(run, trace_id=trace_id, agent_id=agent_id)
            gateway_result = await _run_interruptible(
                run,
                ProcessPhase.GATEWAY,
                lambda: self._bus.request(
                    GlobalRoutes.GATEWAY_PROCESS,
                    message=user_message,
                    identity_scope=identity_scope,
                    ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
                    request_timeout_ms=self._gateway_request_timeout_ms,
                ),
            )

            if gateway_result.kind == "command":
                command_result = gateway_result.command_execution_result
                run.mark_completed()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_COMPLETED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    data={"command_id": command_result.command_id},
                )
                terminal_state = "completed"
                yield {
                    "event": "command_result",
                    "data": command_result.model_dump(mode="json"),
                }
                yield self._command_done(run, command_result)
                return

            run.enter_phase(ProcessPhase.PREPARE)
            agent_profile = await self._resolve_agent_profile(identity_scope)
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                identity_scope=identity_scope,
                interaction_id=process_id,
                gateway_decision=gateway_result.decision,
                enable_memory_retrieval=enable_memory_retrieval,
            )
            # 先写入工作集再校验：scope 不一致时 finally 仍需把它交回 cleanup，
            # 以补偿 prepare 可能已经预建的 Topic。
            working_set.prepared = prepared
            _require_prepared_scope(prepared, identity_scope)

            # CPU 分配的其余部分（附件租借与编译、清单组装）仍在 PREPARE 阶段内完成。
            manifest = self._allocate_cpu_inputs(
                run,
                working_set,
                identity_scope=identity_scope,
                agent_profile=agent_profile,
                selections=attachments or [],
            )

            # 取消检查（Q-15）：只在进入 Alice 之前检查一次。prepare 与分配
            # 期间收到的 stop 请求都在此生效，已取得的租借由 finally 释放。
            if run.outcome is ProcessOutcome.STOP_REQUESTED:
                raise _ProcessCancelled(
                    ProcessPhase.PREPARE,
                    run.stop_reason or "user_requested",
                )

            # 前导事件只在分配成功后发出；分配失败或取消时不发出（与拆分前
            # "prepare 失败时不发出"一致）。
            yield {
                "event": "topic_info",
                "data": {
                    "topic_id": prepared.topic_id,
                    "is_new": prepared.is_new_topic,
                    "pool_topics": [
                        topic.model_dump(mode="json") for topic in prepared.pool_topics
                    ],
                },
            }
            yield {
                "event": "memory_refs",
                "data": {
                    "memories": [_memory_ref_from_atom(memory) for memory in manifest.memories],
                },
            }

            run.enter_phase(ProcessPhase.ALICE)
            self._emit_process_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            loop_result = None
            stream = await _run_interruptible(
                run,
                ProcessPhase.ALICE,
                lambda: self._bus.request(
                    GlobalRoutes.ALICE_RUN_AGENT_STREAM,
                    input_manifest=working_set.input_manifest,
                    generation_options=generation_options,
                ),
            )
            while True:
                try:
                    event = await _run_interruptible(
                        run,
                        ProcessPhase.ALICE,
                        lambda: anext(stream),
                    )
                except StopAsyncIteration:
                    break
                if event["event"] == "done":
                    loop_result = AgentRunResult(**event["data"])
                else:
                    yield event

            if loop_result is None:
                raise RuntimeError("Stream ended without done event")

            if loop_result.status == AgentRunStatus.CANCELLED.value:
                run.mark_cancelled()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                )
                terminal_state = "cancelled"
                yield self._cancelled_done(run, loop_result)
                return
            if loop_result.status == AgentRunStatus.FAILED.value:
                run.mark_failed()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_FAILED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                    severity="error",
                )
                terminal_state = "failed"
                yield self._failed_done(run, loop_result)
                return

            if not run.try_enter_finalizing():
                raise _ProcessCancelled(
                    run.phase,
                    run.stop_reason or "user_requested",
                )
            self._emit_process_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            yield {
                "event": "run_status",
                "data": {
                    "process_id": run.process_id,
                    "status": "finalizing",
                },
            }
            # 分支：正常完成 Alice 后进入 Patchouli finalize；成功后 prepared 不再需要 cleanup。
            # used_attachments 来自进程侧附件编译结果（被预算跳过的附件不在其中）。
            memory_tasks = await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=working_set.prepared,
                loop_result=loop_result,
                used_attachments=working_set.used_attachments,
            )
            prepared_finalized = True
            memory_task_ids = [memory_task.task_id for memory_task in (memory_tasks or [])]
            final_pool_topics = await self._list_final_pool_topics(prepared)

            run.mark_completed()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
                data={"memory_task_ids": memory_task_ids},
            )
            terminal_state = "completed"
            yield {
                "event": "done",
                "data": {
                    "process_id": run.process_id,
                    **loop_result.model_dump(),
                    "status": "completed",
                    "stopped": False,
                    "reason": None,
                    "memory_task_ids": memory_task_ids,
                    "pool_topics": final_pool_topics,
                },
            }

        except _ProcessCancelled as cancelled:
            run.mark_cancelled()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_CANCELLED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id if prepared is not None else None,
                data={"phase": cancelled.phase.value},
            )
            terminal_state = "cancelled"
            yield self._cancelled_done(run)
            return
        except WorkspaceDomainError as exc:
            # Workspace 领域错误（如附件 not_found/not_ready/failed/removed）
            # 携带安全文案：沿现有 Chat 错误边界原样翻译，不做二次包装。
            logger.warning("Chat stream 领域错误: %s", exc.code)
            run.mark_failed()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_FAILED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id if prepared is not None else None,
                severity="error",
                message=exc.code,
            )
            terminal_state = "failed"
            yield {
                "event": "error",
                "data": {"message": str(exc), "code": exc.code},
            }
        except Exception as e:
            logger.error(f"TaskProcessService.chat_stream 异常: {e}", exc_info=True)
            run.mark_failed()
            self._emit_process_event(
                RuntimeEventType.CHAT_RUN_FAILED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id if prepared is not None else None,
                severity="error",
                message="Chat stream failed.",
            )
            terminal_state = "failed"
            yield {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}
        finally:
            # 分支：客户端断开或生成器被提前关闭，且此前没有 completed/cancelled/failed 终态。
            owner_is_cancelling = owner_task is not None and owner_task.cancelling() > 0
            if terminal_state is None and not owner_is_cancelling:
                if run.outcome is ProcessOutcome.RUNNING:
                    run.request_stop("stream_closed")
                run.mark_cancelled()
                self._emit_process_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id if prepared is not None else None,
                    message="Chat stream closed before terminal event.",
                    data={"close_reason": run.stop_reason or "stream_closed"},
                )
                terminal_state = "cancelled"
            # 统一释放：完成、取消、失败、断流与分配失败各条路径都在这里释放
            # 附件租借（幂等）。必须先于下面的 await 同步执行：owner task 在
            # 关闭子流或 cleanup 期间被取消时，CancelledError 不会被
            # except Exception 捕获，释放不能依赖这些 await 完成。附件文本在
            # CPU 分配时已编译进清单，Alice 执行不再读取租借内容。
            working_set.release()
            try:
                # 统一清理：无论正常、取消、失败还是断流，都尝试关闭 Alice 子流。
                if stream is not None:
                    close = getattr(stream, "aclose", None)
                    if callable(close):
                        try:
                            await close()
                        except Exception:
                            logger.warning("关闭 Alice stream 失败", exc_info=True)
                # 统一清理：只要 prepare 成功但 finalize 未成功，就清理可能的新建空 topic。
                if working_set.prepared is not None and not prepared_finalized:
                    try:
                        await self._bus.request(
                            GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                            prepared_run=working_set.prepared,
                        )
                    except Exception:
                        logger.warning("清理 prepared run 失败", exc_info=True)
            finally:
                self._process_table.close(run)
                if tokens is not None:
                    reset_trace_context(tokens)

    # ========== CPU 分配（PREPARE 阶段内，进入 Alice 之前） ==========

    async def _resolve_agent_profile(self, identity_scope: IdentityScope) -> AgentProfile:
        """CPU 分配的 Profile 解析：经 Patchouli 公开路由解析本进程的执行 Profile。

        与拆分前 prepare 使用的本地路由是同一条解析规则；运行上下文只需要能力
        描述，源原子 policy 依据不进入 run（A2 §2.3）。暂不经能力层：能力层需要
        访问上下文（生产入口要到 A1 返工才取得），它依赖的 Profile 缓存也还没有
        失效机制（见任务进程 Idea 1.2）。

        中间态（2026-09-29）：Profile 属于 CPU 分配，但暂时在 Patchouli prepare
        之前解析。当前 prepare 会按 Gateway 的路由决定预先新建 Topic，话题池已满
        时还会先按 LRU 结算一个已有话题；若在 prepare 之后才发现 Profile 缺失，
        失败的请求已经留下这些不可逆的副作用。Topic 的新建与驱逐改到 interaction
        提交之后以后，Profile 解析可以回到 prepare 之后的 CPU 分配步骤。
        """
        resolved_profile: ResolvedAgentProfile = await self._bus.request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
            identity_scope.actor_identity.agent_id,
            identity_scope=identity_scope,
        )
        return resolved_profile.profile

    def _allocate_cpu_inputs(
        self,
        run: ProcessRecord,
        working_set: ProcessWorkingSet,
        *,
        identity_scope: IdentityScope,
        agent_profile: AgentProfile,
        selections: list[AttachmentSelectionRequest],
    ) -> CPUInputManifest:
        """CPU 分配的其余部分：取得附件租借并编译附件与记忆、组装输入清单。

        在 prepare 之后执行，读取工作集中的 prepare 结果；Profile 已由
        :meth:`_resolve_agent_profile` 提前解析。分配期间仍处于 ``PREPARE``
        阶段（不新增阶段取值），stop 请求不打断分配；分配完成后由调用方在
        进入 Alice 之前统一检查取消。任何失败沿异常路径上抛，已取得的租借由
        工作集在进程 ``finally`` 中释放。
        """
        prepared = working_set.prepared
        if prepared is None:
            raise RuntimeError("CPU 分配必须在 prepare 结果写入工作集之后执行")

        # 1. 附件：按用户选择顺序 acquire READY representation 并核对版本摘要。
        #    取得的 lease 由 _acquire_selected_attachment 直接登记进工作集。
        for selection in selections:
            self._acquire_selected_attachment(
                working_set,
                identity_scope,
                selection,
            )

        # 2. 编译：附件与记忆文本由进程生成，CPU 只消费成品。检索为空时
        #    memory_context 为空字符串（与拆分前 prepare 的行为一致）。
        attachment_compile_result = self._attachment_compiler.compile(
            leases=tuple(working_set.attachment_leases),
        )
        working_set.used_attachments = attachment_compile_result.used_attachments
        memories = list(prepared.retrieval_result.memories)
        memory_context = (
            self._memory_compiler.compile(
                memories,
                MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
                MemoryCompileOptions(
                    retrieval_strategy_config=(
                        self._memory_compiler_config.retrieval_context.strategy
                    ),
                ),
            ).text
            if memories
            else ""
        )

        # 3. 清单：组装与 CPU 无关的输入清单交给 Alice。
        manifest = CPUInputManifest(
            process_id=run.process_id,
            identity_scope=identity_scope,
            user_message=prepared.user_message,
            agent_profile=agent_profile,
            memories=memories,
            memory_context=memory_context,
            attachment_context=attachment_compile_result.attachment_context,
            storage_available=prepared.storage_available,
            topic_id=prepared.topic_id,
            topic_context=prepared.topic_context,
        )
        working_set.input_manifest = manifest
        return manifest

    def _acquire_selected_attachment(
        self,
        working_set: ProcessWorkingSet,
        identity_scope: IdentityScope,
        selection: AttachmentSelectionRequest,
    ) -> RepresentationLease:
        """acquire 单个选中附件并核对客户端提供的版本摘要。

        reader 的同一 Store 临界区已完成 Workspace/ref、asset READY 与
        representation READY 校验并建立 lease，无需先做 resolve_asset。
        取得的 lease 先登记进工作集；版本摘要不一致时经工作集释放该租借
        并拒绝整轮，不留游离租借。
        """
        if self._asset_reader is None:
            raise WorkspaceDomainError(
                "当前系统未装配附件读取能力，不能处理附件选择",
                details={"reason": "asset_reader_unavailable"},
            )
        lease = self._asset_reader.acquire_ready_representation(
            identity_scope,
            selection.asset_ref,
        )
        working_set.register_lease(lease)
        representation = lease.representation
        mismatch = (
            lease.asset_ref != selection.asset_ref
            or (
                selection.representation_id is not None
                and representation.representation_id != selection.representation_id
            )
            or (selection.revision is not None and representation.revision != selection.revision)
            or (
                selection.content_hash is not None
                and representation.content_hash != selection.content_hash
            )
        )
        if mismatch:
            working_set.discard_lease(lease)
            raise AssetOperationConflictError(
                "所选附件版本与当前可用表示不一致，请重新选择附件",
                details={
                    "reason": "selection_version_mismatch",
                    "asset_id": representation.asset_id,
                },
            )
        return lease

    # ========== 进程控制 ==========

    def cancel_process_scoped(
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
        self._events.emit(
            RuntimeEvent(
                event_type=RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
                process_id=process_id,
                workspace_id=frozen_scope.workspace_identity.workspace_id,
                status=result.status,
                reason=result.reason,
                data={"cancelled": result.cancelled},
            )
        )
        if run is not None:
            self._emit_process_status(run)
        return result

    def process_status_scoped(
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
        """Chat 必须由具体 Agent 执行；保留 ``system`` actor 在此显式失败。"""
        if agent_id == SYSTEM_AGENT_ID:
            raise WorkspaceDomainError(
                "Chat 不能使用保留 system actor：必须指定具体执行 Agent",
                details={"agent_id": agent_id},
            )

    @staticmethod
    def _cancelled_done(
        run: ProcessRecord,
        loop_result: AgentRunResult | None = None,
    ) -> dict[str, Any]:
        base = loop_result.model_dump() if loop_result is not None else {}
        return {
            "event": "done",
            "data": {
                **base,
                "process_id": run.process_id,
                "status": "cancelled",
                "stopped": True,
                "reason": run.stop_reason or "user_requested",
                "memory_task_ids": [],
            },
        }

    @staticmethod
    def _failed_done(
        run: ProcessRecord,
        loop_result: AgentRunResult,
    ) -> dict[str, Any]:
        return {
            "event": "done",
            "data": {
                **loop_result.model_dump(),
                "process_id": run.process_id,
                "status": "failed",
                "stopped": True,
                "reason": "agent_run_failed",
                "memory_task_ids": [],
            },
        }

    @staticmethod
    def _cancelled_agent_result(
        loop_result: AgentRunResult | None = None,
    ) -> AgentRunResult:
        if loop_result is None:
            return AgentRunResult(status=AgentRunStatus.CANCELLED)
        return loop_result.model_copy(update={"status": AgentRunStatus.CANCELLED})

    @staticmethod
    def _command_done(
        run: ProcessRecord,
        command_result: CommandExecutionResult,
    ) -> dict[str, Any]:
        return {
            "event": "done",
            "data": {
                "process_id": run.process_id,
                "final_text": command_result.message,
                "mtp_iterations": 0,
                "total_iterations": 0,
                "status": "completed",
                "stopped": False,
                "reason": None,
                "memory_task_ids": [],
                "pool_topics": [],
            },
        }

    def _emit_process_status(
        self,
        run: ProcessRecord,
        *,
        trace_id: str | None = None,
        agent_id: str | None = None,
        topic_id: str | None = None,
    ) -> None:
        self._emit_process_event(
            RuntimeEventType.CHAT_RUN_STATUS,
            run,
            trace_id=trace_id,
            agent_id=agent_id,
            topic_id=topic_id,
        )

    def _emit_process_event(
        self,
        event_type: RuntimeEventType,
        run: ProcessRecord,
        *,
        trace_id: str | None = None,
        agent_id: str | None = None,
        topic_id: str | None = None,
        severity: str = "info",
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._events.emit(
            RuntimeEvent(
                event_type=event_type,
                trace_id=trace_id,
                task_type="foreground",
                process_id=run.process_id,
                workspace_id=run.identity_scope.workspace_identity.workspace_id,
                agent_id=agent_id,
                topic_id=topic_id,
                status=self._event_status(run),
                reason=run.stop_reason,
                severity=severity,  # type: ignore[arg-type]
                message=message,
                data=data or {},
            )
        )

    @staticmethod
    def _event_status(run: ProcessRecord) -> str:
        if run.outcome is not ProcessOutcome.RUNNING:
            return run.outcome.value
        return {
            ProcessPhase.CREATED: "created",
            ProcessPhase.GATEWAY: "preparing",
            ProcessPhase.PREPARE: "preparing",
            ProcessPhase.ALICE: "streaming",
            ProcessPhase.FINALIZE: "finalizing",
            ProcessPhase.TERMINAL: "terminal",
        }[run.phase]

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


def _memory_ref_from_atom(memory: MemoryAtom) -> dict[str, Any]:
    """把 MemoryAtom 投影为前端引用列表使用的扁平结构。"""

    memory_type = memory.index.memory_type
    return {
        "id": str(memory.id),
        "title": memory.index.title,
        "summary": memory.index.summary,
        "memory_type": (memory_type.value if hasattr(memory_type, "value") else str(memory_type)),
        "tags": list(memory.index.tags),
        "alias": memory.index.alias,
        "content": memory.payload.content,
        "created_at": memory.meta.created_at,
        "updated_at": memory.meta.updated_at,
        "confidence_score": memory.meta.lifecycle.confidence_score,
        "vitality_score": memory.meta.lifecycle.vitality_score,
        "user_id": memory.workspace_identity.owner_user_id,
        "access_count": memory.meta.lifecycle.access_count,
    }


__all__ = [
    "NonStreamingChatAgentOutcome",
    "NonStreamingChatCommandOutcome",
    "NonStreamingChatResult",
    "TaskProcessService",
]
