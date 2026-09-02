"""
ChatApplicationService — 顶层主动交互应用服务 (Phase D / v0.4.0)

v0.4.0 Phase 1 变更：
    - _generation_events dict 替换为 RuntimeControlRegistry
    - cancel_generation() 返回结构化 CancelResult
    - 取消后默认跳过主动生成提交（通过 loop_result.status 传递）
    - done 事件携带 status/reason/stopped 稳定字段
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    Identity,
    IdentityScope,
    require_identity_scope,
    resolve_default_identity_scope,
)
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    GatewayIngressMode,
)
from hivememory.core.protocol.models import AgentRunResult, AgentRunStatus
from hivememory.infrastructure.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.control import (
    CancelResult,
    ChatGenerationRun,
    ChatGenerationRunRegistry,
    ChatRunOutcome,
    ChatRunPhase,
    ChatRunStatusSnapshot,
)
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

logger = logging.getLogger(__name__)


class _ChatRunCancelled(Exception):  # noqa: N818 - 设计要求使用私有领域分支名
    """Chat application 内部的用户 stop 分支。"""

    def __init__(self, phase: ChatRunPhase, reason: str) -> None:
        super().__init__(f"{phase.value} cancelled: {reason}")
        self.phase = phase
        self.reason = reason


async def _run_interruptible(
    control: ChatGenerationRun,
    phase: ChatRunPhase,
    operation_factory: Callable[[], Awaitable[Any]],
) -> Any:
    """用 Chat application 自有 child task 包装一个可中断阶段。"""
    owner_task = asyncio.current_task()
    if owner_task is None:
        raise RuntimeError("_run_interruptible 必须运行在 asyncio task 中")
    entry_cancelling = owner_task.cancelling()

    if control.outcome is ChatRunOutcome.STOP_REQUESTED:
        raise _ChatRunCancelled(phase, control.stop_reason or "user_requested")

    async def invoke() -> Any:
        return await operation_factory()

    task = asyncio.create_task(invoke())
    control.bind_phase(phase, task)
    try:
        result = await task
        if control.outcome is ChatRunOutcome.STOP_REQUESTED:
            raise _ChatRunCancelled(
                phase,
                control.stop_reason or "user_requested",
            )
        return result
    except asyncio.CancelledError:
        if owner_task.cancelling() > entry_cancelling:
            raise
        if (
            control.outcome is ChatRunOutcome.STOP_REQUESTED
            and control.active_task is task
        ):
            raise _ChatRunCancelled(
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
    """拒绝 prepare 返回与 control registry 不一致的请求级 scope。"""
    prepared_scope = getattr(prepared, "identity_scope", None)
    if (
        not isinstance(prepared_scope, IdentityScope)
        or prepared_scope.scope_fingerprint != identity_scope.scope_fingerprint
    ):
        raise WorkspaceMismatchError(
            "PreparedAgentRun 与 ChatGenerationRun 的身份作用域不一致",
            details={"generation_scope": identity_scope.scope_fingerprint},
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


class ChatApplicationService:
    """顶层聊天应用服务 — 纯总线编排，不直接持有任何子系统引用。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        runtime_events: RuntimeEventSink | None = None,
        gateway_request_timeout_ms: int = 8000,
    ) -> None:
        self._bus = global_bus
        self._registry = ChatGenerationRunRegistry()
        self._events = runtime_events or NullRuntimeEventSink()
        self._gateway_request_timeout_ms = gateway_request_timeout_ms

    # ========== 非流式主链路 ==========

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: str | None = None,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        generation_id: str | None = None,
    ) -> NonStreamingChatResult:
        """公共默认 Workspace 非流式入口。"""
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        identity_scope = resolve_default_identity_scope(identity)
        interaction_id = generation_id or f"interaction_{uuid.uuid4().hex}"
        return await self.chat_scoped(
            user_message=user_message,
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        )

    async def chat_scoped(
        self,
        user_message: str,
        *,
        identity_scope: IdentityScope,
        interaction_id: str,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> NonStreamingChatResult:
        """显式 scope 的内部非流式入口。"""
        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        agent_id = identity.agent_id
        trace_id = generate_trace_id("chat")
        tokens = set_trace_context(trace_id, "ChatApp.Chat", "foreground")
        run = ChatGenerationRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
        )
        prepared = None
        prepared_finalized = False
        try:
            self._registry.register(run)
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_CREATED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
            )
            run.enter_phase(ChatRunPhase.GATEWAY)
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            gateway_result = await _run_interruptible(
                run,
                ChatRunPhase.GATEWAY,
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
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_COMPLETED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                )
                return NonStreamingChatCommandOutcome(
                    command_execution_result=(gateway_result.command_execution_result)
                )

            if run.outcome is ChatRunOutcome.STOP_REQUESTED:
                raise _ChatRunCancelled(
                    ChatRunPhase.PREPARE,
                    run.stop_reason or "user_requested",
                )
            run.enter_phase(ChatRunPhase.PREPARE)
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                identity_scope=identity_scope,
                interaction_id=interaction_id,
                gateway_decision=gateway_result.decision,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
            )
            _require_prepared_scope(prepared, identity_scope)

            if run.outcome is ChatRunOutcome.STOP_REQUESTED:
                raise _ChatRunCancelled(
                    ChatRunPhase.PREPARE,
                    run.stop_reason or "user_requested",
                )

            run.enter_phase(ChatRunPhase.ALICE)
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            loop_result: AgentRunResult = await _run_interruptible(
                run,
                ChatRunPhase.ALICE,
                lambda: self._bus.request(
                    GlobalRoutes.ALICE_RUN_AGENT,
                    agent_run_context=prepared.agent_run_context,
                    generation_options=prepared.generation_options,
                    generation_id=run.generation_id,
                ),
            )

            if loop_result.status == AgentRunStatus.CANCELLED.value:
                run.mark_cancelled()
                self._emit_chat_event(
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
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_FAILED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                    severity="error",
                )
                return NonStreamingChatAgentOutcome(agent_run_result=loop_result)

            if not run.try_enter_finalizing():
                raise _ChatRunCancelled(
                    run.phase,
                    run.stop_reason or "user_requested",
                )
            self._emit_chat_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=prepared,
                loop_result=loop_result,
            )
            prepared_finalized = True

            run.mark_completed()
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            return NonStreamingChatAgentOutcome(agent_run_result=loop_result)
        except _ChatRunCancelled as cancelled:
            run.mark_cancelled()
            self._emit_chat_event(
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
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_FAILED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id if prepared is not None else None,
                severity="error",
            )
            logger.exception("ChatApplicationService.chat 异常")
            raise
        finally:
            if prepared is not None and not prepared_finalized:
                try:
                    await self._bus.request(
                        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                        prepared_run=prepared,
                    )
                except Exception:
                    logger.warning("清理 prepared run 失败", exc_info=True)
            self._registry.close(run)
            reset_trace_context(tokens)

    # ========== 流式主链路 ==========

    async def chat_stream(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: str | None = None,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
        generation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """公共默认 Workspace 流式入口。"""
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        identity_scope = resolve_default_identity_scope(identity)
        interaction_id = generation_id or f"interaction_{uuid.uuid4().hex}"
        async for event in self.chat_stream_scoped(
            user_message=user_message,
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            enable_memory_retrieval=enable_memory_retrieval,
            generation_options=generation_options,
        ):
            yield event

    async def chat_stream_scoped(
        self,
        user_message: str,
        *,
        identity_scope: IdentityScope,
        interaction_id: str,
        enable_memory_retrieval: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        显式 scope 的内部流式 chat 入口。

        编排骨架: interaction_id -> prepare -> prelude events -> run_agent_stream
                  -> [finalize if not cancelled] -> done
        """
        trace_id = generate_trace_id("stream")
        tokens = None

        identity_scope = require_identity_scope(identity_scope)
        identity = identity_scope.actor_identity
        agent_id = identity.agent_id
        run = ChatGenerationRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
        )
        prepared = None
        stream = None
        # 只记录 chat 终态是否已经对外发布；finally 依赖它判断是否需要断流兜底。
        terminal_state: Literal["completed", "cancelled", "failed"] | None = None
        # finalize 成功后 Patchouli 已接管本轮交互，不再清理 prepared run。
        prepared_finalized = False
        owner_task = asyncio.current_task()
        try:
            tokens = set_trace_context(trace_id, "ChatApp.Stream", "foreground")
            self._registry.register(run)
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_CREATED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
            )
            yield {"event": "generation_id", "data": {"generation_id": run.generation_id}}

            run.enter_phase(ChatRunPhase.GATEWAY)
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            gateway_result = await _run_interruptible(
                run,
                ChatRunPhase.GATEWAY,
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
                self._emit_chat_event(
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

            if run.outcome is ChatRunOutcome.STOP_REQUESTED:
                raise _ChatRunCancelled(
                    ChatRunPhase.PREPARE,
                    run.stop_reason or "user_requested",
                )
            run.enter_phase(ChatRunPhase.PREPARE)
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                identity_scope=identity_scope,
                interaction_id=interaction_id,
                gateway_decision=gateway_result.decision,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
            )
            _require_prepared_scope(prepared, identity_scope)

            if run.outcome is ChatRunOutcome.STOP_REQUESTED:
                raise _ChatRunCancelled(
                    ChatRunPhase.PREPARE,
                    run.stop_reason or "user_requested",
                )

            prelude = prepared.stream_prelude
            yield {
                "event": "topic_info",
                "data": {
                    "topic_id": prelude.topic_id,
                    "is_new": prelude.is_new_topic,
                    "pool_topics": [topic.model_dump(mode="json") for topic in prelude.pool_topics],
                },
            }
            yield {"event": "memory_refs", "data": {"memories": prelude.memory_refs}}

            run.enter_phase(ChatRunPhase.ALICE)
            self._emit_chat_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prelude.topic_id,
            )
            loop_result = None
            stream = await _run_interruptible(
                run,
                ChatRunPhase.ALICE,
                lambda: self._bus.request(
                    GlobalRoutes.ALICE_RUN_AGENT_STREAM,
                    agent_run_context=prepared.agent_run_context,
                    generation_options=prepared.generation_options,
                    generation_id=run.generation_id,
                ),
            )
            while True:
                try:
                    event = await _run_interruptible(
                        run,
                        ChatRunPhase.ALICE,
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
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prelude.topic_id,
                )
                terminal_state = "cancelled"
                yield self._cancelled_done(run, loop_result)
                return
            if loop_result.status == AgentRunStatus.FAILED.value:
                run.mark_failed()
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_FAILED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prelude.topic_id,
                    severity="error",
                )
                terminal_state = "failed"
                yield self._failed_done(run, loop_result)
                return

            if not run.try_enter_finalizing():
                raise _ChatRunCancelled(
                    run.phase,
                    run.stop_reason or "user_requested",
                )
            self._emit_chat_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prelude.topic_id,
            )
            yield {
                "event": "run_status",
                "data": {
                    "generation_id": run.generation_id,
                    "status": "finalizing",
                },
            }
            # 分支：正常完成 Alice 后进入 Patchouli finalize；成功后 prepared 不再需要 cleanup。
            memory_tasks = await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=prepared,
                loop_result=loop_result,
            )
            prepared_finalized = True
            memory_task_ids = [memory_task.task_id for memory_task in (memory_tasks or [])]
            final_pool_topics = await self._list_final_pool_topics(prepared)

            run.mark_completed()
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prelude.topic_id,
                data={"memory_task_ids": memory_task_ids},
            )
            terminal_state = "completed"
            yield {
                "event": "done",
                "data": {
                    "generation_id": run.generation_id,
                    **loop_result.model_dump(),
                    "status": "completed",
                    "stopped": False,
                    "reason": None,
                    "memory_task_ids": memory_task_ids,
                    "pool_topics": final_pool_topics,
                },
            }

        except _ChatRunCancelled as cancelled:
            run.mark_cancelled()
            self._emit_chat_event(
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
        except Exception as e:
            logger.error(f"ChatApplicationService.chat_stream 异常: {e}", exc_info=True)
            run.mark_failed()
            self._emit_chat_event(
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
                if run.outcome is ChatRunOutcome.RUNNING:
                    run.request_stop("stream_closed")
                run.mark_cancelled()
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id if prepared is not None else None,
                    message="Chat stream closed before terminal event.",
                    data={"close_reason": run.stop_reason or "stream_closed"},
                )
                terminal_state = "cancelled"
            # 统一清理：无论正常、取消、失败还是断流，都尝试关闭 Alice 子流。
            if stream is not None:
                close = getattr(stream, "aclose", None)
                if callable(close):
                    try:
                        await close()
                    except Exception:
                        logger.warning("关闭 Alice stream 失败", exc_info=True)
            # 统一清理：只要 prepare 成功但 finalize 未成功，就清理可能的新建空 topic。
            if prepared is not None and not prepared_finalized:
                try:
                    await self._bus.request(
                        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                        prepared_run=prepared,
                    )
                except Exception:
                    logger.warning("清理 prepared run 失败", exc_info=True)
            self._registry.close(run)
            if tokens is not None:
                reset_trace_context(tokens)

    # ========== Generation 控制 ==========

    def cancel_generation(
        self,
        generation_id: str,
        *,
        user_id: str = "default",
        agent_id: str = "omni_doll",
        reason: str = "user_requested",
    ) -> CancelResult:
        """公共默认 Workspace 的幂等取消入口。"""
        identity_scope = resolve_default_identity_scope(
            Identity(user_id=user_id, agent_id=agent_id),
        )
        return self.cancel_generation_scoped(
            generation_id,
            identity_scope=identity_scope,
            reason=reason,
        )

    def cancel_generation_scoped(
        self,
        generation_id: str,
        *,
        identity_scope: IdentityScope,
        reason: str = "user_requested",
    ) -> CancelResult:
        """按 owner/workspace 校验的内部取消入口。"""
        identity_scope = require_identity_scope(identity_scope)
        result = self._registry.cancel(
            generation_id,
            identity_scope,
            reason=reason,
        )
        run = self._registry.get(generation_id, identity_scope)
        self._events.emit(
            RuntimeEvent(
                event_type=RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
                generation_id=generation_id,
                interaction_id=generation_id,
                workspace_id=identity_scope.workspace_identity.workspace_id,
                status=result.status,
                reason=result.reason,
                data={"cancelled": result.cancelled},
            )
        )
        if run is not None:
            self._emit_chat_status(run)
        return result

    def generation_status_scoped(
        self,
        generation_id: str,
        *,
        identity_scope: IdentityScope,
    ) -> ChatRunStatusSnapshot | None:
        """返回 scoped Chat 状态；错误 scope 与不存在统一为 ``None``。"""
        return self._registry.status(
            generation_id,
            require_identity_scope(identity_scope),
        )

    # ========== 内部辅助 ==========

    @staticmethod
    def _cancelled_done(
        run: ChatGenerationRun,
        loop_result: AgentRunResult | None = None,
    ) -> dict[str, Any]:
        base = loop_result.model_dump() if loop_result is not None else {}
        return {
            "event": "done",
            "data": {
                **base,
                "generation_id": run.generation_id,
                "status": "cancelled",
                "stopped": True,
                "reason": run.stop_reason or "user_requested",
                "memory_task_ids": [],
            },
        }

    @staticmethod
    def _failed_done(
        run: ChatGenerationRun,
        loop_result: AgentRunResult,
    ) -> dict[str, Any]:
        return {
            "event": "done",
            "data": {
                **loop_result.model_dump(),
                "generation_id": run.generation_id,
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
        run: ChatGenerationRun,
        command_result: CommandExecutionResult,
    ) -> dict[str, Any]:
        return {
            "event": "done",
            "data": {
                "generation_id": run.generation_id,
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

    def _emit_chat_status(
        self,
        run: ChatGenerationRun,
        *,
        trace_id: str | None = None,
        agent_id: str | None = None,
        topic_id: str | None = None,
    ) -> None:
        self._emit_chat_event(
            RuntimeEventType.CHAT_RUN_STATUS,
            run,
            trace_id=trace_id,
            agent_id=agent_id,
            topic_id=topic_id,
        )

    def _emit_chat_event(
        self,
        event_type: RuntimeEventType,
        run: ChatGenerationRun,
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
                generation_id=run.generation_id,
                interaction_id=run.interaction_id,
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
    def _event_status(run: ChatGenerationRun) -> str:
        if run.outcome is not ChatRunOutcome.RUNNING:
            return run.outcome.value
        return {
            ChatRunPhase.CREATED: "created",
            ChatRunPhase.GATEWAY: "preparing",
            ChatRunPhase.PREPARE: "preparing",
            ChatRunPhase.ALICE: "streaming",
            ChatRunPhase.FINALIZE: "finalizing",
            ChatRunPhase.TERMINAL: "terminal",
        }[run.phase]

    async def _list_final_pool_topics(self, prepared_run) -> list[dict[str, Any]]:
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


__all__ = [
    "ChatApplicationService",
    "NonStreamingChatAgentOutcome",
    "NonStreamingChatCommandOutcome",
    "NonStreamingChatResult",
]
