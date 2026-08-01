"""
ChatApplicationService — 顶层主动交互应用服务 (Phase D / v0.4.0)

v0.4.0 Phase 1 变更：
    - _generation_events dict 替换为 RuntimeControlRegistry
    - cancel_generation() 返回结构化 CancelResult
    - 取消后默认跳过主动生成提交（通过 loop_result.status 传递）
    - done 事件携带 status/reason/stopped 稳定字段
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    GatewayCancelledError,
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
    ChatGenerationRunStatus,
)
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

logger = logging.getLogger(__name__)


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
        """顶层非流式入口，统一执行 Gateway 后再进入 Agent 主链路。"""
        trace_id = generate_trace_id("chat")
        tokens = set_trace_context(trace_id, "ChatApp.Chat", "foreground")
        run = ChatGenerationRun(generation_id=generation_id or str(uuid.uuid4()))
        self._registry.register(run)
        self._emit_chat_event(
            RuntimeEventType.CHAT_RUN_CREATED,
            run,
            trace_id=trace_id,
            agent_id=agent_id,
        )
        prepared = None
        prepared_finalized = False
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        try:
            run.status = ChatGenerationRunStatus.PREPARING
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            gateway_result = await self._bus.request(
                GlobalRoutes.GATEWAY_PROCESS,
                message=user_message,
                identity=identity,
                ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
                cancel_event=run.cancel_event,
                request_timeout_ms=self._gateway_request_timeout_ms,
            )

            if gateway_result.kind == "command":
                run.status = ChatGenerationRunStatus.COMPLETED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_COMPLETED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                )
                return NonStreamingChatCommandOutcome(
                    command_execution_result=(gateway_result.command_execution_result)
                )

            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                user_id=user_id,
                gateway_decision=gateway_result.decision,
                agent_id=agent_id,
                session_id=session_id,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
            )

            if run.cancelled:
                run.status = ChatGenerationRunStatus.CANCELLED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                )
                return NonStreamingChatAgentOutcome(agent_run_result=self._cancelled_agent_result())

            run.status = ChatGenerationRunStatus.STREAMING
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            loop_result: AgentRunResult = await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT,
                agent_run_context=prepared.agent_run_context,
                generation_options=prepared.generation_options,
                cancel_event=run.cancel_event,
            )

            if run.cancelled or loop_result.status == AgentRunStatus.CANCELLED.value:
                run.status = ChatGenerationRunStatus.CANCELLED
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
                run.status = ChatGenerationRunStatus.FAILED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_FAILED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                    severity="error",
                )
                return NonStreamingChatAgentOutcome(agent_run_result=loop_result)

            run.status = ChatGenerationRunStatus.FINALIZING
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

            run.status = ChatGenerationRunStatus.COMPLETED
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            return NonStreamingChatAgentOutcome(agent_run_result=loop_result)
        except GatewayCancelledError:
            run.status = ChatGenerationRunStatus.CANCELLED
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_CANCELLED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
            )
            return NonStreamingChatAgentOutcome(agent_run_result=self._cancelled_agent_result())
        except Exception:
            run.status = ChatGenerationRunStatus.FAILED
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
            self._registry.close(run.generation_id, run.status)
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
        """
        顶层流式 chat 入口。

        编排骨架: generation_id -> prepare -> prelude events -> run_agent_stream
                  -> [finalize if not cancelled] -> done
        """
        trace_id = generate_trace_id("stream")
        tokens = None

        run = ChatGenerationRun(generation_id=generation_id or str(uuid.uuid4()))
        prepared = None
        stream = None
        # 只记录 chat 终态是否已经对外发布；finally 依赖它判断是否需要断流兜底。
        terminal_state: Literal["completed", "cancelled", "failed"] | None = None
        # finalize 成功后 Patchouli 已接管本轮交互，不再清理 prepared run。
        prepared_finalized = False
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

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

            run.status = ChatGenerationRunStatus.PREPARING
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            gateway_result = await self._bus.request(
                GlobalRoutes.GATEWAY_PROCESS,
                message=user_message,
                identity=identity,
                ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
                cancel_event=run.cancel_event,
                request_timeout_ms=self._gateway_request_timeout_ms,
            )

            if gateway_result.kind == "command":
                command_result = gateway_result.command_execution_result
                run.status = ChatGenerationRunStatus.COMPLETED
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

            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                user_id=user_id,
                gateway_decision=gateway_result.decision,
                agent_id=agent_id,
                session_id=session_id,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
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

            # 分支：用户在 prepare 期间请求取消。跳过 Alice 和 finalize，返回 cancelled done。
            if run.cancelled:
                run.status = ChatGenerationRunStatus.CANCELLED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prelude.topic_id,
                )
                terminal_state = "cancelled"
                yield self._cancelled_done(run)
                return

            run.status = ChatGenerationRunStatus.STREAMING
            self._emit_chat_status(
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prelude.topic_id,
            )
            loop_result = None
            stream = await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT_STREAM,
                agent_run_context=prepared.agent_run_context,
                generation_options=prepared.generation_options,
                cancel_event=run.cancel_event,
            )
            async for event in stream:
                if event["event"] == "done":
                    loop_result = AgentRunResult(**event["data"])
                else:
                    yield event

            # 分支：Alice stream 非异常结束但没有终态 done，按协议错误进入 failed。
            if loop_result is None:
                raise RuntimeError("Stream ended without done event")

            # 分支：用户取消或 Alice 响应取消。跳过 finalize，不触发主动记忆生成。
            if run.cancelled or loop_result.status == AgentRunStatus.CANCELLED.value:
                run.status = ChatGenerationRunStatus.CANCELLED
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
                run.status = ChatGenerationRunStatus.FAILED
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

            run.status = ChatGenerationRunStatus.FINALIZING
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
                    "status": run.status.value,
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

            run.status = ChatGenerationRunStatus.COMPLETED
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

        except GatewayCancelledError:
            run.status = ChatGenerationRunStatus.CANCELLED
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_CANCELLED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
            )
            terminal_state = "cancelled"
            yield self._cancelled_done(run)
            return
        except Exception as e:
            # 分支：prepare / Alice / finalize 任一阶段抛出非取消异常，统一发布 failed。
            logger.error(f"ChatApplicationService.chat_stream 异常: {e}", exc_info=True)
            run.status = ChatGenerationRunStatus.FAILED
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
            if terminal_state is None:
                if not run.cancelled:
                    run.request_cancel("stream_closed")
                run.status = ChatGenerationRunStatus.CANCELLED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id if prepared is not None else None,
                    message="Chat stream closed before terminal event.",
                    data={"close_reason": run.cancel_reason or "stream_closed"},
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
            self._registry.close(run.generation_id, run.status)
            if tokens is not None:
                reset_trace_context(tokens)

    # ========== Generation 控制 ==========

    def cancel_generation(
        self,
        generation_id: str,
        reason: str = "user_requested",
    ) -> CancelResult:
        """幂等取消：重复调用返回当前状态，不报错。"""
        result = self._registry.cancel(generation_id, reason=reason)
        run = self._registry.get(generation_id)
        self._events.emit(
            RuntimeEvent(
                event_type=RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
                generation_id=generation_id,
                status=result.status,
                reason=result.reason,
                data={"cancelled": result.cancelled},
            )
        )
        if run is not None:
            self._emit_chat_status(run)
        return result

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
                "reason": run.cancel_reason or "user_requested",
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
                agent_id=agent_id,
                topic_id=topic_id,
                status=run.status.value,
                reason=run.cancel_reason,
                severity=severity,  # type: ignore[arg-type]
                message=message,
                data=data or {},
            )
        )

    async def _list_final_pool_topics(self, prepared_run) -> list[dict[str, Any]]:
        try:
            topics = await self._bus.request(
                GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE,
                identity=prepared_run.identity,
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
