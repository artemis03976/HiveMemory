"""
ChatApplicationService — 顶层主动交互应用服务 (Phase D / v0.4.0)

v0.4.0 Phase 1 变更：
    - _generation_events dict 替换为 RuntimeControlRegistry
    - cancel_generation() 返回结构化 CancelResult
    - 取消后默认跳过 run_active_generation（通过 loop_result.cancelled 标志传递）
    - done 事件携带 status/reason/stopped 稳定字段
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from hivememory.core.protocol.models import AgentRunResult
from hivememory.infrastructure.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.control import (
    CancelResult,
    ChatGenerationRun,
    ChatGenerationRunStatus,
    RuntimeControlRegistry,
)
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

logger = logging.getLogger(__name__)


class ChatApplicationService:
    """顶层聊天应用服务 — 纯总线编排，不直接持有任何子系统引用。"""

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._bus = global_bus
        self._registry = RuntimeControlRegistry()
        self._events = runtime_events or NullRuntimeEventSink()

    # ========== 非流式主链路 ==========

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
        generation_id: Optional[str] = None,
    ) -> AgentRunResult:
        """顶层非流式 chat 入口。编排骨架: prepare -> run_agent -> finalize"""
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

        try:
            run.status = ChatGenerationRunStatus.PREPARING
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                user_id=user_id,
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
                return self._cancelled_agent_result()

            run.status = ChatGenerationRunStatus.STREAMING
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            loop_result: AgentRunResult = await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT,
                agent_run_context=prepared.agent_run_context,
                generation_options=prepared.generation_options,
                cancel_event=run.cancel_event,
            )

            if run.cancelled or loop_result.cancelled:
                run.status = ChatGenerationRunStatus.CANCELLED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prepared.topic_id,
                )
                return self._cancelled_agent_result(loop_result)

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

            run.status = ChatGenerationRunStatus.COMPLETED
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prepared.topic_id,
            )
            return loop_result
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
            self._registry.close(run.generation_id, run.status)
            reset_trace_context(tokens)

    # ========== 流式主链路 ==========

    async def chat_stream(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
        generation_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        顶层流式 chat 入口。

        编排骨架: generation_id -> prepare -> prelude events -> run_agent_stream
                  -> [finalize if not cancelled] -> done
        """
        trace_id = generate_trace_id("stream")
        tokens = set_trace_context(trace_id, "ChatApp.Stream", "foreground")

        run = ChatGenerationRun(generation_id=generation_id or str(uuid.uuid4()))
        self._registry.register(run)
        self._emit_chat_event(
            RuntimeEventType.CHAT_RUN_CREATED,
            run,
            trace_id=trace_id,
            agent_id=agent_id,
        )
        prepared = None

        try:
            yield {"event": "generation_id", "data": {"generation_id": run.generation_id}}

            run.status = ChatGenerationRunStatus.PREPARING
            self._emit_chat_status(run, trace_id=trace_id, agent_id=agent_id)
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                user_id=user_id,
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
                    "pool": prelude.pool_snapshot,
                },
            }
            yield {"event": "memory_refs", "data": {"memories": prelude.memory_refs}}

            # 若 prepare 后已被取消，提前结束
            if run.cancelled:
                run.status = ChatGenerationRunStatus.CANCELLED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prelude.topic_id,
                )
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

            if loop_result is None:
                raise RuntimeError("Stream ended without done event")

            # 取消路径：跳过 finalize，不触发主动记忆生成
            if run.cancelled or loop_result.cancelled:
                run.status = ChatGenerationRunStatus.CANCELLED
                self._emit_chat_event(
                    RuntimeEventType.CHAT_RUN_CANCELLED,
                    run,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    topic_id=prelude.topic_id,
                )
                yield self._cancelled_done(run, loop_result)
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
            memory_tasks = await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=prepared,
                loop_result=loop_result,
            )
            memory_task_ids = [
                memory_task.task_id
                for memory_task in (memory_tasks or [])
            ]

            run.status = ChatGenerationRunStatus.COMPLETED
            self._emit_chat_event(
                RuntimeEventType.CHAT_RUN_COMPLETED,
                run,
                trace_id=trace_id,
                agent_id=agent_id,
                topic_id=prelude.topic_id,
                data={"memory_task_ids": memory_task_ids},
            )
            yield {
                "event": "done",
                "data": {
                    "generation_id": run.generation_id,
                    **loop_result.model_dump(),
                    "status": "completed",
                    "stopped": False,
                    "reason": None,
                    "memory_task_ids": memory_task_ids,
                },
            }

        except Exception as e:
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
            if prepared is not None:
                try:
                    await self._bus.request(
                        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                        prepared_run=prepared,
                    )
                except Exception:
                    logger.warning("清理 prepared run 失败", exc_info=True)
            yield {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}
        finally:
            self._registry.close(run.generation_id, run.status)
            reset_trace_context(tokens)

    # ========== Generation 控制 ==========

    def cancel_generation(self, generation_id: str) -> CancelResult:
        """幂等取消：重复调用返回当前状态，不报错。"""
        result = self._registry.cancel(generation_id, reason="user_requested")
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
        loop_result: Optional[AgentRunResult] = None,
    ) -> Dict[str, Any]:
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
    def _cancelled_agent_result(
        loop_result: Optional[AgentRunResult] = None,
    ) -> AgentRunResult:
        if loop_result is None:
            return AgentRunResult(cancelled=True)
        return loop_result.model_copy(update={"cancelled": True})

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
