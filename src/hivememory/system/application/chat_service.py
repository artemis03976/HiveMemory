"""
ChatApplicationService — 顶层主动交互应用服务 (Phase D)

职责:
    - 统一入口参数归一化
    - 通过 GlobalSystemBus 调用 Patchouli prepare / finalize
    - 通过 GlobalSystemBus 调用 Alice runtime (run_agent / run_agent_stream)
    - 流式协议整形与前置事件输出
    - generation 生命周期注册与取消
    - 统一错误语义与日志
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from hivememory.core.protocol.models import AgentRunResult
from hivememory.infrastructure.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)


class ChatApplicationService:
    """顶层聊天应用服务 — 纯总线编排，不直接持有任何子系统引用。"""

    def __init__(self, global_bus: GlobalSystemBus) -> None:
        self._bus = global_bus
        self._generation_events: dict[str, asyncio.Event] = {}

    # ========== 非流式主链路 ==========

    async def chat(
        self,
        user_message: str,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
        enable_memory_retrieval: bool = True,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> AgentRunResult:
        """
        顶层非流式 chat 入口。

        编排骨架: prepare -> run_agent -> finalize
        """
        trace_id = generate_trace_id("chat")
        tokens = set_trace_context(trace_id, "ChatApp.Chat", "foreground")

        try:
            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
            )

            loop_result: AgentRunResult = await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT,
                agent_run_context=prepared.agent_run_context,
                generation_options=prepared.generation_options,
            )

            await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=prepared,
                loop_result=loop_result,
            )

            return loop_result
        except Exception:
            logger.exception("ChatApplicationService.chat 异常")
            raise
        finally:
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        顶层流式 chat 入口。

        编排骨架: generation_id -> prepare -> prelude events -> run_agent_stream -> finalize -> done
        """
        trace_id = generate_trace_id("stream")
        tokens = set_trace_context(trace_id, "ChatApp.Stream", "foreground")

        generation_id = str(uuid.uuid4())
        cancel_event = asyncio.Event()
        prepared = None

        try:
            self._generation_events[generation_id] = cancel_event
            yield {
                "event": "generation_id",
                "data": {"generation_id": generation_id},
            }

            prepared = await self._bus.request(
                GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
                user_message=user_message,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                enable_memory_retrieval=enable_memory_retrieval,
                generation_options=generation_options,
            )

            # Stream prelude events
            prelude = prepared.stream_prelude
            yield {
                "event": "topic_info",
                "data": {
                    "topic_id": prelude.topic_id,
                    "is_new": prelude.is_new_topic,
                    "pool": prelude.pool_snapshot,
                },
            }
            yield {
                "event": "memory_refs",
                "data": {"memories": prelude.memory_refs},
            }

            # Alice runtime streaming execution
            loop_result = None
            stream = await self._bus.request(
                GlobalRoutes.ALICE_RUN_AGENT_STREAM,
                agent_run_context=prepared.agent_run_context,
                generation_options=prepared.generation_options,
                cancel_event=cancel_event,
            )
            async for event in stream:
                if event["event"] == "done":
                    loop_result = AgentRunResult(**event["data"])
                else:
                    yield event

            if loop_result is None:
                raise RuntimeError("Stream ended without done event")

            await self._bus.request(
                GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
                prepared_run=prepared,
                loop_result=loop_result,
            )

            yield {
                "event": "done",
                "data": {
                    **loop_result.model_dump(),
                    "stopped": cancel_event.is_set(),
                },
            }
        except Exception as e:
            logger.error(f"ChatApplicationService.chat_stream 异常: {e}", exc_info=True)
            if prepared is not None:
                try:
                    await self._bus.request(
                        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
                        prepared_run=prepared,
                    )
                except Exception:
                    logger.warning(
                        "ChatApplicationService.chat_stream 清理 prepared run 失败",
                        exc_info=True,
                    )
            yield {
                "event": "error",
                "data": {"message": "系统错误，请检查后端服务器"},
            }
        finally:
            self._generation_events.pop(generation_id, None)
            reset_trace_context(tokens)

    # ========== Generation 控制 ==========

    def cancel_generation(self, generation_id: str) -> bool:
        """停止正在进行的流式生成。"""
        cancel_event = self._generation_events.get(generation_id)
        if cancel_event is None:
            return False
        cancel_event.set()
        return True
