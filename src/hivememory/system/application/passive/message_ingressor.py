from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayIngressMode,
    RetrievalMode,
)
from hivememory.core.protocol.models import (
    InteractionPayload,
    RetrievalRequest,
    RetrievalResponse,
)
from hivememory.system.application.passive.message_turn_buffer import (
    FlushResult,
    MessageTurnBufferManager,
)
from hivememory.system.application.passive.models import (
    PassiveIngressEvent,
    PassiveIngressOutcome,
)
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)


class PassiveMessageIngressor:
    """顶层被动消息编排器，显式请求 Gateway 决策与 Patchouli 检索。"""

    def __init__(
        self,
        bus: GlobalSystemBus,
        *,
        gateway_request_timeout_ms: int = 8000,
    ) -> None:
        self._bus = bus
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._buffers = MessageTurnBufferManager()
        self._idle_timeout: float = 30.0
        self._on_flush_callback: Callable[[InteractionPayload, str | None], Coroutine[Any, Any, None]] | None = None

    @property
    def buffers(self) -> MessageTurnBufferManager:
        return self._buffers

    def configure_idle_flush(
        self,
        timeout_seconds: float = 30.0,
        on_flush_callback: Callable[[InteractionPayload, str | None], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._idle_timeout = timeout_seconds
        self._on_flush_callback = on_flush_callback

    async def ingest_user_async(
        self,
        content: str,
        identity: Identity,
    ) -> tuple[GatewayDecision, RetrievalResponse, FlushResult | None]:
        gateway_result = await self._bus.request(
            GlobalRoutes.GATEWAY_PROCESS,
            message=content,
            identity=identity,
            ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
            request_timeout_ms=self._gateway_request_timeout_ms,
        )
        if gateway_result.kind != "decision":
            raise RuntimeError("PASSIVE_MEMORY 不得返回 command outcome")

        decision = gateway_result.decision
        retrieval_result = await self._retrieve_for_decision(decision, identity)
        buffer = self._buffers.get_buffer(identity)
        flushed = buffer.accept_user(
            content=content,
            gateway_decision=decision,
        )
        return decision, retrieval_result, flushed

    async def _retrieve_for_decision(
        self,
        decision: GatewayDecision,
        identity: Identity,
    ) -> RetrievalResponse:
        if (
            decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return RetrievalResponse()

        return await self._bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            request=RetrievalRequest(
                semantic_query=decision.rewritten_query,
                keywords=list(decision.search_keywords),
                identity=identity,
                top_k=decision.retrieval_plan.top_k,
            ),
        )

    def ingest_assistant(self, content: str, identity: Identity) -> None:
        buffer = self._buffers.get_buffer(identity)
        buffer.accept_assistant(content)

    def ingest_tool_call(
        self,
        content: str,
        identity: Identity,
        *,
        action_id: str | None = None,
        tool_name: str | None = None,
        tool_kind: str | None = None,
        tool_args: dict[str, Any] | None = None,
        target: str | None = None,
    ) -> None:
        buffer = self._buffers.get_buffer(identity)
        buffer.accept_tool_call(
            content,
            action_id=action_id,
            tool_name=tool_name,
            tool_kind=tool_kind,
            tool_args=tool_args,
            target=target,
        )

    def ingest_tool_result(
        self,
        content: str,
        identity: Identity,
        *,
        action_id: str | None = None,
        status: str | None = None,
        render_as: str = "plain",
    ) -> None:
        buffer = self._buffers.get_buffer(identity)
        buffer.accept_tool_result(
            content,
            action_id=action_id,
            status=status,
            render_as=render_as,
        )

    async def route_event(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
    ) -> PassiveIngressOutcome:
        if event.role == "user":
            decision, retrieval_result, flushed = await self.ingest_user_async(
                content=event.content,
                identity=identity,
            )
            return PassiveIngressOutcome(
                kind="user",
                gateway_decision=decision,
                retrieval_result=retrieval_result,
                flushed=flushed,
            )

        if event.role == "assistant":
            self.ingest_assistant(content=event.content, identity=identity)
            return PassiveIngressOutcome(kind="buffered")

        if event.role == "tool_call":
            self.ingest_tool_call(
                event.content,
                identity,
                action_id=event.action_id,
                tool_name=event.tool_name,
                tool_kind=event.tool_kind,
                tool_args=event.tool_args,
                target=event.target,
            )
            return PassiveIngressOutcome(kind="buffered")

        if event.role == "tool_result":
            self.ingest_tool_result(
                event.content,
                identity,
                action_id=event.action_id,
                status=event.status,
                render_as=event.render_as,
            )
            return PassiveIngressOutcome(kind="buffered")

        return PassiveIngressOutcome(kind="ignored")

    def flush_session(self, identity: Identity) -> FlushResult | None:
        buffer = self._buffers.get_buffer(identity)
        return buffer.flush()

    def flush_all_pending_sessions(self) -> list[FlushResult]:
        return self._buffers.flush_idle_buffers(-1.0)

    async def scan_idle_sessions_once(self) -> int:
        results = self._buffers.flush_idle_buffers(self._idle_timeout)
        if not results:
            return 0

        for payload, target_topic in results:
            if self._on_flush_callback:
                await self._on_flush_callback(payload, target_topic)

        return len(results)
