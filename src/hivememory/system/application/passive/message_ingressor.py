from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayIngressMode,
    RetrievalMode,
)
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.system.application.passive.dedup import ExternalEventDedupRegistry
from hivememory.system.application.passive.message_turn_buffer import (
    FlushResult,
    MessageTurnBufferManager,
)
from hivememory.system.application.passive.models import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveIngressOutcome,
)
from hivememory.system.application.passive.outbox import (
    SealedTurn,
    SealedTurnOutbox,
    SealReason,
)
from hivememory.system.config.passive import PassiveIngressConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus

logger = logging.getLogger(__name__)

SubmitSealedTurn = Callable[[SealedTurn], Awaitable[None]]


class PassiveMessageIngressor:
    """顶层被动消息编排器。

    读写时序（v0.6.0 设计 §5）::

        user event
          -> 若上一 turn 尚未提交，先 seal 并提交上一 turn
          -> Gateway.process(PASSIVE_MEMORY)
          -> GatewayDecision
          -> Patchouli retrieval
          -> 用 decision 初始化当前 turn buffer

        assistant / tool_call / tool_result
          -> 追加到当前 turn buffer（不调用 Gateway，不执行 tool）
          -> is_final 时 seal 并提交

    sealed turn 一律先进入 pending outbox，只有 Patchouli submit 成功后才移除；
    失败时保留 item 供重试，且不阻塞下一 turn accumulator。
    """

    def __init__(
        self,
        bus: GlobalSystemBus,
        *,
        submit_sealed_turn: SubmitSealedTurn | None = None,
        gateway_request_timeout_ms: int = 8000,
        config: PassiveIngressConfig | None = None,
    ) -> None:
        self._bus = bus
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._config = config or PassiveIngressConfig()
        self._buffers = MessageTurnBufferManager(
            max_buffered_events_per_turn=(
                self._config.max_buffered_events_per_turn
            ),
        )
        self._outbox = SealedTurnOutbox(
            max_items_per_conversation=(
                self._config.max_outbox_items_per_conversation
            ),
        )
        self._dedup = ExternalEventDedupRegistry(
            ttl_seconds=self._config.dedup_ttl_seconds,
            max_entries=self._config.max_dedup_entries,
        )
        self._submit_sealed_turn = submit_sealed_turn
        self._idle_timeout: float = 30.0
        self._drain_locks: dict[PassiveConversationKey, asyncio.Lock] = {}
        self._drain_locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # 配置与内部状态访问
    # ------------------------------------------------------------------

    @property
    def buffers(self) -> MessageTurnBufferManager:
        return self._buffers

    @property
    def outbox(self) -> SealedTurnOutbox:
        return self._outbox

    @property
    def dedup(self) -> ExternalEventDedupRegistry:
        return self._dedup

    def configure_submission(self, submit_sealed_turn: SubmitSealedTurn) -> None:
        self._submit_sealed_turn = submit_sealed_turn

    def configure_idle_flush(self, timeout_seconds: float = 30.0) -> None:
        self._idle_timeout = timeout_seconds

    def _drain_lock_for(self, key: PassiveConversationKey) -> asyncio.Lock:
        with self._drain_locks_guard:
            lock = self._drain_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._drain_locks[key] = lock
            return lock

    # ------------------------------------------------------------------
    # seal / outbox / 提交
    # ------------------------------------------------------------------

    def _seal_into_outbox(
        self,
        key: PassiveConversationKey,
        flushed: FlushResult,
        *,
        seal_reason: SealReason,
        turn_id: str | None = None,
    ) -> None:
        payload, target_topic = flushed
        self._outbox.enqueue(
            SealedTurn(
                conversation_key=key,
                payload=payload,
                target_topic=target_topic,
                seal_reason=seal_reason,
                turn_id=turn_id,
            )
        )

    def _seal_current_turn(
        self,
        key: PassiveConversationKey,
        identity: Identity,
        *,
        seal_reason: SealReason,
    ) -> bool:
        """把当前 accumulator 封口进 outbox。返回是否有内容被 seal。"""
        buffer = self._buffers.peek_buffer(key)
        if buffer is None or not buffer.has_pending_round:
            return False

        turn_id = buffer.turn_id
        flushed = buffer.flush()
        if flushed is None:
            return False

        self._seal_into_outbox(
            key,
            flushed,
            seal_reason=seal_reason,
            turn_id=turn_id,
        )
        return True

    async def drain_outbox(self, key: PassiveConversationKey) -> int:
        """尝试提交该会话全部挂起的 sealed turn。

        提交失败时保留剩余 item（按原顺序放回队首）并停止该会话本轮 drain，
        以保证会话内提交顺序，且不阻塞下一 turn accumulator。

        Returns:
            成功提交并从 outbox 移除的 item 数量。
        """
        if self._submit_sealed_turn is None:
            logger.error(
                "Passive ingress 未配置 sealed turn 提交回调，outbox 无法 drain: %s",
                key.label,
            )
            return 0

        async with self._drain_lock_for(key):
            pending = self._outbox.take_all(key)
            if not pending:
                return 0

            submitted = 0
            for index, item in enumerate(pending):
                attempted = item.with_attempt()
                try:
                    await self._submit_sealed_turn(attempted)
                except Exception as exc:
                    logger.warning(
                        "Passive sealed turn 提交失败，保留 outbox 供重试: "
                        "conversation=%s, seal_reason=%s, attempts=%s, error=%s",
                        key.label,
                        attempted.seal_reason,
                        attempted.attempts,
                        exc,
                    )
                    remaining = [attempted, *pending[index + 1:]]
                    self._outbox.requeue_front(key, remaining)
                    return submitted
                submitted += 1

            return submitted

    async def drain_all_outbox(self) -> int:
        total = 0
        for key in self._outbox.list_keys():
            total += await self.drain_outbox(key)
        return total

    # ------------------------------------------------------------------
    # 事件路由
    # ------------------------------------------------------------------

    async def route_event(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
    ) -> PassiveIngressOutcome:
        key = event.conversation_key(identity)

        if not self._dedup.register(event.dedup_key):
            logger.info(
                "Passive ingress 忽略重复事件: source=%s, external_event_id=%s",
                event.source,
                event.external_event_id,
            )
            return PassiveIngressOutcome(
                kind="duplicate",
                outbox_pending=self._outbox.pending_count(key),
            )

        if event.role == "user":
            return await self._handle_user(event, identity, key)

        if event.role in ("assistant", "tool_call", "tool_result"):
            return await self._handle_buffered(event, identity, key)

        return PassiveIngressOutcome(
            kind="ignored",
            outbox_pending=self._outbox.pending_count(key),
        )

    async def _handle_user(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
        key: PassiveConversationKey,
    ) -> PassiveIngressOutcome:
        # 1. 先 seal 并提交上一 turn，再分析新 user。
        #    新请求的 Gateway/retrieval 失败不得连带阻塞上一轮记忆提交。
        self._seal_current_turn(key, identity, seal_reason="next_user")
        submitted = await self.drain_outbox(key)

        # 2. Gateway 决策
        decision = await self._request_gateway_decision(event.content, identity)

        # 3. Patchouli retrieval
        retrieval_result = await self._retrieve_for_decision(decision, identity)

        # 4. 用本轮 decision 初始化当前 turn accumulator
        buffer = self._buffers.get_buffer(key, identity)
        residual = buffer.accept_user(
            content=event.content,
            gateway_decision=decision,
            turn_id=event.turn_id,
        )
        if residual is not None:
            # 不变量兜底：accept_user 内的隐式 seal 结果不得丢弃
            self._seal_into_outbox(key, residual, seal_reason="next_user")
            submitted += await self.drain_outbox(key)

        if event.is_final:
            if self._seal_current_turn(key, identity, seal_reason="explicit_final"):
                submitted += await self.drain_outbox(key)

        return PassiveIngressOutcome(
            kind="user",
            gateway_decision=decision,
            retrieval_result=retrieval_result,
            submitted_turns=submitted,
            outbox_pending=self._outbox.pending_count(key),
        )

    async def _handle_buffered(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
        key: PassiveConversationKey,
    ) -> PassiveIngressOutcome:
        buffer = self._buffers.get_buffer(key, identity)

        if event.role == "assistant":
            buffer.accept_assistant(event.content)
        elif event.role == "tool_call":
            buffer.accept_tool_call(
                event.content,
                action_id=event.action_id,
                tool_name=event.tool_name,
                tool_kind=event.tool_kind,
                tool_args=event.tool_args,
                target=event.target,
            )
        else:
            buffer.accept_tool_result(
                event.content,
                action_id=event.action_id,
                status=event.status,
                render_as=event.render_as,
            )

        submitted = 0
        if event.is_final:
            if self._seal_current_turn(key, identity, seal_reason="explicit_final"):
                submitted = await self.drain_outbox(key)

        return PassiveIngressOutcome(
            kind="buffered",
            submitted_turns=submitted,
            outbox_pending=self._outbox.pending_count(key),
        )

    # ------------------------------------------------------------------
    # Gateway / Patchouli 请求
    # ------------------------------------------------------------------

    async def _request_gateway_decision(
        self,
        content: str,
        identity: Identity,
    ) -> GatewayDecision:
        gateway_result = await self._bus.request(
            GlobalRoutes.GATEWAY_PROCESS,
            message=content,
            identity=identity,
            ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
            request_timeout_ms=self._gateway_request_timeout_ms,
        )
        if gateway_result.kind != "decision":
            raise RuntimeError("PASSIVE_MEMORY 不得返回 command outcome")
        return gateway_result.decision

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

    # ------------------------------------------------------------------
    # 显式 flush / idle / shutdown
    # ------------------------------------------------------------------

    async def flush_conversation(
        self,
        key: PassiveConversationKey,
        identity: Identity,
        *,
        seal_reason: SealReason = "manual_flush",
    ) -> int:
        """显式 seal 当前 turn 并 drain 该会话 outbox。"""
        self._seal_current_turn(key, identity, seal_reason=seal_reason)
        return await self.drain_outbox(key)

    async def scan_idle_sessions_once(self) -> int:
        """idle timeout：seal 超时 turn 后 drain 全部 outbox。"""
        idle_items = self._buffers.flush_idle_buffers(self._idle_timeout)
        for key, flushed, turn_id in idle_items:
            self._seal_into_outbox(
                key,
                flushed,
                seal_reason="idle_timeout",
                turn_id=turn_id,
            )
        return await self.drain_all_outbox()

    async def shutdown_drain(self) -> dict[str, Any]:
        """shutdown：seal 全部挂起 turn 后尽力 drain。"""
        sealed = 0
        for key, buffer in self._buffers.list_active_buffers().items():
            turn_id = buffer.turn_id
            flushed = buffer.flush()
            if flushed is None:
                continue
            self._seal_into_outbox(
                key,
                flushed,
                seal_reason="shutdown_drain",
                turn_id=turn_id,
            )
            sealed += 1

        submitted = await self.drain_all_outbox()
        return {
            "sealed_turns": sealed,
            "submitted_turns": submitted,
            "outbox_pending": self._outbox.pending_count(),
        }


__all__ = ["PassiveMessageIngressor", "SubmitSealedTurn"]
