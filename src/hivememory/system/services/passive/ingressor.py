"""顶层被动消息编排器 — 只负责事件路由与 seal 时序。

memory context 准备下沉到 `MemoryContextProvider`，
submission work 由 `InteractionSubmissionQueue` 承接；`SealedTurnSubmitter` 仅保留
queue admission 与旧回调兼容。本模块保留路由策略本身。
"""

from __future__ import annotations

import logging
from typing import Any

from hivememory.core.models import Identity
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
)
from hivememory.system.config.passive import PassiveIngressConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RuntimeEventSink
from hivememory.system.services.passive.dedup import ExternalEventDedupRegistry
from hivememory.system.services.passive.events import PassiveIngressEventEmitter
from hivememory.system.services.passive.memory_context import (
    MemoryContextProvider,
)
from hivememory.system.services.passive.models import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveIngressOutcome,
)
from hivememory.system.services.passive.outbox import (
    SealedTurn,
    SealedTurnOutbox,
    SealReason,
)
from hivememory.system.services.passive.serial_gate import (
    PassiveIngressSerialGate,
)
from hivememory.system.services.passive.submitter import (
    SealedTurnSubmitter,
    SubmitSealedTurn,
)
from hivememory.system.services.passive.turn_buffer import (
    FlushResult,
    MessageTurnBufferManager,
)

logger = logging.getLogger(__name__)


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

    观测（设计 §9）一律经 `RuntimeEventSink` 发布，不在 outcome 或公共响应中
    累积 trace/fallback 细节。
    """

    def __init__(
        self,
        bus: GlobalSystemBus,
        *,
        submit_sealed_turn: SubmitSealedTurn | None = None,
        interaction_queue: InteractionSubmissionQueue | None = None,
        gateway_request_timeout_ms: int = 8000,
        config: PassiveIngressConfig | None = None,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._config = config or PassiveIngressConfig()
        self._events = PassiveIngressEventEmitter(runtime_events)
        self._buffers = MessageTurnBufferManager(
            max_buffered_events_per_turn=(
                self._config.max_buffered_events_per_turn
            ),
        )
        self._memory_context = MemoryContextProvider(
            bus,
            gateway_request_timeout_ms=gateway_request_timeout_ms,
            events=self._events,
        )
        self._interaction_queue = interaction_queue
        self._submitter = SealedTurnSubmitter(
            submit_sealed_turn=(
                self._enqueue_interaction_submission
                if interaction_queue is not None
                else submit_sealed_turn
            ),
            max_items_per_conversation=(
                self._config.max_outbox_items_per_conversation
            ),
            events=self._events,
        )
        self._dedup = ExternalEventDedupRegistry(
            ttl_seconds=self._config.dedup_ttl_seconds,
            max_entries=self._config.max_dedup_entries,
        )
        self._serial_gate = PassiveIngressSerialGate()
        self._idle_timeout: float = 30.0

    # ------------------------------------------------------------------
    # 配置与内部状态访问
    # ------------------------------------------------------------------

    @property
    def buffers(self) -> MessageTurnBufferManager:
        return self._buffers

    @property
    def outbox(self) -> SealedTurnOutbox:
        return self._submitter.outbox

    @property
    def dedup(self) -> ExternalEventDedupRegistry:
        return self._dedup

    def configure_idle_flush(self, timeout_seconds: float = 30.0) -> None:
        self._idle_timeout = timeout_seconds

    # ------------------------------------------------------------------
    # seal / 提交
    # ------------------------------------------------------------------

    async def _enqueue_interaction_submission(self, sealed: SealedTurn) -> str | None:
        """把 passive sealed turn 投影为通用 submission work item。"""
        if self._interaction_queue is None:
            raise RuntimeError("interaction submission queue is not configured")
        await self._interaction_queue.submit(
            InteractionSubmission(
                interaction_id=sealed.interaction_id,
                payload=sealed.payload,
                requested_topic_id=sealed.target_topic or "NEW_TOPIC",
                ordering_key=sealed.conversation_key.ordering_key,
                origin="passive_memory",
                correlation={
                    "source": sealed.conversation_key.source,
                    "external_conversation_id": (
                        sealed.conversation_key.external_conversation_id
                    ),
                    "turn_id": sealed.turn_id or "",
                },
            )
        )
        # enqueue 只表示 queue 已接受；真实 topic_id 由 work outcome 保存。
        return None

    def _seal_into_outbox(
        self,
        key: PassiveConversationKey,
        flushed: FlushResult,
        *,
        seal_reason: SealReason,
        turn_id: str | None = None,
    ) -> None:
        payload, target_topic = flushed
        self._submitter.enqueue(
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
        return await self._submitter.drain(key)

    async def drain_all_outbox(self) -> int:
        return await self._submitter.drain_all()

    # ------------------------------------------------------------------
    # 事件路由
    # ------------------------------------------------------------------

    async def route_event(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
    ) -> PassiveIngressOutcome:
        key = event.conversation_key(identity)

        async with self._serial_gate.hold(key):
            return await self._route_event_serialized(event, identity, key)

    async def _route_event_serialized(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
        key: PassiveConversationKey,
    ) -> PassiveIngressOutcome:
        """在当前会话串行门内完成一次事件的全部状态变更。"""

        if not self._dedup.register(event.dedup_key):
            logger.info(
                "Passive ingress 忽略重复事件: source=%s, external_event_id=%s",
                event.source,
                event.external_event_id,
            )
            self._events.duplicate_ignored(
                key=key,
                external_event_id=event.external_event_id,
                role=event.role,
            )
            return PassiveIngressOutcome(kind="duplicate")

        self._events.event_accepted(
            key=key,
            external_event_id=event.external_event_id,
            role=event.role,
            turn_id=event.turn_id,
            sequence=event.sequence,
            is_final=event.is_final,
        )

        if event.role == "user":
            return await self._handle_user(event, identity, key)

        if event.role in ("assistant", "tool_call", "tool_result"):
            return await self._handle_buffered(event, identity, key)

        return PassiveIngressOutcome(kind="ignored")

    async def _handle_user(
        self,
        event: PassiveIngressEvent,
        identity: Identity,
        key: PassiveConversationKey,
    ) -> PassiveIngressOutcome:
        # 1. 先 seal 并提交上一 turn，再分析新 user。
        #    新请求的 Gateway/retrieval 失败不得连带阻塞上一轮记忆提交。
        self._seal_current_turn(key, seal_reason="next_user")
        await self.drain_outbox(key)

        # 2-3. Gateway 决策 + Patchouli retrieval。
        #      可恢复失败在此收敛为降级结果，不阻止 user 进入 buffer。
        attempt = await self._memory_context.prepare(event, identity, key)

        # 4. 用本轮 decision 初始化当前 turn accumulator。
        #    降级时 decision 为 None：payload 仍保留原始交互，
        #    只是缺少 rewritten_query / worth_saving / target_topic。
        buffer = self._buffers.get_buffer(key, identity)
        residual = buffer.accept_user(
            content=event.content,
            gateway_decision=attempt.decision,
            turn_id=event.turn_id,
        )
        if residual is not None:
            # 不变量兜底：accept_user 内的隐式 seal 结果不得丢弃
            self._seal_into_outbox(key, residual, seal_reason="next_user")
            await self.drain_outbox(key)

        if event.is_final and self._seal_current_turn(
            key, seal_reason="explicit_final"
        ):
            await self.drain_outbox(key)

        return PassiveIngressOutcome(
            kind="user",
            gateway_decision=attempt.decision,
            retrieval_result=attempt.retrieval_result,
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

        if event.is_final and self._seal_current_turn(
            key, seal_reason="explicit_final"
        ):
            await self.drain_outbox(key)

        return PassiveIngressOutcome(kind="buffered")

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
        async with self._serial_gate.hold(key):
            self._seal_current_turn(key, seal_reason=seal_reason)
            return await self.drain_outbox(key)

    async def scan_idle_conversations_once(self) -> int:
        """idle timeout：seal 超时 turn 后 drain 全部 outbox。"""
        for key in self._buffers.list_active_buffers():
            async with self._serial_gate.hold(key):
                idle_item = self._buffers.flush_idle_buffer(
                    key,
                    self._idle_timeout,
                )
                if idle_item is None:
                    continue
                flushed, turn_id = idle_item
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
        keys = set(self._buffers.list_active_buffers())
        keys.update(self._serial_gate.active_keys())
        for key in keys:
            async with self._serial_gate.hold(key):
                if self._seal_current_turn(key, seal_reason="shutdown_drain"):
                    sealed += 1

        submitted = await self.drain_all_outbox()
        return {
            "sealed_turns": sealed,
            "submitted_turns": submitted,
            "outbox_pending": self._submitter.pending_count(),
        }


__all__ = ["PassiveMessageIngressor", "SubmitSealedTurn"]
