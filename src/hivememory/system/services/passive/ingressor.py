"""顶层被动消息编排器——负责事件路由与 accumulator 提交时序。

memory context 准备下沉到 ``MemoryContextProvider``；完成的 passive turn
直接构建 ``InteractionSubmission`` 并移交通用队列，不再维护第二层 outbox。
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
from hivememory.system.services.passive.memory_context import MemoryContextProvider
from hivememory.system.services.passive.models import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveIngressOutcome,
    SealReason,
)
from hivememory.system.services.passive.serial_gate import PassiveIngressSerialGate
from hivememory.system.services.passive.turn_buffer import MessageTurnBufferManager

logger = logging.getLogger(__name__)


class PassiveMessageIngressor:
    """顶层被动消息编排器。

    同一外部会话在串行门内完成事件累积与队列 admission。队列接收成功前，
    accumulator 不会清空；接收失败会向调用方施加背压，下一轮也不会覆盖它。
    """

    def __init__(
        self,
        bus: GlobalSystemBus,
        *,
        interaction_queue: InteractionSubmissionQueue,
        gateway_request_timeout_ms: int = 8000,
        config: PassiveIngressConfig | None = None,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        if interaction_queue is None:
            raise TypeError("interaction_queue is required")

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
    def dedup(self) -> ExternalEventDedupRegistry:
        return self._dedup

    def configure_idle_flush(self, timeout_seconds: float = 30.0) -> None:
        self._idle_timeout = timeout_seconds

    # ------------------------------------------------------------------
    # accumulator -> submission queue
    # ------------------------------------------------------------------

    async def _finalize_current_turn(
        self,
        key: PassiveConversationKey,
        *,
        seal_reason: SealReason,
    ) -> bool:
        """提交当前 accumulator；只有 queue admission 成功后才 reset。"""
        buffer = self._buffers.peek_buffer(key)
        if buffer is None or not buffer.has_pending_round:
            return False

        prepared = buffer.prepare_flush()
        if prepared is None:
            return False

        interaction_id = buffer.interaction_id
        if interaction_id is None:
            raise RuntimeError(
                f"pending passive turn is missing interaction_id: conversation={key.label}"
            )

        payload, target_topic = prepared
        await self._interaction_queue.submit(
            InteractionSubmission(
                interaction_id=interaction_id,
                payload=payload,
                requested_topic_id=target_topic or "NEW_TOPIC",
                ordering_key=key.ordering_key,
                origin="passive_memory",
                correlation={
                    "source": key.source,
                    "external_conversation_id": key.external_conversation_id,
                    "turn_id": buffer.turn_id or "",
                    "seal_reason": seal_reason,
                },
            )
        )

        # submit 返回即表示 work 已由通用队列接收；后续 apply/retry 由 queue 负责。
        buffer.commit_flush(interaction_id)
        return True

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
        buffer = self._buffers.peek_buffer(key)
        if (
            event.is_final
            and buffer is not None
            and buffer.pending_final_event_key == event.dedup_key
        ):
            # 上次调用已经追加了该 final，只是 queue admission 失败。
            # 这里仅重试 finalize，不能重复 retrieval 或再次追加事件内容。
            await self._finalize_current_turn(
                key,
                seal_reason="explicit_final",
            )
            return PassiveIngressOutcome(
                kind="user" if event.role == "user" else "buffered"
            )

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
        # 先把上一轮移交队列，再分析新 user。admission 失败时不覆盖旧 accumulator，
        # 同时撤销新事件的 dedup 占位，让 connector 可以原样重试本次 user。
        try:
            await self._finalize_current_turn(key, seal_reason="next_user")
        except BaseException:
            self._dedup.discard(event.dedup_key)
            raise

        # Gateway/retrieval 的可恢复失败在 provider 内收敛为降级结果。
        attempt = await self._memory_context.prepare(event, identity, key)

        buffer = self._buffers.get_buffer(key, identity)
        buffer.accept_user(
            content=event.content,
            gateway_decision=attempt.decision,
            turn_id=event.turn_id,
        )

        if event.is_final:
            buffer.mark_finalization_pending(event.dedup_key)
            await self._finalize_current_turn(
                key,
                seal_reason="explicit_final",
            )

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

        if event.is_final:
            if buffer.has_pending_round:
                buffer.mark_finalization_pending(event.dedup_key)
            await self._finalize_current_turn(
                key,
                seal_reason="explicit_final",
            )

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
        """把指定会话的当前 turn 移交 submission queue。"""
        del identity  # key 已包含完整身份维度，保留参数用于公共入口兼容。
        async with self._serial_gate.hold(key):
            accepted = await self._finalize_current_turn(
                key,
                seal_reason=seal_reason,
            )
            return int(accepted)

    async def scan_idle_conversations_once(self) -> int:
        """把达到 idle timeout 的 turn 移交 submission queue。"""
        accepted = 0
        for key in self._buffers.list_active_buffers():
            async with self._serial_gate.hold(key):
                if not self._buffers.is_idle_timeout(key, self._idle_timeout):
                    continue
                accepted += int(
                    await self._finalize_current_turn(
                        key,
                        seal_reason="idle_timeout",
                    )
                )
        return accepted

    async def shutdown_drain(self) -> dict[str, Any]:
        """把全部挂起 turn 移交队列；队列 work 的执行等待由 service 负责。"""
        accepted = 0
        keys = set(self._buffers.list_active_buffers())
        keys.update(self._serial_gate.active_keys())
        for key in keys:
            async with self._serial_gate.hold(key):
                accepted += int(
                    await self._finalize_current_turn(
                        key,
                        seal_reason="shutdown_drain",
                    )
                )

        return {
            "finalized_turns": accepted,
            "accepted_submissions": accepted,
        }


__all__ = ["PassiveMessageIngressor"]
