"""
MessageTurnBuffer — 被动观测模式的单轮结构化事件缓冲器
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Any

from hivememory.core.models import Identity
from hivememory.core.models.interaction import TurnEvent
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import InteractionPayload
from hivememory.system.services.passive.models import PassiveConversationKey

logger = logging.getLogger(__name__)

FlushResult = tuple[InteractionPayload, str | None]

DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN = 256


class MessageBufferState(str, Enum):
    IDLE = "idle"
    AWAITING_RESPONSE = "awaiting"
    SEALED = "sealed"


class MessageTurnBuffer:
    """单外部会话 / 单轮的结构化事件缓冲器。"""

    def __init__(
        self,
        identity: Identity,
        conversation_key: PassiveConversationKey,
        *,
        max_buffered_events_per_turn: int = DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN,
    ) -> None:
        self._identity = identity
        self._conversation_key = conversation_key
        self._max_events = max(1, max_buffered_events_per_turn)
        self._reset()

    def _reset(self) -> None:
        self._state = MessageBufferState.IDLE
        self._user_content: str | None = None
        self._assistant_parts: list[str] = []
        self._turn_events: list[TurnEvent] = []
        self._sequence_counter: int = 0
        self._gateway_decision: GatewayDecision | None = None
        self._target_topic: str | None = None
        self._turn_id: str | None = None
        self._dropped_events: int = 0
        self._last_activity: float = datetime.now().timestamp()

    def _next_sequence(self) -> int:
        seq = self._sequence_counter
        self._sequence_counter += 1
        return seq

    def _append_event(self, event: TurnEvent) -> bool:
        """在事件上限内追加事件；超限时丢弃并计数。"""
        if len(self._turn_events) >= self._max_events:
            self._dropped_events += 1
            logger.warning(
                "MessageTurnBuffer 达到单轮事件上限，丢弃事件: "
                "conversation=%s, kind=%s, limit=%s, dropped=%s",
                self._conversation_key.label,
                event.kind,
                self._max_events,
                self._dropped_events,
            )
            return False
        self._turn_events.append(event)
        return True

    @property
    def conversation_key(self) -> PassiveConversationKey:
        return self._conversation_key

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    @property
    def event_count(self) -> int:
        return len(self._turn_events)

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def state(self) -> MessageBufferState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == MessageBufferState.IDLE

    @property
    def is_awaiting(self) -> bool:
        return self._state == MessageBufferState.AWAITING_RESPONSE

    @property
    def is_sealed(self) -> bool:
        return self._state == MessageBufferState.SEALED

    @property
    def has_pending_round(self) -> bool:
        return self._state != MessageBufferState.IDLE

    @property
    def last_activity_time(self) -> float:
        return self._last_activity

    def accept_user(
        self,
        content: str,
        gateway_decision: GatewayDecision | None = None,
        *,
        turn_id: str | None = None,
    ) -> FlushResult | None:
        """开启新一轮。

        调用方（Ingressor）应已先显式 flush 上一轮并提交；这里保留隐式 flush
        仅作为不变量兜底，返回值仍需被调用方送入 outbox，不得丢弃。
        """
        flushed: FlushResult | None = None

        if self.has_pending_round:
            flushed = (self._build_payload(), self._target_topic)
            logger.warning(
                "MessageTurnBuffer 在 accept_user 时发现未提交的上一轮，已兜底 seal: "
                f"conversation={self._conversation_key.label}"
            )
            self._reset()

        self._user_content = content
        self._gateway_decision = gateway_decision
        self._target_topic = (
            gateway_decision.target_topic_id if gateway_decision else None
        )
        self._turn_id = turn_id
        self._append_event(
            TurnEvent(
                kind="user_message",
                sequence=self._next_sequence(),
                role="user",
                content=content,
            )
        )
        self._state = MessageBufferState.AWAITING_RESPONSE
        self._last_activity = datetime.now().timestamp()

        logger.debug(
            "MessageTurnBuffer 接收 user 消息: "
            f"conversation={self._conversation_key.label}, "
            f"target_topic={self._target_topic}, "
            f"flushed_previous={flushed is not None}"
        )

        return flushed

    def accept_assistant(self, content: str) -> None:
        if self._state == MessageBufferState.IDLE:
            logger.warning(
                "MessageTurnBuffer 收到 assistant 消息但无配对的 user 消息，忽略: "
                f"conversation={self._conversation_key.label}, "
                f"content='{content[:50]}...'"
            )
            return

        if self._append_event(
            TurnEvent(
                kind="assistant_message",
                sequence=self._next_sequence(),
                role="assistant",
                content=content,
            )
        ):
            self._assistant_parts.append(content)
        self._state = MessageBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

    def accept_tool_call(
        self,
        content: str,
        *,
        action_id: str | None = None,
        tool_name: str | None = None,
        tool_kind: str | None = None,
        tool_args: dict[str, Any] | None = None,
        target: str | None = None,
    ) -> None:
        if self._state == MessageBufferState.IDLE:
            logger.warning(
                "MessageTurnBuffer 收到 tool_call 但无配对的 user 消息，忽略: "
                f"conversation={self._conversation_key.label}"
            )
            return

        self._append_event(
            TurnEvent(
                kind="tool_call",
                sequence=self._next_sequence(),
                role="assistant",
                content=content,
                action_id=action_id,
                tool_name=tool_name,
                tool_kind=tool_kind,
                tool_args=tool_args,
                target=target,
            )
        )
        self._state = MessageBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

    def accept_tool_result(
        self,
        content: str,
        *,
        action_id: str | None = None,
        status: str | None = None,
        render_as: str = "plain",
    ) -> None:
        if self._state == MessageBufferState.IDLE:
            logger.warning(
                "MessageTurnBuffer 收到 tool_result 但无配对的 user 消息，忽略: "
                f"conversation={self._conversation_key.label}"
            )
            return

        self._append_event(
            TurnEvent(
                kind="tool_result",
                sequence=self._next_sequence(),
                role="system",
                content=content,
                action_id=action_id,
                status=status,
                render_as=render_as,
            )
        )
        self._state = MessageBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

    def flush(self) -> FlushResult | None:
        if self._state == MessageBufferState.IDLE:
            return None

        result = (self._build_payload(), self._target_topic)
        self._reset()
        logger.debug(
            f"MessageTurnBuffer flush: conversation={self._conversation_key.label}"
        )
        return result

    def _build_payload(self) -> InteractionPayload:
        assistant_final_text = (
            "\n".join(self._assistant_parts) if self._assistant_parts else ""
        )

        return InteractionPayload(
            user_message=self._user_content or "",
            assistant_final_text=assistant_final_text or None,
            turn_events=list(self._turn_events),
            mtp_traces=[],
            identity=self._identity,
            rewritten_query=(
                self._gateway_decision.rewritten_query
                if self._gateway_decision
                else None
            ),
            worth_saving=(
                self._gateway_decision.worth_saving
                if self._gateway_decision
                else None
            ),
        )


class MessageTurnBufferManager:
    """按外部会话 key 分桶的 MessageTurnBuffer 池管理器。"""

    def __init__(
        self,
        *,
        max_buffered_events_per_turn: int = DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN,
    ) -> None:
        self._buffers: dict[PassiveConversationKey, MessageTurnBuffer] = {}
        self._max_buffered_events_per_turn = max_buffered_events_per_turn
        self._lock = threading.RLock()
        logger.info("MessageTurnBufferManager 初始化完成")

    def get_buffer(
        self,
        key: PassiveConversationKey,
        identity: Identity,
    ) -> MessageTurnBuffer:
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = MessageTurnBuffer(
                    identity=identity,
                    conversation_key=key,
                    max_buffered_events_per_turn=(
                        self._max_buffered_events_per_turn
                    ),
                )
                logger.debug(f"创建新 MessageTurnBuffer: {key.label}")
            return self._buffers[key]

    def peek_buffer(
        self,
        key: PassiveConversationKey,
    ) -> MessageTurnBuffer | None:
        with self._lock:
            return self._buffers.get(key)

    def remove_buffer(self, key: PassiveConversationKey) -> None:
        with self._lock:
            self._buffers.pop(key, None)

    def list_active_buffers(
        self,
    ) -> dict[PassiveConversationKey, MessageTurnBuffer]:
        with self._lock:
            return dict(self._buffers)

    def flush_idle_buffers(
        self,
        timeout_seconds: float,
    ) -> list[tuple[PassiveConversationKey, FlushResult, str | None]]:
        """flush 空闲超时的 buffer。

        Returns:
            `(conversation_key, flush_result, turn_id)` 列表，
            由调用方封装为 SealedTurn 并进入 outbox。
        """
        results: list[tuple[PassiveConversationKey, FlushResult, str | None]] = []

        for key in self.list_active_buffers():
            flushed = self.flush_idle_buffer(key, timeout_seconds)
            if flushed is not None:
                result, turn_id = flushed
                results.append((key, result, turn_id))

        return results

    def flush_idle_buffer(
        self,
        key: PassiveConversationKey,
        timeout_seconds: float,
    ) -> tuple[FlushResult, str | None] | None:
        """重新检查并 flush 一个空闲超时的会话 buffer。"""
        now = datetime.now().timestamp()
        with self._lock:
            buf = self._buffers.get(key)
            if buf is None or not buf.has_pending_round:
                return None

            idle_duration = now - buf.last_activity_time
            if idle_duration <= timeout_seconds:
                return None

            turn_id = buf.turn_id
            flushed = buf.flush()
            if flushed is None:
                return None

            logger.info(
                f"Message idle timeout flush: conversation={key.label}, idle={idle_duration:.1f}s"
            )
            return flushed, turn_id


__all__ = [
    "DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN",
    "MessageBufferState",
    "MessageTurnBuffer",
    "MessageTurnBufferManager",
    "FlushResult",
]
