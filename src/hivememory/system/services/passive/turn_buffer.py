"""
MessageTurnBuffer — 被动观测模式的单轮结构化事件缓冲器
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Any

from hivememory.core.models import IdentityScope
from hivememory.core.models.interaction import TurnEvent
from hivememory.core.protocol.gateway import GatewayDecision
from hivememory.core.protocol.models import InteractionPayload
from hivememory.system.services.passive.models import PassiveConversationKey

logger = logging.getLogger(__name__)

FlushResult = tuple[InteractionPayload, str | None]

DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN = 256


class MessageBufferState(str, Enum):
    IDLE = "idle"
    ACCUMULATING = "accumulating"


class MessageTurnBuffer:
    """单外部会话 / 单轮的结构化事件缓冲器。"""

    def __init__(
        self,
        conversation_key: PassiveConversationKey,
        *,
        max_buffered_events_per_turn: int = DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN,
    ) -> None:
        self._conversation_key = conversation_key
        self._max_events = max(1, max_buffered_events_per_turn)
        self._reset()

    def _reset(self) -> None:
        self._state = MessageBufferState.IDLE
        self._identity_scope: IdentityScope | None = None
        self._interaction_id: str | None = None
        self._user_content: str | None = None
        self._assistant_parts: list[str] = []
        self._turn_events: list[TurnEvent] = []
        self._sequence_counter: int = 0
        self._gateway_decision: GatewayDecision | None = None
        self._target_topic: str | None = None
        self._turn_id: str | None = None
        self._pending_final_event_key: tuple[str, str] | None = None
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
    def interaction_id(self) -> str | None:
        """当前 turn 的稳定提交标识；只有队列接收成功后才会被清除。"""
        return self._interaction_id

    @property
    def identity_scope(self) -> IdentityScope | None:
        """当前 turn 冻结的身份作用域；由顶层 ingress 一次性生成。"""
        return self._identity_scope

    @property
    def pending_final_event_key(self) -> tuple[str, str] | None:
        """已追加但尚未完成 queue admission 的显式 final 事件。"""
        return self._pending_final_event_key

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
    def is_accumulating(self) -> bool:
        return self._state == MessageBufferState.ACCUMULATING

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
        identity_scope: IdentityScope,
        interaction_id: str,
        turn_id: str | None = None,
    ) -> None:
        """开启新一轮；调用方必须先完成上一轮的队列 admission。"""
        if self.has_pending_round:
            raise RuntimeError(
                "MessageTurnBuffer 仍有未提交 turn，不能覆盖当前 accumulator: "
                f"conversation={self._conversation_key.label}, "
                f"interaction_id={self.interaction_id}"
            )

        # 顶层 ingress 已生成本轮 interaction_id；buffer 不得另造第二份身份事实。
        self._identity_scope = identity_scope
        self._interaction_id = interaction_id
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
        self._state = MessageBufferState.ACCUMULATING
        self._last_activity = datetime.now().timestamp()

        logger.debug(
            "MessageTurnBuffer 接收 user 消息: "
            f"conversation={self._conversation_key.label}, "
            f"target_topic={self._target_topic}, "
            f"interaction_id={self.interaction_id}"
        )

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
        self._last_activity = datetime.now().timestamp()

    def prepare_flush(self) -> FlushResult | None:
        """构建当前 turn 的提交快照，但不清空 accumulator。"""
        if self._state == MessageBufferState.IDLE:
            return None

        return self._build_payload(), self._target_topic

    def mark_finalization_pending(self, event_key: tuple[str, str]) -> None:
        """在显式 final 已写入 buffer 后记录其可恢复 admission 状态。"""
        if not self.has_pending_round:
            raise RuntimeError("cannot mark finalization on an idle turn buffer")
        if (
            self._pending_final_event_key is not None
            and self._pending_final_event_key != event_key
        ):
            raise RuntimeError("another explicit final event is already pending admission")
        self._pending_final_event_key = event_key

    def commit_flush(self, interaction_id: str) -> None:
        """确认同一 turn 已被队列接收，然后清空 accumulator。"""
        if self.interaction_id != interaction_id:
            raise RuntimeError(
                "MessageTurnBuffer commit 对应的 interaction 已发生变化: "
                f"conversation={self._conversation_key.label}, "
                f"expected={interaction_id}, actual={self.interaction_id}"
            )

        self._reset()
        logger.debug(
            "MessageTurnBuffer commit flush: "
            f"conversation={self._conversation_key.label}, "
            f"interaction_id={interaction_id}"
        )

    def _build_payload(self) -> InteractionPayload:
        assistant_final_text = (
            "\n".join(self._assistant_parts) if self._assistant_parts else ""
        )

        return InteractionPayload(
            user_message=self._user_content or "",
            assistant_final_text=assistant_final_text or None,
            turn_events=list(self._turn_events),
            mtp_traces=[],
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
    ) -> MessageTurnBuffer:
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = MessageTurnBuffer(
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

    def is_idle_timeout(
        self,
        key: PassiveConversationKey,
        timeout_seconds: float,
    ) -> bool:
        """重新检查一个会话是否仍有 turn 且已达到空闲阈值。"""
        now = datetime.now().timestamp()
        with self._lock:
            buf = self._buffers.get(key)
            if buf is None or not buf.has_pending_round:
                return False

            idle_duration = now - buf.last_activity_time
            if idle_duration <= timeout_seconds:
                return False

            logger.info(
                f"Message idle timeout reached: conversation={key.label}, idle={idle_duration:.1f}s"
            )
            return True


__all__ = [
    "DEFAULT_MAX_BUFFERED_EVENTS_PER_TURN",
    "MessageBufferState",
    "MessageTurnBuffer",
    "MessageTurnBufferManager",
    "FlushResult",
]
