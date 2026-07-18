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
from hivememory.system.application.passive.models import PassiveSessionKey

logger = logging.getLogger(__name__)

FlushResult = tuple[InteractionPayload, str | None]


class MessageBufferState(str, Enum):
    IDLE = "idle"
    AWAITING_RESPONSE = "awaiting"
    SEALED = "sealed"


class MessageTurnBuffer:
    """单 session / 单轮的结构化事件缓冲器。"""

    def __init__(
        self,
        identity: Identity,
        session_key: PassiveSessionKey | None = None,
    ) -> None:
        self._identity = identity
        self._session_key = session_key or PassiveSessionKey.from_identity(identity)
        self._reset()

    def _reset(self) -> None:
        self._state = MessageBufferState.IDLE
        self._user_content: str | None = None
        self._assistant_parts: list[str] = []
        self._turn_events: list[TurnEvent] = []
        self._sequence_counter: int = 0
        self._gateway_decision: GatewayDecision | None = None
        self._target_topic: str | None = None
        self._last_activity: float = datetime.now().timestamp()

    def _next_sequence(self) -> int:
        seq = self._sequence_counter
        self._sequence_counter += 1
        return seq

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
    ) -> FlushResult | None:
        flushed: FlushResult | None = None

        if self.has_pending_round:
            flushed = (self._build_payload(), self._target_topic)
            self._reset()

        self._user_content = content
        self._gateway_decision = gateway_decision
        self._target_topic = (
            gateway_decision.target_topic_id if gateway_decision else None
        )
        self._turn_events.append(
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
            f"session={self._session_key.label}, "
            f"target_topic={self._target_topic}, "
            f"flushed_previous={flushed is not None}"
        )

        return flushed

    def accept_assistant(self, content: str) -> None:
        if self._state == MessageBufferState.IDLE:
            logger.warning(
                "MessageTurnBuffer 收到 assistant 消息但无配对的 user 消息，忽略: "
                f"session={self._session_key.label}, "
                f"content='{content[:50]}...'"
            )
            return

        self._assistant_parts.append(content)
        self._turn_events.append(
            TurnEvent(
                kind="assistant_message",
                sequence=self._next_sequence(),
                role="assistant",
                content=content,
            )
        )
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
                f"session={self._session_key.label}"
            )
            return

        self._turn_events.append(
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
                f"session={self._session_key.label}"
            )
            return

        self._turn_events.append(
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
        logger.debug(f"MessageTurnBuffer flush: session={self._session_key.label}")
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
    """MessageTurnBuffer 池管理器。"""

    def __init__(self) -> None:
        self._buffers: dict[PassiveSessionKey, MessageTurnBuffer] = {}
        self._lock = threading.RLock()
        logger.info("MessageTurnBufferManager 初始化完成")

    @staticmethod
    def key_for_identity(identity: Identity) -> PassiveSessionKey:
        return PassiveSessionKey.from_identity(identity)

    def get_buffer(self, identity: Identity) -> MessageTurnBuffer:
        key = self.key_for_identity(identity)
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = MessageTurnBuffer(identity=identity, session_key=key)
                logger.debug(f"创建新 MessageTurnBuffer: {key.label}")
            return self._buffers[key]

    def remove_buffer(self, identity: Identity) -> None:
        key = self.key_for_identity(identity)
        with self._lock:
            self._buffers.pop(key, None)

    def list_active_buffers(self) -> dict[PassiveSessionKey, MessageTurnBuffer]:
        with self._lock:
            return dict(self._buffers)

    def flush_idle_buffers(self, timeout_seconds: float) -> list[FlushResult]:
        now = datetime.now().timestamp()
        results: list[FlushResult] = []

        with self._lock:
            for key, buf in list(self._buffers.items()):
                if buf.has_pending_round:
                    idle_duration = now - buf.last_activity_time
                    if idle_duration > timeout_seconds:
                        flushed = buf.flush()
                        if flushed:
                            results.append(flushed)
                            logger.info(
                                "Message idle timeout flush: "
                                f"session={key}, idle={idle_duration:.1f}s"
                            )

        return results


__all__ = [
    "MessageBufferState",
    "MessageTurnBuffer",
    "MessageTurnBufferManager",
    "FlushResult",
]
