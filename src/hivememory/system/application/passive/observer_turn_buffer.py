"""
ObserverTurnBuffer — 被动观测模式的单轮结构化事件缓冲器
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from hivememory.core.models import Identity
from hivememory.core.models.interaction import TurnEvent
from hivememory.patchouli.protocol.models import InteractionPayload
from hivememory.system.application.passive.models import PassiveSessionKey

if TYPE_CHECKING:
    from hivememory.patchouli.protocol.models import EyeGazeResult

logger = logging.getLogger(__name__)

FlushResult = Tuple[InteractionPayload, Optional[str]]


class ObserverBufferState(str, Enum):
    IDLE = "idle"
    AWAITING_RESPONSE = "awaiting"
    SEALED = "sealed"


class ObserverTurnBuffer:
    """单 session / 单轮的结构化事件缓冲器。"""

    def __init__(
        self,
        identity: Identity,
        session_key: Optional[PassiveSessionKey] = None,
    ) -> None:
        self._identity = identity
        self._session_key = session_key or PassiveSessionKey.from_identity(identity)
        self._reset()

    def _reset(self) -> None:
        self._state = ObserverBufferState.IDLE
        self._user_content: Optional[str] = None
        self._assistant_parts: List[str] = []
        self._turn_events: List[TurnEvent] = []
        self._sequence_counter: int = 0
        self._gaze_result: Optional[EyeGazeResult] = None
        self._target_topic: Optional[str] = None
        self._last_activity: float = datetime.now().timestamp()

    def _next_sequence(self) -> int:
        seq = self._sequence_counter
        self._sequence_counter += 1
        return seq

    @property
    def state(self) -> ObserverBufferState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == ObserverBufferState.IDLE

    @property
    def is_awaiting(self) -> bool:
        return self._state == ObserverBufferState.AWAITING_RESPONSE

    @property
    def is_sealed(self) -> bool:
        return self._state == ObserverBufferState.SEALED

    @property
    def has_pending_round(self) -> bool:
        return self._state != ObserverBufferState.IDLE

    @property
    def last_activity_time(self) -> float:
        return self._last_activity

    def accept_user(
        self,
        content: str,
        gaze_result: Optional[EyeGazeResult] = None,
    ) -> Optional[FlushResult]:
        flushed: Optional[FlushResult] = None

        if self.has_pending_round:
            flushed = (self._build_payload(), self._target_topic)
            self._reset()

        self._user_content = content
        self._gaze_result = gaze_result
        self._target_topic = gaze_result.target_topic if gaze_result else None
        self._turn_events.append(
            TurnEvent(
                kind="user_message",
                sequence=self._next_sequence(),
                role="user",
                content=content,
            )
        )
        self._state = ObserverBufferState.AWAITING_RESPONSE
        self._last_activity = datetime.now().timestamp()

        logger.debug(
            "ObserverTurnBuffer 接收 user 消息: "
            f"session={self._session_key.label}, "
            f"target_topic={self._target_topic}, "
            f"flushed_previous={flushed is not None}"
        )

        return flushed

    def accept_assistant(self, content: str) -> None:
        if self._state == ObserverBufferState.IDLE:
            logger.warning(
                "ObserverTurnBuffer 收到 assistant 消息但无配对的 user 消息，忽略: "
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
        self._state = ObserverBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

    def accept_tool_call(
        self,
        content: str,
        *,
        action_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_kind: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        target: Optional[str] = None,
    ) -> None:
        if self._state == ObserverBufferState.IDLE:
            logger.warning(
                "ObserverTurnBuffer 收到 tool_call 但无配对的 user 消息，忽略: "
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
        self._state = ObserverBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

    def accept_tool_result(
        self,
        content: str,
        *,
        action_id: Optional[str] = None,
        status: Optional[str] = None,
        render_as: str = "plain",
    ) -> None:
        if self._state == ObserverBufferState.IDLE:
            logger.warning(
                "ObserverTurnBuffer 收到 tool_result 但无配对的 user 消息，忽略: "
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
        self._state = ObserverBufferState.SEALED
        self._last_activity = datetime.now().timestamp()

    def flush(self) -> Optional[FlushResult]:
        if self._state == ObserverBufferState.IDLE:
            return None

        result = (self._build_payload(), self._target_topic)
        self._reset()
        logger.debug(f"ObserverTurnBuffer flush: session={self._session_key.label}")
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
            write_focus=None,
            update_focus=None,
            identity=self._identity,
            rewritten_query=(
                self._gaze_result.rewritten_query if self._gaze_result else None
            ),
            worth_saving=(self._gaze_result.worth_saving if self._gaze_result else None),
        )


class ObserverTurnBufferManager:
    """ObserverTurnBuffer 池管理器。"""

    def __init__(self) -> None:
        self._buffers: Dict[PassiveSessionKey, ObserverTurnBuffer] = {}
        self._lock = threading.RLock()
        logger.info("ObserverTurnBufferManager 初始化完成")

    @staticmethod
    def key_for_identity(identity: Identity) -> PassiveSessionKey:
        return PassiveSessionKey.from_identity(identity)

    def get_buffer(self, identity: Identity) -> ObserverTurnBuffer:
        key = self.key_for_identity(identity)
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = ObserverTurnBuffer(identity=identity, session_key=key)
                logger.debug(f"创建新 ObserverTurnBuffer: {key.label}")
            return self._buffers[key]

    def remove_buffer(self, identity: Identity) -> None:
        key = self.key_for_identity(identity)
        with self._lock:
            self._buffers.pop(key, None)

    def list_active_buffers(self) -> Dict[PassiveSessionKey, ObserverTurnBuffer]:
        with self._lock:
            return dict(self._buffers)

    def flush_idle_buffers(self, timeout_seconds: float) -> List[FlushResult]:
        now = datetime.now().timestamp()
        results: List[FlushResult] = []

        with self._lock:
            for key, buf in list(self._buffers.items()):
                if buf.has_pending_round:
                    idle_duration = now - buf.last_activity_time
                    if idle_duration > timeout_seconds:
                        flushed = buf.flush()
                        if flushed:
                            results.append(flushed)
                            logger.info(
                                "Observer idle timeout flush: "
                                f"session={key}, idle={idle_duration:.1f}s"
                            )

        return results


__all__ = [
    "ObserverBufferState",
    "ObserverTurnBuffer",
    "ObserverTurnBufferManager",
    "FlushResult",
]
