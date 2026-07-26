"""
Pending sealed-turn outbox

每个外部会话分离"当前可变 turn accumulator"（MessageTurnBuffer）和
"不可变 pending sealed-turn outbox"（本模块）。

契约（v0.6.0 设计 §5/§6）：
    - turn 一旦 seal 就先进入 outbox，只有 Patchouli submit 成功后才移除 item。
    - 上一 turn 提交失败时保留 outbox item，但不占用也不覆盖当前 accumulator；
      新 user 仍可开始下一 turn。
    - v0.6.0 的 outbox 是有界进程内结构，后续持久化时保持同一提交语义。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

from hivememory.core.protocol.models import InteractionPayload
from hivememory.system.application.passive.models import PassiveConversationKey

logger = logging.getLogger(__name__)

SealReason = Literal[
    "next_user",
    "explicit_final",
    "idle_timeout",
    "manual_flush",
    "shutdown_drain",
]


@dataclass(frozen=True)
class SealedTurn:
    """已封口的不可变 turn 提交项。"""

    conversation_key: PassiveConversationKey
    payload: InteractionPayload
    target_topic: str | None
    seal_reason: SealReason
    turn_id: str | None = None
    sealed_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0

    def with_attempt(self) -> SealedTurn:
        return replace(self, attempts=self.attempts + 1)


class SealedTurnOutbox:
    """按外部会话分桶的有界 pending sealed-turn 队列。"""

    def __init__(self, *, max_items_per_conversation: int = 32) -> None:
        self._max_items = max(1, max_items_per_conversation)
        self._queues: dict[PassiveConversationKey, deque[SealedTurn]] = {}
        self._lock = threading.RLock()

    def enqueue(self, item: SealedTurn) -> None:
        with self._lock:
            queue = self._queues.setdefault(item.conversation_key, deque())
            queue.append(item)
            while len(queue) > self._max_items:
                dropped = queue.popleft()
                logger.warning(
                    "Passive outbox 溢出丢弃最旧 sealed turn: "
                    "conversation=%s, seal_reason=%s, attempts=%s",
                    dropped.conversation_key.label,
                    dropped.seal_reason,
                    dropped.attempts,
                )

    def take_all(self, key: PassiveConversationKey) -> list[SealedTurn]:
        """取出该会话全部挂起项，交由调用方尝试提交。"""
        with self._lock:
            queue = self._queues.pop(key, None)
            return list(queue) if queue else []

    def requeue_front(
        self,
        key: PassiveConversationKey,
        items: list[SealedTurn],
    ) -> None:
        """把未提交成功的项按原顺序放回队首，保持会话内提交顺序。"""
        if not items:
            return
        with self._lock:
            queue = self._queues.setdefault(key, deque())
            for item in reversed(items):
                queue.appendleft(item)
            while len(queue) > self._max_items:
                dropped = queue.pop()
                logger.warning(
                    "Passive outbox requeue 溢出丢弃最新 sealed turn: "
                    "conversation=%s, seal_reason=%s",
                    dropped.conversation_key.label,
                    dropped.seal_reason,
                )

    def list_keys(self) -> list[PassiveConversationKey]:
        with self._lock:
            return [key for key, queue in self._queues.items() if queue]

    def pending_count(self, key: PassiveConversationKey | None = None) -> int:
        with self._lock:
            if key is not None:
                queue = self._queues.get(key)
                return len(queue) if queue else 0
            return sum(len(queue) for queue in self._queues.values())


__all__ = ["SealReason", "SealedTurn", "SealedTurnOutbox"]
