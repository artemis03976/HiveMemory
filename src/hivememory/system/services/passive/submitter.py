"""Sealed turn 提交器 — 持有 outbox 并负责重试语义。

契约（v0.6.0 设计 §5/§6）：
    - sealed turn 一律先进入 pending outbox，只有 Patchouli submit 成功后才移除。
    - 提交失败时保留剩余 item（按原顺序放回队首）并停止该会话本轮 drain，
      以保证会话内提交顺序，且不阻塞下一 turn accumulator。
    - 每个外部会话一把 asyncio 锁，避免并发 drain 撕裂顺序。

提交结果只经 `PassiveIngressEventEmitter` 发布，不在返回值里累积 trace。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable

from hivememory.system.services.passive.events import PassiveIngressEventEmitter
from hivememory.system.services.passive.models import PassiveConversationKey
from hivememory.system.services.passive.outbox import (
    SealedTurn,
    SealedTurnOutbox,
)

logger = logging.getLogger(__name__)

# 返回值是 Patchouli 落定的真实 topic_id（None 表示提交方未回传）。
SubmitSealedTurn = Callable[[SealedTurn], Awaitable[str | None]]


class SealedTurnSubmitter:
    """把 sealed turn 从 outbox 提交到 Patchouli，失败保留供重试。"""

    def __init__(
        self,
        *,
        submit_sealed_turn: SubmitSealedTurn | None = None,
        max_items_per_conversation: int = 32,
        events: PassiveIngressEventEmitter,
    ) -> None:
        self._submit_sealed_turn = submit_sealed_turn
        self._events = events
        self._outbox = SealedTurnOutbox(
            max_items_per_conversation=max_items_per_conversation,
        )
        self._drain_locks: dict[PassiveConversationKey, asyncio.Lock] = {}
        self._drain_locks_guard = threading.Lock()

    @property
    def outbox(self) -> SealedTurnOutbox:
        return self._outbox

    def enqueue(self, item: SealedTurn) -> None:
        self._outbox.enqueue(item)

    def pending_count(self, key: PassiveConversationKey | None = None) -> int:
        return self._outbox.pending_count(key)

    def _drain_lock_for(self, key: PassiveConversationKey) -> asyncio.Lock:
        with self._drain_locks_guard:
            lock = self._drain_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._drain_locks[key] = lock
            return lock

    async def drain(self, key: PassiveConversationKey) -> int:
        """尝试提交该会话全部挂起的 sealed turn。

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
                    settled_topic_id = await self._submit_sealed_turn(attempted)
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
                    self._events.turn_submit_failed(
                        key=key,
                        turn_id=attempted.turn_id,
                        error_class=type(exc).__name__,
                        seal_reason=attempted.seal_reason,
                        attempts=attempted.attempts,
                        will_retry=True,
                        outbox_pending=self._outbox.pending_count(key),
                    )
                    return submitted

                submitted += 1
                self._events.turn_submitted(
                    key=key,
                    turn_id=attempted.turn_id,
                    topic_id=settled_topic_id or attempted.target_topic,
                    event_count=len(attempted.payload.turn_events),
                    seal_reason=attempted.seal_reason,
                    attempts=attempted.attempts,
                )

            return submitted

    async def drain_all(self) -> int:
        total = 0
        for key in self._outbox.list_keys():
            total += await self.drain(key)
        return total


__all__ = ["SealedTurnSubmitter", "SubmitSealedTurn"]
