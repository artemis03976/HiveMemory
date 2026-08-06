"""Passive ingress 的进程内按会话串行门。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from hivememory.system.services.passive.models import PassiveConversationKey


@dataclass
class _SerialEntry:
    lock: asyncio.Lock
    users: int = 0


class PassiveIngressSerialGate:
    """在单 event loop 内串行化同一外部会话的状态变更。

    `users` 同时统计持有者和等待者。最后一个使用者离开后立即移除 entry，
    避免只访问一次的外部会话永久占用 keyed-lock 字典。
    """

    def __init__(self) -> None:
        self._entries: dict[PassiveConversationKey, _SerialEntry] = {}
        self._guard = threading.Lock()

    @asynccontextmanager
    async def hold(
        self,
        key: PassiveConversationKey,
    ) -> AsyncGenerator[None, None]:
        entry = self._retain(key)
        try:
            async with entry.lock:
                yield
        finally:
            self._release(key, entry)

    def active_keys(self) -> tuple[PassiveConversationKey, ...]:
        """返回当前持有者或等待者涉及的会话 key 快照。"""
        with self._guard:
            return tuple(self._entries)

    @property
    def active_key_count(self) -> int:
        with self._guard:
            return len(self._entries)

    def _retain(self, key: PassiveConversationKey) -> _SerialEntry:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _SerialEntry(lock=asyncio.Lock())
                self._entries[key] = entry
            entry.users += 1
            return entry

    def _release(
        self,
        key: PassiveConversationKey,
        entry: _SerialEntry,
    ) -> None:
        with self._guard:
            current = self._entries.get(key)
            if current is not entry:
                raise RuntimeError("Passive ingress serial entry identity mismatch")

            entry.users -= 1
            if entry.users < 0:
                raise RuntimeError("Passive ingress serial entry users underflow")
            if entry.users == 0:
                self._entries.pop(key, None)


__all__ = ["PassiveIngressSerialGate"]
