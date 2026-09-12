"""按 key 串行化异步操作的公共运行时机制。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class _SerialEntry:
    lock: asyncio.Lock
    users: int = 0


class KeyedSerialGate[Key: Hashable]:
    """在同一实例、同一 event loop 内串行化相同 key 的操作。

    users 同时统计持有者和等待者，取消等待也会释放计数。注册与回收不含
    await，最后一人离开即删除 key。key 的构造和持锁范围由调用方决定，
    同一协调范围内复用实例；不同实例互不影响。

    不支持跨线程、跨 event loop 使用，也不支持同一任务对同 key 重入。
    不保存业务状态或幂等结果，不接管调用方的停止接纳与 shutdown 流程。
    """

    def __init__(self) -> None:
        self._entries: dict[Key, _SerialEntry] = {}

    @asynccontextmanager
    async def hold(self, key: Key) -> AsyncIterator[None]:
        entry = self._entries.get(key)
        if entry is None:
            entry = _SerialEntry(lock=asyncio.Lock())
            self._entries[key] = entry
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._entries[key]

    def active_keys(self) -> tuple[Key, ...]:
        """当前持有者或等待者涉及的 key 快照；不包含此后新进入的操作。"""
        return tuple(self._entries)


__all__ = ["KeyedSerialGate"]
