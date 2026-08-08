"""
外部事件幂等 registry

v0.6.0 只承诺有界的进程内幂等：同一 `source + external_event_id` 在 TTL 窗口内
重复到达时不重复追加 buffer、不重复 retrieval、不重复提交 interaction。
不承诺跨进程 exactly-once；后续可替换为持久化 ingress store。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from time import monotonic


class ExternalEventDedupRegistry:
    """有界 TTL 幂等 registry。"""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 4096,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._lock = threading.RLock()

    def _prune_expired(self, now: float) -> None:
        expired: list[tuple[str, str]] = []
        for key, seen_at in self._entries.items():
            if now - seen_at <= self._ttl_seconds:
                # OrderedDict 按插入顺序，首个未过期项之后都不会过期
                break
            expired.append(key)
        for key in expired:
            self._entries.pop(key, None)

    def register(self, key: tuple[str, str]) -> bool:
        """登记事件键。

        Returns:
            True 表示首次出现应继续处理；False 表示重复事件应被忽略。
        """
        now = monotonic()
        with self._lock:
            self._prune_expired(now)

            if key in self._entries:
                return False

            self._entries[key] = now
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return True

    def seen(self, key: tuple[str, str]) -> bool:
        now = monotonic()
        with self._lock:
            self._prune_expired(now)
            return key in self._entries

    def discard(self, key: tuple[str, str]) -> None:
        """撤销尚未完成处理的事件登记，允许调用方稍后重试。"""
        with self._lock:
            self._entries.pop(key, None)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


__all__ = ["ExternalEventDedupRegistry"]
