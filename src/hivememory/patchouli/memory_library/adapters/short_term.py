"""
InMemoryShortTermStorage — ShortTermStoragePort 的内存态实现

将短期话题池的底层存储职责收敛到此适配器。
LRU 驱逐、摘要更新与 blocks 裁剪等上层调度逻辑保留在 ShortTermMemoryStore。

版本: 0.1.0 (Phase 1)
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import ShortTermStoragePort


class InMemoryShortTermStorage(ShortTermStoragePort):
    """
    内存态短期存储适配器。

    线程安全，使用 RLock 保护 _buffers / _user_index。
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, SemanticBuffer] = {}
        self._user_index: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def get(self, topic_id: str) -> Optional[SemanticBuffer]:
        with self._lock:
            return self._buffers.get(topic_id)

    def put(self, topic_id: str, buffer: SemanticBuffer) -> None:
        with self._lock:
            self._buffers[topic_id] = buffer
            uid = buffer.user_id
            self._user_index.setdefault(uid, set()).add(topic_id)

    def pop(self, topic_id: str) -> Optional[SemanticBuffer]:
        with self._lock:
            buf = self._buffers.pop(topic_id, None)
            if buf is not None:
                topics = self._user_index.get(buf.user_id, set())
                topics.discard(topic_id)
                if not topics:
                    self._user_index.pop(buf.user_id, None)
            return buf

    def list_by_user(self, user_id: str) -> List[SemanticBuffer]:
        with self._lock:
            ids = self._user_index.get(user_id, set())
            return [self._buffers[t] for t in ids if t in self._buffers]

    def list_all(self) -> List[SemanticBuffer]:
        with self._lock:
            return list(self._buffers.values())

    def count(self) -> int:
        with self._lock:
            return len(self._buffers)

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(
            name="short_term",
            healthy=True,
            detail="in-memory",
        )

    # Backward-compatible aliases for older tests and diagnostics.

    def _get_sync(self, topic_id: str) -> Optional[SemanticBuffer]:
        return self.get(topic_id)

    def _put_sync(self, topic_id: str, buffer: SemanticBuffer) -> None:
        self.put(topic_id, buffer)

    def _pop_sync(self, topic_id: str) -> Optional[SemanticBuffer]:
        return self.pop(topic_id)

    def _list_by_user_sync(self, user_id: str) -> List[SemanticBuffer]:
        return self.list_by_user(user_id)

    def _list_all_sync(self) -> List[SemanticBuffer]:
        return self.list_all()

    def _count(self) -> int:
        return self.count()


__all__ = ["InMemoryShortTermStorage"]
