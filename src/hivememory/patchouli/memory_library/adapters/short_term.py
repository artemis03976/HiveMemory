"""
InMemoryShortTermStorage — ShortTermStoragePort 的内存态实现

将短期话题池的底层存储职责收敛到此适配器。
LRU 驱逐、摘要更新与 blocks 裁剪等上层调度逻辑保留在 ShortTermMemoryStore。

实现阶段: Phase 1
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from hivememory.core.models import WorkspaceIdentity, WorkspaceTopicKey
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import ShortTermStoragePort


class InMemoryShortTermStorage(ShortTermStoragePort):
    """
    内存态短期存储适配器。

    线程安全，使用 RLock 保护复合键索引。
    """

    def __init__(self) -> None:
        self._buffers: Dict[WorkspaceTopicKey, SemanticBuffer] = {}
        self._workspace_index: Dict[tuple[str, str], Set[WorkspaceTopicKey]] = {}
        self._lock = threading.RLock()

    def get(self, key: WorkspaceTopicKey) -> Optional[SemanticBuffer]:
        with self._lock:
            return self._buffers.get(key)

    def put(self, key: WorkspaceTopicKey, buffer: SemanticBuffer) -> None:
        with self._lock:
            if key != buffer.topic_key:
                raise ValueError("Topic 复合键与 buffer Workspace 归属不一致")
            self._buffers[key] = buffer
            scope = (key.owner_user_id, key.workspace_id)
            self._workspace_index.setdefault(scope, set()).add(key)

    def pop(self, key: WorkspaceTopicKey) -> Optional[SemanticBuffer]:
        with self._lock:
            buf = self._buffers.pop(key, None)
            if buf is not None:
                scope = (key.owner_user_id, key.workspace_id)
                topics = self._workspace_index.get(scope, set())
                topics.discard(key)
                if not topics:
                    self._workspace_index.pop(scope, None)
            return buf

    def list_by_workspace(self, workspace: WorkspaceIdentity) -> List[SemanticBuffer]:
        with self._lock:
            scope = (workspace.owner_user_id, workspace.workspace_id)
            keys = self._workspace_index.get(scope, set())
            return [self._buffers[key] for key in keys if key in self._buffers]

    def list_all(self) -> List[SemanticBuffer]:
        with self._lock:
            return list(self._buffers.values())

    def count(self, workspace: WorkspaceIdentity) -> int:
        with self._lock:
            scope = (workspace.owner_user_id, workspace.workspace_id)
            return len(self._workspace_index.get(scope, ()))

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(
            name="short_term",
            healthy=True,
            detail="in-memory",
        )


__all__ = ["InMemoryShortTermStorage"]
