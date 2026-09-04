"""In-memory adapter for the short-term storage port.

The adapter is the only short-term component that knows about
``WorkspaceTopicKey``.  The port and store exchange immutable ``TopicData``
snapshots and the adapter stores those frozen objects directly — reads return
the stored snapshot as-is (callers cannot mutate it), so no defensive deep
copy is needed.  A topic ID is globally unique: attempting to write the same
ID into another workspace is rejected instead of silently creating a second
local namespace.
"""

from __future__ import annotations

import threading

from hivememory.core.models import TopicData, WorkspaceIdentity, WorkspaceTopicKey
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import ShortTermStoragePort


class InMemoryShortTermStorage(ShortTermStoragePort):
    """Thread-safe in-memory implementation of :class:`ShortTermStoragePort`.

    ``WorkspaceTopicKey`` remains an implementation detail here.  The adapter
    never mutates a stored ``TopicData``: writers must submit a new frozen
    snapshot (``model_copy``) to change topic content.
    """

    def __init__(self) -> None:
        # 直接存储 frozen 的 TopicData 快照；读取原样返回，无需防御性深拷贝。
        self._topics: dict[WorkspaceTopicKey, TopicData] = {}
        self._workspace_index: dict[tuple[str, str], set[WorkspaceTopicKey]] = {}
        self._topic_scopes: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(workspace: WorkspaceIdentity, topic_id: str) -> WorkspaceTopicKey:
        return WorkspaceTopicKey(
            owner_user_id=workspace.owner_user_id,
            workspace_id=workspace.workspace_id,
            topic_id=topic_id,
        )

    @staticmethod
    def _scope(workspace: WorkspaceIdentity) -> tuple[str, str]:
        return workspace.owner_user_id, workspace.workspace_id

    def get(self, workspace: WorkspaceIdentity, topic_id: str) -> TopicData | None:
        key = self._key(workspace, topic_id)
        with self._lock:
            return self._topics.get(key)

    def put(self, topic: TopicData) -> None:
        if not isinstance(topic, TopicData):
            raise TypeError("short-term storage accepts TopicData snapshots")
        key = self._key(topic.workspace_identity, topic.topic_id)
        scope = self._scope(topic.workspace_identity)
        with self._lock:
            previous_scope = self._topic_scopes.get(topic.topic_id)
            if previous_scope is not None and previous_scope != scope:
                raise ValueError(f"topic '{topic.topic_id}' already belongs to another Workspace")
            self._topics[key] = topic
            self._workspace_index.setdefault(scope, set()).add(key)
            self._topic_scopes[topic.topic_id] = scope

    def delete(self, workspace: WorkspaceIdentity, topic_id: str) -> bool:
        key = self._key(workspace, topic_id)
        scope = self._scope(workspace)
        with self._lock:
            removed = self._topics.pop(key, None)
            if removed is None:
                return False
            topics = self._workspace_index.get(scope)
            if topics is not None:
                topics.discard(key)
                if not topics:
                    self._workspace_index.pop(scope, None)
            self._topic_scopes.pop(topic_id, None)
            return True

    def list_by_workspace(self, workspace: WorkspaceIdentity) -> list[TopicData]:
        scope = self._scope(workspace)
        with self._lock:
            keys = tuple(self._workspace_index.get(scope, ()))
            return [self._topics[key] for key in keys if key in self._topics]

    def list_all(self) -> list[TopicData]:
        with self._lock:
            return list(self._topics.values())

    def count(self, workspace: WorkspaceIdentity) -> int:
        with self._lock:
            return len(self._workspace_index.get(self._scope(workspace), ()))

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="short_term", healthy=True, detail="in-memory")


__all__ = ["InMemoryShortTermStorage"]
