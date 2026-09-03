"""In-memory adapter for the short-term storage port.

The adapter is the only short-term component that knows about
``WorkspaceTopicKey`` and mutable ``SemanticBuffer`` instances.  The port and
store exchange immutable ``TopicData`` snapshots, so callers cannot mutate the
adapter's resident state by retaining a value returned from ``get`` or
``list_by_workspace``.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Set

from hivememory.core.models import TopicData, WorkspaceIdentity, WorkspaceTopicKey
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import ShortTermStoragePort


class InMemoryShortTermStorage(ShortTermStoragePort):
    """Thread-safe in-memory implementation of :class:`ShortTermStoragePort`.

    ``WorkspaceTopicKey`` remains an implementation detail here.  A topic ID is
    globally unique: attempting to write the same ID into another workspace is
    rejected instead of silently creating a second local namespace.
    """

    def __init__(self) -> None:
        self._buffers: Dict[WorkspaceTopicKey, SemanticBuffer] = {}
        self._workspace_index: Dict[tuple[str, str], Set[WorkspaceTopicKey]] = {}
        self._topic_scopes: Dict[str, tuple[str, str]] = {}
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

    @staticmethod
    def _buffer_from_topic(topic: TopicData) -> SemanticBuffer:
        # ``model_dump`` followed by validation gives the adapter its own
        # mutable representation and deep-copies nested blocks/bindings.
        return SemanticBuffer.model_validate(topic.model_dump())

    @staticmethod
    def _topic_from_buffer(buffer: SemanticBuffer) -> TopicData:
        return TopicData(
            topic_id=buffer.topic_id,
            workspace_identity=buffer.workspace_identity,
            current_agent_id=buffer.current_agent_id,
            topic_title=buffer.topic_title,
            topic_summary=buffer.topic_summary,
            state_summary=buffer.state_summary,
            blocks=tuple(block.model_copy(deep=True) for block in buffer.blocks),
            bindings=tuple(binding.model_copy(deep=True) for binding in buffer.bindings),
            state=buffer.state,
            last_update=buffer.last_update,
            last_accessed_at=buffer.last_accessed_at,
            total_tokens=buffer.total_tokens,
            model_used=buffer.model_used,
        )

    def get(self, workspace: WorkspaceIdentity, topic_id: str) -> TopicData | None:
        key = self._key(workspace, topic_id)
        with self._lock:
            buffer = self._buffers.get(key)
            return self._topic_from_buffer(buffer) if buffer is not None else None

    def put(self, topic: TopicData) -> None:
        if not isinstance(topic, TopicData):
            raise TypeError("short-term storage accepts TopicData snapshots")
        key = self._key(topic.workspace_identity, topic.topic_id)
        scope = self._scope(topic.workspace_identity)
        with self._lock:
            previous_scope = self._topic_scopes.get(topic.topic_id)
            if previous_scope is not None and previous_scope != scope:
                raise ValueError(
                    f"topic '{topic.topic_id}' already belongs to another Workspace"
                )
            self._buffers[key] = self._buffer_from_topic(topic)
            self._workspace_index.setdefault(scope, set()).add(key)
            self._topic_scopes[topic.topic_id] = scope

    def delete(self, workspace: WorkspaceIdentity, topic_id: str) -> bool:
        key = self._key(workspace, topic_id)
        scope = self._scope(workspace)
        with self._lock:
            removed = self._buffers.pop(key, None)
            if removed is None:
                return False
            topics = self._workspace_index.get(scope)
            if topics is not None:
                topics.discard(key)
                if not topics:
                    self._workspace_index.pop(scope, None)
            self._topic_scopes.pop(topic_id, None)
            return True

    def list_by_workspace(self, workspace: WorkspaceIdentity) -> List[TopicData]:
        scope = self._scope(workspace)
        with self._lock:
            keys = tuple(self._workspace_index.get(scope, ()))
            return [
                self._topic_from_buffer(self._buffers[key])
                for key in keys
                if key in self._buffers
            ]

    def list_all(self) -> List[TopicData]:
        with self._lock:
            return [self._topic_from_buffer(buffer) for buffer in self._buffers.values()]

    def count(self, workspace: WorkspaceIdentity) -> int:
        with self._lock:
            return len(self._workspace_index.get(self._scope(workspace), ()))

    async def check_health(self) -> StorageHealthComponent:
        return StorageHealthComponent(name="short_term", healthy=True, detail="in-memory")


__all__ = ["InMemoryShortTermStorage"]
