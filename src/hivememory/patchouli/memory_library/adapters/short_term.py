"""短期存储端口的内存适配器。

该适配器是唯一知晓 ``WorkspaceTopicKey`` 的短期组件。端口与 store 交换不可变
的 ``TopicData`` 快照，适配器直接存储这些冻结对象——读取原样返回已存快照
（调用方无法修改），因此无需防御性深拷贝。topic ID 全局唯一：把同一 ID 写入
另一个 Workspace 会被拒绝，而不是静默创建第二个局部命名空间。
"""

from __future__ import annotations

import threading

from hivememory.core.models import TopicData, WorkspaceIdentity, WorkspaceTopicKey
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import ShortTermStoragePort


class InMemoryShortTermStorage(ShortTermStoragePort):
    """:class:`ShortTermStoragePort` 的线程安全内存实现。

    ``WorkspaceTopicKey`` 在此仅为实现细节。适配器从不修改已存的
    ``TopicData``：写入方必须提交新的冻结快照（``model_copy``）才能变更话题内容。
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
