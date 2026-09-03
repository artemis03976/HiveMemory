"""InMemoryShortTermStorage adapter contract tests."""

from __future__ import annotations

import threading

import pytest

from hivememory.core.models import LogicalBlock, TopicData, TurnRecord
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from tests.helpers.workspace import make_identity_scope


def _topic(scope, topic_id: str) -> TopicData:
    return TopicData(
        topic_id=topic_id,
        workspace_identity=scope.workspace_identity,
        topic_title=f"topic-{topic_id}",
        last_update=1.0,
        last_accessed_at=1.0,
        blocks=(
            LogicalBlock(
                turn=TurnRecord(
                    identity=scope.actor_identity,
                    user_query="q",
                    assistant_final_text="a",
                )
            ),
        ),
    )


@pytest.mark.unit
def test_get_put_delete_use_workspace_and_topic_id():
    storage = InMemoryShortTermStorage()
    scope = make_identity_scope(user_id="u1", workspace_id="workspace-a")
    topic = _topic(scope, "topic-1")

    assert storage.get(scope.workspace_identity, topic.topic_id) is None
    storage.put(topic)
    loaded = storage.get(scope.workspace_identity, topic.topic_id)
    assert loaded == topic
    assert storage.delete(scope.workspace_identity, topic.topic_id) is True
    assert storage.delete(scope.workspace_identity, topic.topic_id) is False


@pytest.mark.unit
def test_workspace_listing_and_count_are_isolated():
    storage = InMemoryShortTermStorage()
    first = make_identity_scope(user_id="u1", workspace_id="workspace-a")
    second = make_identity_scope(user_id="u1", workspace_id="workspace-b")
    storage.put(_topic(first, "topic-a"))
    storage.put(_topic(second, "topic-b"))

    topics = storage.list_by_workspace(first.workspace_identity)
    assert {topic.topic_id for topic in topics} == {"topic-a"}
    assert storage.count(second.workspace_identity) == 1
    assert len(storage.list_all()) == 2


@pytest.mark.unit
def test_topic_id_cannot_be_reused_in_another_workspace():
    storage = InMemoryShortTermStorage()
    first = make_identity_scope(user_id="u1", workspace_id="workspace-a")
    second = make_identity_scope(user_id="u2", workspace_id="workspace-b")
    storage.put(_topic(first, "globally-unique"))

    with pytest.raises(ValueError, match="another Workspace"):
        storage.put(_topic(second, "globally-unique"))


@pytest.mark.unit
def test_get_returns_deep_snapshot_copy():
    storage = InMemoryShortTermStorage()
    scope = make_identity_scope()
    storage.put(_topic(scope, "topic-1"))

    first = storage.get(scope.workspace_identity, "topic-1")
    second = storage.get(scope.workspace_identity, "topic-1")
    assert first is not second
    assert first.blocks is not second.blocks
    assert first.blocks[0] is not second.blocks[0]


@pytest.mark.unit
def test_adapter_map_operations_are_thread_safe():
    storage = InMemoryShortTermStorage()
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            scope = make_identity_scope(user_id=f"u{index}")
            for item in range(50):
                topic = _topic(scope, f"topic-{index}-{item}")
                storage.put(topic)
                assert storage.get(scope.workspace_identity, topic.topic_id) is not None
        except Exception as exc:  # pragma: no cover - failure is asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(storage.list_all()) == 200
