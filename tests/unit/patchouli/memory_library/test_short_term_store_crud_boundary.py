"""ShortTermMemoryStore CRUD and snapshot-boundary contract tests."""

from __future__ import annotations

import inspect

import pytest

from hivememory.core.models import IdentityScope, LogicalBlock, TurnRecord
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from tests.helpers.workspace import make_identity_scope


def _topic_with_block(topic, scope: IdentityScope):
    actor = scope.actor_identity
    block = LogicalBlock(
        turn=TurnRecord(identity=actor, user_query="question", assistant_final_text="answer"),
        total_tokens=2,
    )
    return topic.model_copy(update={"blocks": (block,), "total_tokens": 2})


@pytest.mark.unit
def test_store_public_surface_is_crud_only_and_key_free():
    public_methods = {
        name
        for name, member in inspect.getmembers(ShortTermMemoryStore, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_methods == {
        "get",
        "put",
        "create",
        "delete",
        "list_by_workspace",
        "list_all",
        "count",
        "check_health",
    }
    assert not any("key" in name for name in public_methods)


@pytest.mark.unit
def test_store_enforces_workspace_boundary_and_global_topic_identity():
    store = ShortTermMemoryStore()
    first_scope = make_identity_scope(user_id="u1", workspace_id="workspace-a")
    second_scope = make_identity_scope(user_id="u2", workspace_id="workspace-b")
    topic = store.create(first_scope, topic_id="globally-unique")

    assert store.get(second_scope, topic.topic_id) is None
    with pytest.raises(ValueError, match="another Workspace"):
        store.put(topic.model_copy(update={"workspace_identity": second_scope.workspace_identity}))


@pytest.mark.unit
def test_store_returns_independent_immutable_snapshots_and_supports_write_back():
    store = ShortTermMemoryStore()
    scope = make_identity_scope()
    topic = store.create(scope)
    store.put(_topic_with_block(topic, scope))

    first = store.get(scope, topic.topic_id)
    second = store.get(scope, topic.topic_id)
    assert first is not None and second is not None
    assert first is not second
    assert first.blocks is not second.blocks
    assert first.blocks[0] is not second.blocks[0]

    replacement = first.model_copy(update={"topic_title": "updated"})
    store.put(replacement)
    assert store.get(scope, topic.topic_id).topic_title == "updated"


@pytest.mark.unit
def test_store_touch_updates_access_metadata_through_snapshot_write_back():
    store = ShortTermMemoryStore()
    scope = make_identity_scope()
    topic = store.create(scope)
    before = store.get(scope, topic.topic_id, touch=False).last_accessed_at
    touched = store.get(scope, topic.topic_id, touch=True)
    assert touched.last_accessed_at >= before
