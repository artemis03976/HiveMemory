"""ShortTermMemoryStore 纯 CRUD 与快照边界契约测试。"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

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
    # Store 不再提供访问追踪入口；签名中不得有 touch 参数
    assert "touch" not in inspect.signature(ShortTermMemoryStore.get).parameters


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
def test_get_returns_frozen_snapshot_and_put_supports_write_back():
    store = ShortTermMemoryStore()
    scope = make_identity_scope()
    topic = store.create(scope, topic_title="新建话题")
    store.put(_topic_with_block(topic, scope))

    snapshot = store.get(scope, topic.topic_id)
    assert snapshot is not None
    assert snapshot.block_count == 1
    # frozen 模型：读取方无法原地修改，必须提交新快照
    with pytest.raises(ValidationError):
        snapshot.topic_title = "mutated"

    store.put(snapshot.model_copy(update={"topic_title": "updated"}))
    assert store.get(scope, topic.topic_id).topic_title == "updated"


@pytest.mark.unit
def test_create_applies_default_title_and_explicit_title():
    store = ShortTermMemoryStore()
    scope = make_identity_scope()

    default_topic = store.create(scope)
    titled_topic = store.create(scope, topic_title="Session", topic_summary="sum")

    assert default_topic.topic_title == "新建话题"
    assert titled_topic.topic_title == "Session"
    assert titled_topic.topic_summary == "sum"
    assert store.count(scope) == 2
