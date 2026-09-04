"""Perception contracts must not expose the physical Topic storage key."""

from __future__ import annotations

import inspect

import pytest

from hivememory.core.models import TopicData
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import FlushEvent, TriggerReason
from tests.helpers.workspace import make_identity_scope


@pytest.mark.unit
def test_flush_event_and_topic_snapshot_use_stable_scope_and_topic_id():
    scope = make_identity_scope()
    event = FlushEvent(
        identity_scope=scope,
        topic_id="topic-1",
        reason=TriggerReason.MANUAL_COMPACT,
    )
    assert event.identity_scope == scope
    assert event.topic_id == "topic-1"
    assert not hasattr(event, "topic_key")
    topic = TopicData(
        topic_id="topic-1",
        workspace_identity=scope.workspace_identity,
        topic_title="t",
        last_update=1.0,
    )
    assert not hasattr(topic, "topic_key")


@pytest.mark.unit
def test_perception_interface_signatures_do_not_contain_storage_key():
    for name in (
        "ingest_payload",
        "route_and_ingest",
        "prepare_topic",
    ):
        signature = inspect.signature(getattr(BasePerceptionLayer, name))
        assert "topic_key" not in signature.parameters


@pytest.mark.asyncio
async def test_semantic_flow_ingest_writes_back_snapshot_without_storage_key():
    from tests.helpers.perception import build_perception_stack

    scope = make_identity_scope()
    layer, store, _ = build_perception_stack()
    payload = InteractionPayload(
        user_message="question",
        assistant_final_text="answer",
        turn_events=[
            {"kind": "user_message", "sequence": 0, "role": "user", "content": "question"},
            {"kind": "assistant_message", "sequence": 1, "role": "assistant", "content": "answer"},
        ],
    )

    topic_id, settlement = await layer.route_and_ingest(
        "NEW_TOPIC",
        payload,
        identity_scope=scope,
    )

    assert settlement is None
    topic = store.get(scope, topic_id)
    assert topic is not None and topic.block_count == 1
    assert not hasattr(topic, "topic_key")
