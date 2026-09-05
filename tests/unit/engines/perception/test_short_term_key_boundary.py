"""Perception contracts must not expose the physical Topic storage key."""

from __future__ import annotations

from hivememory.core.models import TopicData
from hivememory.engines.perception.models import FlushEvent, TriggerReason
from tests.helpers.workspace import make_identity_scope


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
