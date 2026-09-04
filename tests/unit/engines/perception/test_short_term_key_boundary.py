"""Perception contracts must not expose the physical Topic storage key."""

from __future__ import annotations

import inspect

import pytest

from hivememory.core.models import BufferState, LogicalBlock, TurnRecord
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import FlushEvent, TriggerReason
from hivememory.engines.perception.relay_controller import NoOpRelayController
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.topic_buffer import TopicBufferService
from tests.helpers.workspace import make_identity_scope


@pytest.mark.unit
def test_flush_event_uses_stable_scope_and_topic_id():
    scope = make_identity_scope()
    event = FlushEvent(
        identity_scope=scope,
        topic_id="topic-1",
        reason=TriggerReason.MANUAL_COMPACT,
    )
    assert event.identity_scope == scope
    assert event.topic_id == "topic-1"
    assert not hasattr(event, "topic_key")
    buffer = SemanticBuffer(workspace_identity=scope.workspace_identity)
    assert not hasattr(buffer, "topic_key")


@pytest.mark.unit
def test_perception_interface_signatures_do_not_contain_storage_key():
    for name in (
        "ingest_payload",
        "route_and_ingest",
        "prepare_topic",
    ):
        signature = inspect.signature(getattr(BasePerceptionLayer, name))
        assert "topic_key" not in signature.parameters


@pytest.mark.unit
def test_topic_buffer_service_state_transition_and_settlement_use_crud_store():
    scope = make_identity_scope()
    store = ShortTermMemoryStore()
    topic = store.create(scope)
    block = LogicalBlock(
        turn=TurnRecord(identity=scope.actor_identity, user_query="q", assistant_final_text="a"),
        total_tokens=1,
    )
    store.put(topic.model_copy(update={"blocks": (block,), "total_tokens": 1}))
    service = TopicBufferService(store=store, relay_controller=NoOpRelayController())

    assert service.reserve_processing(scope, topic.topic_id)
    assert store.get(scope, topic.topic_id, touch=False).state is BufferState.PROCESSING
    service.release_processing(scope, topic.topic_id)

    reservation = service.begin_settlement(scope, topic.topic_id, TriggerReason.IDLE_TIMEOUT)
    assert reservation is not None and reservation.task is not None
    outcome = service.complete_settlement(
        scope, topic.topic_id,
        generation_task_id="task-1",
        reason=TriggerReason.IDLE_TIMEOUT,
    )
    assert outcome.removed is True
    assert store.get(scope, topic.topic_id, touch=False) is None


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
    topic = store.get(scope, topic_id, touch=False)
    assert topic is not None and topic.block_count == 1
    assert not hasattr(topic, "topic_key")
