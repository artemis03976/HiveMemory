"""Perception contracts must not expose the physical Topic storage key."""

from __future__ import annotations

import inspect

import pytest

from hivememory.core.models import BufferState, LogicalBlock, TurnRecord
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import FlushEvent, FlushReason
from hivememory.engines.perception.relay_controller import NoOpRelayController
from hivememory.engines.perception.trigger_manager import TriggerManager
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.core.protocol.models import InteractionPayload
from hivememory.patchouli.control.interaction_apply_journal import InMemoryInteractionApplyJournal
from hivememory.system.config.patchouli import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


@pytest.mark.unit
def test_flush_event_uses_stable_scope_and_topic_id():
    scope = make_identity_scope()
    event = FlushEvent(
        identity_scope=scope,
        topic_id="topic-1",
        reason=FlushReason.MANUAL_COMPACT,
    )
    assert event.identity_scope == scope
    assert event.topic_id == "topic-1"
    assert not hasattr(event, "topic_key")
    buffer = SemanticBuffer(workspace_identity=scope.workspace_identity)
    assert not hasattr(buffer, "topic_key")


@pytest.mark.unit
def test_perception_interface_signatures_do_not_contain_storage_key():
    for name in (
        "settle_topic",
        "prepare_settlement",
        "commit_settlement",
        "abort_settlement",
        "swap_out_topic",
    ):
        signature = inspect.signature(getattr(BasePerceptionLayer, name))
        assert "topic_key" not in signature.parameters


@pytest.mark.unit
def test_trigger_manager_state_transition_and_settlement_use_crud_store():
    scope = make_identity_scope()
    store = ShortTermMemoryStore()
    topic = store.create(scope)
    block = LogicalBlock(
        turn=TurnRecord(identity=scope.actor_identity, user_query="q", assistant_final_text="a"),
        total_tokens=1,
    )
    store.put(topic.model_copy(update={"blocks": (block,), "total_tokens": 1}))
    manager = TriggerManager(store, NoOpRelayController())

    assert manager.reserve_processing(scope, topic.topic_id)
    assert store.get(scope, topic.topic_id, touch=False).state is BufferState.PROCESSING
    manager.release_processing(scope, topic.topic_id)
    result = manager.settle_and_evict(scope, topic.topic_id, FlushReason.IDLE_TIMEOUT)

    assert result.evicted is True
    assert result.settlement is not None
    assert store.get(scope, topic.topic_id, touch=False) is None


@pytest.mark.asyncio
async def test_semantic_flow_ingest_writes_back_snapshot_without_storage_key():
    from hivememory.engines.perception.semantic_flow_perception_layer import (
        SemanticFlowPerceptionLayer,
    )

    scope = make_identity_scope()
    store = ShortTermMemoryStore()
    layer = SemanticFlowPerceptionLayer(
        SemanticFlowPerceptionConfig(fold_token_threshold=99999),
        NoOpRelayController(),
        store,
        InMemoryInteractionApplyJournal(),
    )
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


@pytest.mark.asyncio
async def test_manual_compact_failure_releases_processing_state():
    from hivememory.engines.perception.models import FlushEvent
    from hivememory.engines.perception.trigger_manager import TriggerManager

    class FailingRelay:
        def generate_summary(self, blocks_to_fold, previous_summary=None):
            raise RuntimeError("summary failed")

    scope = make_identity_scope()
    store = ShortTermMemoryStore()
    topic = store.create(scope)
    block = LogicalBlock(
        turn=TurnRecord(identity=scope.actor_identity, user_query="q", assistant_final_text="a"),
        total_tokens=1,
    )
    store.put(topic.model_copy(update={"blocks": (block, block), "total_tokens": 2}))
    manager = TriggerManager(store, FailingRelay())

    with pytest.raises(RuntimeError, match="summary failed"):
        await manager.resolve_topic(
            FlushEvent(
                identity_scope=scope,
                topic_id=topic.topic_id,
                reason=FlushReason.MANUAL_COMPACT,
            ),
            retain_recent_blocks=1,
        )

    assert store.get(scope, topic.topic_id, touch=False).state is BufferState.IDLE
