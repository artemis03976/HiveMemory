"""
Structured ingestion path tests.

Current contract:
1. turn_events=[] raises instead of falling back to legacy assistant text.
2. turn_events=[...] persists assistant_final_text without parsing assistant text.
3. mtp_traces=[item] is persisted directly; trace reduction happens before ingestion.
"""

from unittest.mock import Mock

import pytest

from hivememory.core.models import Identity, TraceItem, TurnEvent
from hivememory.core.protocol import InteractionPayload
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


def _make_layer() -> tuple[SemanticFlowPerceptionLayer, ShortTermMemoryStore]:
    config = SemanticFlowPerceptionConfig()
    relay = Mock()
    relay.should_relay.return_value = None
    store = ShortTermMemoryStore()
    layer = SemanticFlowPerceptionLayer(
        config=config,
        relay_controller=relay,
        short_term_store=store,
        interaction_journal=InMemoryInteractionApplyJournal(),
    )
    return layer, store


def _identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1")


def _identity_scope():
    return make_identity_scope(actor_identity=_identity())


def _turn_event(kind="tool_call", tool_kind="READ", target="alias_x") -> TurnEvent:
    return TurnEvent(
        kind=kind,
        sequence=0,
        role="assistant",
        content="",
        tool_kind=tool_kind,
        tool_name=target if tool_kind == "RUN" else None,
        target=target,
    )


@pytest.mark.asyncio
async def test_missing_turn_events_raises_error():
    layer, _ = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        turn_events=[],
    )
    with pytest.raises(ValueError, match="turn_events is required"):
        await layer.route_and_ingest("NEW_TOPIC", payload, identity_scope=_identity_scope())


@pytest.mark.asyncio
async def test_structured_path_persists_assistant_final_text():
    layer, store = _make_layer()
    turn_event = _turn_event()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="clean reply",
        turn_events=[turn_event],
    )

    topic_id, _ = await layer.route_and_ingest(
        "NEW_TOPIC", payload, identity_scope=_identity_scope()
    )

    topic_data = store.get(_identity_scope(), topic_id, touch=False)
    assert topic_data is not None
    block = topic_data.blocks[0]

    assert block.assistant_final_text == "clean reply"
    assert len(block.turn_events) == 1
    assert block.turn_events[0].kind == turn_event.kind
    assert block.turn_events[0].tool_kind == turn_event.tool_kind


@pytest.mark.asyncio
async def test_structured_path_reduces_turn_events_to_actions():
    layer, store = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="clean reply",
        turn_events=[
            TurnEvent(
                kind="tool_call",
                sequence=0,
                role="assistant",
                content="<< READ | alias_x >>",
                action_id="a1",
                tool_kind="READ",
                tool_name="alias_x",
                target="alias_x",
            ),
            TurnEvent(
                kind="tool_result",
                sequence=1,
                role="user",
                content="result",
                action_id="a1",
                tool_kind="READ",
                tool_name="alias_x",
                status="success",
                render_as="system_tool_result",
            ),
        ],
    )

    topic_id, _ = await layer.route_and_ingest(
        "NEW_TOPIC", payload, identity_scope=_identity_scope()
    )

    topic_data = store.get(_identity_scope(), topic_id, touch=False)
    assert topic_data is not None
    block = topic_data.blocks[0]
    assert len(block.actions) == 1
    assert block.actions[0].action_id == "a1"
    assert block.actions[0].tool_kind == "READ"
    assert block.actions[0].tool_name == "alias_x"
    assert block.actions[0].status == "success"
    assert len(block.actions[0].results) == 1


@pytest.mark.asyncio
async def test_structured_path_persists_payload_mtp_traces():
    layer, store = _make_layer()
    trace = TraceItem(action="SEARCH", query="my query")
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="clean",
        mtp_traces=[trace],
        turn_events=[_turn_event()],
    )

    topic_id, _ = await layer.route_and_ingest(
        "NEW_TOPIC", payload, identity_scope=_identity_scope()
    )

    topic_data = store.get(_identity_scope(), topic_id, touch=False)
    assert topic_data is not None
    block = topic_data.blocks[0]
    assert [t.action for t in block.semantic_traces] == ["SEARCH"]


@pytest.mark.asyncio
async def test_structured_path_keeps_semantic_traces_empty_when_payload_empty():
    layer, store = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="clean",
        mtp_traces=[],
        turn_events=[_turn_event()],
    )

    topic_id, _ = await layer.route_and_ingest(
        "NEW_TOPIC", payload, identity_scope=_identity_scope()
    )

    topic_data = store.get(_identity_scope(), topic_id, touch=False)
    assert topic_data is not None
    block = topic_data.blocks[0]
    assert block.semantic_traces == ()


@pytest.mark.asyncio
async def test_structured_path_empty_final_text_stays_empty():
    layer, store = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="",
        turn_events=[_turn_event()],
    )

    topic_id, _ = await layer.route_and_ingest(
        "NEW_TOPIC", payload, identity_scope=_identity_scope()
    )

    topic_data = store.get(_identity_scope(), topic_id, touch=False)
    assert topic_data is not None
    block = topic_data.blocks[0]
    assert block.assistant_final_text == ""
