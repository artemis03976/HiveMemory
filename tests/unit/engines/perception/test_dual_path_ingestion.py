"""
ingest_payload 双路径单测

验证 Phase 4B 后的双路径:
1. turn_events=[] → 直接消费 legacy assistant_message
2. turn_events=[...] → clean_response == assistant_final_text，且不依赖 assistant_message
3. turn_events=[...] + mtp_traces=[item] → 优先用 mtp_traces，不调用 MTPTraceReducer
4. turn_events=[...] + mtp_traces=[] → 调用 MTPTraceReducer
"""

import pytest
from unittest.mock import Mock, patch

from hivememory.core.models import Identity
from hivememory.engines.perception.models import (
    InteractionPayload,
    TraceItem,
    TurnEvent,
)
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.patchouli.config import SemanticFlowPerceptionConfig


def _make_layer() -> SemanticFlowPerceptionLayer:
    config = SemanticFlowPerceptionConfig()
    relay = Mock()
    relay.should_relay.return_value = None
    layer = SemanticFlowPerceptionLayer(config=config, relay_controller=relay)
    layer.set_generation_callback(Mock())
    return layer


def _identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1")


def _turn_event(kind="mtp_command", verb="READ", target="alias_x") -> TurnEvent:
    return TurnEvent(
        kind=kind,
        sequence=0,
        role="assistant",
        content="",
        verb=verb,
        target=target,
    )


def _trace_item() -> TraceItem:
    return TraceItem(action="READ", target="alias_x")


# ============ Legacy fallback 路径 ============

@pytest.mark.asyncio
async def test_fallback_path_uses_legacy_assistant_message_when_no_turn_events():
    """turn_events=[] → 直接消费 assistant_message，不再依赖 parser"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="legacy assistant text",
        identity=_identity(),
        turn_events=[],  # 空 → fallback
    )
    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert block.clean_response == "legacy assistant text"


@pytest.mark.asyncio
async def test_fallback_uses_mtp_traces_over_parser_traces():
    """legacy fallback 路径: 仅消费 payload.mtp_traces"""
    layer = _make_layer()
    koakuma_trace = TraceItem(action="SEARCH", query="koakuma query")
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw text",
        identity=_identity(),
        mtp_traces=[koakuma_trace],
        turn_events=[],
    )
    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]

    assert any(t.action == "SEARCH" for t in block.semantic_traces)


# ============ 结构化路径 ============

@pytest.mark.asyncio
async def test_structured_path_skips_mtp_log_parser():
    """turn_events 有值时，结构化路径不依赖 assistant_message"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw text with ⟪ MTP ⟫",
        assistant_final_text="干净回复",
        identity=_identity(),
        turn_events=[_turn_event()],
    )
    await layer.route_and_ingest("NEW_TOPIC", payload)


@pytest.mark.asyncio
async def test_structured_path_clean_response_equals_final_text():
    """结构化路径: clean_response == assistant_final_text，且结构化字段落盘到 block"""
    layer = _make_layer()
    turn_event = _turn_event()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw noisy text",
        assistant_final_text="干净回复",
        identity=_identity(),
        turn_events=[turn_event],
    )

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]

    assert block.clean_response == "干净回复"
    assert block.assistant_final_text == "干净回复"
    assert len(block.turn_events) == 1
    assert block.turn_events[0].kind == turn_event.kind
    assert block.turn_events[0].verb == turn_event.verb


@pytest.mark.asyncio
async def test_structured_path_mtp_traces_takes_priority_over_reducer():
    """结构化路径 + mtp_traces 有值: 优先用 mtp_traces，不调用 MTPTraceReducer"""
    layer = _make_layer()
    koakuma_trace = TraceItem(action="SEARCH", query="my query")
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw",
        assistant_final_text="clean",
        identity=_identity(),
        mtp_traces=[koakuma_trace],
        turn_events=[_turn_event()],
    )

    with patch("hivememory.patchouli.mtp.trace_reducer.MTPTraceReducer.reduce") as mock_reduce:
        await layer.route_and_ingest("NEW_TOPIC", payload)
        mock_reduce.assert_not_called()

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert any(t.action == "SEARCH" for t in block.semantic_traces)


@pytest.mark.asyncio
async def test_structured_path_calls_reducer_when_no_mtp_traces():
    """结构化路径 + mtp_traces=[]: 调用 MTPTraceReducer"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw",
        assistant_final_text="clean",
        identity=_identity(),
        mtp_traces=[],
        turn_events=[_turn_event()],
    )

    reducer_trace = TraceItem(action="READ", target="alias_x")

    with patch("hivememory.patchouli.mtp.trace_reducer.MTPTraceReducer.reduce",
               return_value=[reducer_trace]) as mock_reduce:
        await layer.route_and_ingest("NEW_TOPIC", payload)
        mock_reduce.assert_called_once_with(payload.turn_events)


@pytest.mark.asyncio
async def test_structured_path_empty_final_text_keeps_clean_response_empty():
    """结构化路径 + assistant_final_text 为空: 不再回退到 parser"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw text",
        assistant_final_text="",
        identity=_identity(),
        turn_events=[_turn_event()],
    )

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert block.clean_response == ""


# ============ 兼容性验证 ============

@pytest.mark.asyncio
async def test_raw_response_still_populated_from_assistant_message():
    """无论哪条路径，raw_response 始终来自 payload.assistant_message（调试兼容）"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="⟪ RAW ⟫ text",
        assistant_final_text="clean",
        identity=_identity(),
        turn_events=[_turn_event()],
    )

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert block.raw_response == "⟪ RAW ⟫ text"
