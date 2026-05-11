"""
ingest_payload 双路径单测

验证 Phase 1 新增的结构化优先路径与 MTPLogParser 降级路径:
1. turn_events=[] → MTPLogParser 被调用（fallback）
2. turn_events=[...] → MTPLogParser.parse 不被调用，clean_response == assistant_final_text
3. turn_events=[...] + mtp_traces=[item] → 优先用 mtp_traces，不调用 MTPTraceReducer
4. turn_events=[...] + mtp_traces=[] → 调用 MTPTraceReducer
5. turn_events=[...] + assistant_final_text="" → 防御性回退到 MTPLogParser
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

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


# ============ Fallback 路径 ============

@pytest.mark.asyncio
async def test_fallback_path_calls_mtp_log_parser_when_no_turn_events():
    """turn_events=[] → MTPLogParser.parse 被调用"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="⟪ READ | alias_x ⟫ 找到了",
        identity=_identity(),
        turn_events=[],  # 空 → fallback
    )

    with patch("hivememory.patchouli.mtp.log_parser.MTPLogParser.parse",
               return_value=("找到了", [_trace_item()])) as mock_parse:
        await layer.route_and_ingest("NEW_TOPIC", payload)
        mock_parse.assert_called_once_with(payload.assistant_message)


@pytest.mark.asyncio
async def test_fallback_uses_mtp_traces_over_parser_traces():
    """fallback 路径: mtp_traces 优先于 parser 的 fallback_traces"""
    layer = _make_layer()
    koakuma_trace = TraceItem(action="SEARCH", query="koakuma query")
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw text",
        identity=_identity(),
        mtp_traces=[koakuma_trace],
        turn_events=[],
    )

    parser_trace = TraceItem(action="READ", target="parser_target")
    with patch("hivememory.patchouli.mtp.log_parser.MTPLogParser.parse",
               return_value=("clean text", [parser_trace])):
        await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]

    # 应用 koakuma_trace，不是 parser_trace
    assert any(t.action == "SEARCH" for t in block.semantic_traces)
    assert not any(t.action == "READ" for t in block.semantic_traces)


# ============ 结构化路径 ============

@pytest.mark.asyncio
async def test_structured_path_skips_mtp_log_parser():
    """turn_events 有值时，MTPLogParser.parse 不被调用"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw text with ⟪ MTP ⟫",
        assistant_final_text="干净回复",
        identity=_identity(),
        turn_events=[_turn_event()],
    )

    with patch("hivememory.patchouli.mtp.log_parser.MTPLogParser.parse",
               return_value=("should not be called", [])) as mock_parse:
        await layer.route_and_ingest("NEW_TOPIC", payload)
        mock_parse.assert_not_called()


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
async def test_structured_path_defensive_fallback_when_final_text_empty():
    """结构化路径 + assistant_final_text 为空: 防御性回退调用 MTPLogParser"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_message="raw text",
        assistant_final_text="",   # 空 → 触发防御性回退
        identity=_identity(),
        turn_events=[_turn_event()],
    )

    with patch("hivememory.patchouli.mtp.log_parser.MTPLogParser.parse",
               return_value=("fallback clean", [])) as mock_parse:
        await layer.route_and_ingest("NEW_TOPIC", payload)
        mock_parse.assert_called_once()

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert block.clean_response == "fallback clean"


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
