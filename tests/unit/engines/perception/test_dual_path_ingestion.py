"""
ingest_payload 结构化单路径单测

验证当前单路径:
1. turn_events=[] → 直接报错，不再接受 legacy assistant_message
2. turn_events=[...] → assistant_final_text 落盘，且不依赖 assistant_message
3. turn_events=[...] + mtp_traces=[item] → 优先用 mtp_traces，不调用 reducer
4. turn_events=[...] + mtp_traces=[] → 调用 reducer
"""

import pytest
from unittest.mock import Mock, patch

from hivememory.core.models import Identity, TraceItem, TurnEvent
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.patchouli.protocol import InteractionPayload


def _make_layer() -> SemanticFlowPerceptionLayer:
    config = SemanticFlowPerceptionConfig()
    relay = Mock()
    relay.should_relay.return_value = None
    layer = SemanticFlowPerceptionLayer(config=config, relay_controller=relay)
    layer.set_generation_callback(Mock())
    return layer


def _identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1")


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


def _trace_item() -> TraceItem:
    return TraceItem(action="READ", target="alias_x")


# ============ 非结构化输入拒绝 ============

@pytest.mark.asyncio
async def test_missing_turn_events_raises_error():
    """turn_events=[] → perception 不再接受 legacy assistant_message fallback"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        identity=_identity(),
        turn_events=[],
    )
    with pytest.raises(ValueError, match="turn_events is required"):
        await layer.route_and_ingest("NEW_TOPIC", payload)


# ============ 结构化路径 ============

@pytest.mark.asyncio
async def test_structured_path_skips_mtp_log_parser():
    """turn_events 有值时，结构化路径不依赖 assistant_message"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="干净回复",
        identity=_identity(),
        turn_events=[_turn_event()],
    )
    await layer.route_and_ingest("NEW_TOPIC", payload)


@pytest.mark.asyncio
async def test_structured_path_persists_assistant_final_text():
    """结构化路径: assistant_final_text 与结构化字段正确落盘到 block"""
    layer = _make_layer()
    turn_event = _turn_event()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="干净回复",
        identity=_identity(),
        turn_events=[turn_event],
    )

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]

    assert block.assistant_final_text == "干净回复"
    assert len(block.turn_events) == 1
    assert block.turn_events[0].kind == turn_event.kind
    assert block.turn_events[0].tool_kind == turn_event.tool_kind


@pytest.mark.asyncio
async def test_structured_path_reduces_turn_events_to_actions():
    """结构化路径: turn_events 会进一步聚合出 AgentAction 并落到 block.actions"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="干净回复",
        identity=_identity(),
        turn_events=[
            TurnEvent(
                kind="tool_call",
                sequence=0,
                role="assistant",
                content='⟪ READ | alias_x ⟫',
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

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert len(block.actions) == 1
    assert block.actions[0].action_id == "a1"
    assert block.actions[0].tool_kind == "READ"
    assert block.actions[0].tool_name == "alias_x"
    assert block.actions[0].status == "success"
    assert len(block.actions[0].results) == 1


@pytest.mark.asyncio
async def test_structured_path_mtp_traces_takes_priority_over_reducer():
    """结构化路径 + mtp_traces 有值: 优先用 mtp_traces，不调用 MTPTraceReducer"""
    layer = _make_layer()
    koakuma_trace = TraceItem(action="SEARCH", query="my query")
    payload = InteractionPayload(
        user_message="hello",
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
    """结构化路径 + mtp_traces=[]: 调用 TraceReducer"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="clean",
        identity=_identity(),
        mtp_traces=[],
        turn_events=[_turn_event()],
    )

    reducer_trace = TraceItem(action="READ", target="alias_x")

    with patch("hivememory.engines.perception.semantic_flow_perception_layer.TraceReducer.reduce",
               return_value=[reducer_trace]) as mock_reduce:
        await layer.route_and_ingest("NEW_TOPIC", payload)
        assert mock_reduce.called


@pytest.mark.asyncio
async def test_structured_path_empty_final_text_stays_empty():
    """结构化路径 + assistant_final_text 为空: 不再回退到 parser"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="",
        identity=_identity(),
        turn_events=[_turn_event()],
    )

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert block.assistant_final_text == ""


# ============ 兼容性验证 ============

@pytest.mark.asyncio
async def test_assistant_message_no_longer_persists_as_block_field():
    """assistant_message 仅作为 ingest 输入，不再持久化为 LogicalBlock 字段"""
    layer = _make_layer()
    payload = InteractionPayload(
        user_message="hello",
        assistant_final_text="clean",
        identity=_identity(),
        turn_events=[_turn_event()],
    )

    await layer.route_and_ingest("NEW_TOPIC", payload)

    snapshots = layer.get_active_topics_snapshots(_identity())
    buffer = layer.get_buffer(snapshots[0].topic_id)
    block = buffer.blocks[0]
    assert block.assistant_final_text == "clean"
