from unittest.mock import Mock

import pytest

from hivememory.core.models import Identity, LogicalBlock, TraceItem, TurnEvent, TurnRecord
from hivememory.core.protocol import InteractionPayload
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.system.config import SemanticFlowPerceptionConfig


def _make_identity():
    return Identity(user_id="u1", agent_id="a1")


def _make_payload(user_msg="hello", assistant_msg="world", identity=None, traces=None):
    identity = identity or _make_identity()
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_msg,
            )
        ],
        identity=identity,
        mtp_traces=traces or [],
    )


def _make_layer(
    *,
    fold_token_threshold=999999,
    fold_retain_recent_blocks=2,
    relay=None,
    store=None,
):
    config = SemanticFlowPerceptionConfig(
        fold_token_threshold=fold_token_threshold,
        fold_retain_recent_blocks=fold_retain_recent_blocks,
    )
    relay = relay or Mock()
    relay.should_relay.return_value = None
    return SemanticFlowPerceptionLayer(
        config=config,
        relay_controller=relay,
        short_term_store=store or ShortTermMemoryStore(),
    )


class TestBlockTokenComputation:
    @pytest.mark.asyncio
    async def test_block_total_tokens_computed(self):
        layer = _make_layer()

        topic_id, settle_payload = await layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("What is Python?", "Python is a language"),
        )

        assert settle_payload is None
        topic_data = layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1
        assert topic_data.blocks[0].total_tokens > 0
        assert topic_data.total_tokens > 0

    @pytest.mark.asyncio
    async def test_block_tokens_include_traces(self):
        layer = _make_layer()
        traces = [
            TraceItem(action="SEARCH", query="how to sort a list"),
            TraceItem(action="READ", target="my_notes_alias"),
        ]

        topic_id, settle_payload = await layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("q", "a", traces=traces),
        )

        assert settle_payload is None
        topic_data = layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert topic_data.blocks[0].total_tokens > 0


class TestPageFoldingThreshold:
    @pytest.mark.asyncio
    async def test_fold_not_triggered_below_threshold(self):
        layer = _make_layer(fold_token_threshold=999999)
        identity = _make_identity()

        topic_id = None
        settle_payload = None
        for i in range(5):
            target = topic_id or "NEW_TOPIC"
            topic_id, settle_payload = await layer.route_and_ingest(
                target,
                _make_payload(f"msg{i}", f"reply{i}", identity),
            )

        assert settle_payload is None
        topic_data = layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 5
        assert topic_data.state_summary == ""

    @pytest.mark.asyncio
    async def test_token_overflow_compacts_clears_blocks_and_returns_no_settlement(self):
        relay = Mock()
        relay.should_relay.return_value = None
        relay.generate_summary.return_value = "Test summary"
        layer = _make_layer(fold_token_threshold=10, relay=relay)

        topic_id, settle_payload = await layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("x" * 400, "short reply"),
        )

        assert settle_payload is None
        relay.generate_summary.assert_called_once()
        topic_data = layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert topic_data.state_summary == "Test summary"
        assert topic_data.blocks == ()
        assert topic_data.total_tokens == 0

    @pytest.mark.asyncio
    async def test_store_update_summary_can_retain_recent_blocks_independently(self):
        store = ShortTermMemoryStore()
        buffer = store.create_buffer(_make_identity().user_id)
        topic_id = buffer.topic_id

        for i in range(10):
            store.add_block(
                topic_id,
                LogicalBlock(
                    turn=TurnRecord(
                        user_query=f"question {i}",
                        assistant_final_text=f"answer {i}",
                    ),
                    total_tokens=20,
                ),
            )

        folded = store.update_summary(topic_id, "Test summary", retain_count=2)

        topic_data = store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert folded == 8
        assert len(topic_data.blocks) == 2
        assert topic_data.state_summary == "Test summary"
        assert topic_data.total_tokens == 40


class TestPageFoldingCumulative:
    @pytest.mark.asyncio
    async def test_fold_cumulative_summary(self):
        relay = Mock()
        relay.should_relay.return_value = None
        relay.generate_summary.side_effect = (
            lambda blocks_to_fold, previous_summary: previous_summary + "---folded"
        )
        layer = _make_layer(fold_token_threshold=50, relay=relay)
        identity = _make_identity()

        topic_id = await layer.create_new_topic(identity)
        for i in range(4):
            await layer.route_and_ingest(
                topic_id,
                _make_payload(f"wave1 q{i} " * 20, f"wave1 a{i} " * 20, identity),
            )

        topic_data = layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        first_summary = topic_data.state_summary
        assert first_summary != ""

        for i in range(4):
            await layer.route_and_ingest(
                topic_id,
                _make_payload(f"wave2 q{i} " * 20, f"wave2 a{i} " * 20, identity),
            )

        topic_data = layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert "---" in topic_data.state_summary
        assert first_summary in topic_data.state_summary
