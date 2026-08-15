from unittest.mock import Mock

import pytest

from hivememory.core.models import Identity, TurnEvent
from hivememory.core.protocol import InteractionPayload
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.system.config import SemanticFlowPerceptionConfig


def _make_payload(user_msg="msg", assistant_msg="reply", identity=None):
    identity = identity or Identity(user_id="u1", agent_id="a1")
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
    )


class TestSemanticFlowPerceptionLayer:
    def setup_method(self):
        self.relay = Mock()
        self.relay.should_relay.return_value = None
        self.store = ShortTermMemoryStore()
        self.layer = SemanticFlowPerceptionLayer(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
            relay_controller=self.relay,
            short_term_store=self.store,
            interaction_journal=InMemoryInteractionApplyJournal(),
        )

    @pytest.mark.asyncio
    async def test_route_and_ingest_adds_block_to_new_topic(self):
        identity = Identity(user_id="u1", agent_id="a1")

        topic_id, settle_payload = await self.layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("hi", "hello", identity),
        )

        assert settle_payload is None
        topic_data = self.store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1
        assert topic_data.blocks[0].identity.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_route_and_ingest_reuses_topic_buffer(self):
        identity = Identity(user_id="u1", agent_id="a1")
        topic_id, _ = await self.layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("old topic", "old response", identity),
        )

        real_topic_id, settle_payload = await self.layer.route_and_ingest(
            topic_id,
            _make_payload("new topic", "new response", identity),
        )

        assert real_topic_id == topic_id
        assert settle_payload is None
        topic_data = self.store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 2

    @pytest.mark.asyncio
    async def test_ingest_payload_requires_turn_events(self):
        identity = Identity(user_id="u1", agent_id="a1")
        topic_id = await self.layer.create_new_topic(identity)
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
            identity=identity,
        )

        with pytest.raises(ValueError, match="turn_events is required"):
            await self.layer.ingest_payload(payload, topic_id)

    @pytest.mark.asyncio
    async def test_clear_buffer_keeps_topic_shell(self):
        topic_id, _ = await self.layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("hi", "hello"),
        )

        cleared = self.store.reset_topic_content(topic_id)

        assert len(cleared) == 1
        topic_data = self.store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert topic_data.blocks == ()
