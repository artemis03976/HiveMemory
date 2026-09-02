from unittest.mock import Mock

import pytest

from hivememory.core.models import Identity, TurnEvent, WorkspaceTopicKey
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
            identity_scope=make_identity_scope(actor_identity=identity),
        )

        assert settle_payload is None
        topic_data = self.store.get_topic_data(
            make_identity_scope(actor_identity=identity), topic_id, touch=False
        )
        assert topic_data is not None
        assert len(topic_data.blocks) == 1
        assert topic_data.blocks[0].identity.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_route_and_ingest_reuses_topic_buffer(self):
        identity = Identity(user_id="u1", agent_id="a1")
        topic_id, _ = await self.layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("old topic", "old response", identity),
            identity_scope=make_identity_scope(actor_identity=identity),
        )

        real_topic_id, settle_payload = await self.layer.route_and_ingest(
            topic_id,
            _make_payload("new topic", "new response", identity),
            identity_scope=make_identity_scope(actor_identity=identity),
        )

        assert real_topic_id == topic_id
        assert settle_payload is None
        topic_data = self.store.get_topic_data(
            make_identity_scope(actor_identity=identity), topic_id, touch=False
        )
        assert topic_data is not None
        assert len(topic_data.blocks) == 2

    @pytest.mark.asyncio
    async def test_ingest_payload_requires_turn_events(self):
        identity = Identity(user_id="u1", agent_id="a1")
        topic_id = await self.layer.create_new_topic(
            make_identity_scope(actor_identity=identity)
        )
        payload = InteractionPayload(
            user_message="hi",
            assistant_final_text="hello",
            turn_events=[],
        )

        with pytest.raises(ValueError, match="turn_events is required"):
            await self.layer.ingest_payload(
                payload,
                topic_id,
                identity_scope=make_identity_scope(actor_identity=identity),
            )

    @pytest.mark.asyncio
    async def test_clear_blocks_keeps_topic_shell(self):
        topic_id, _ = await self.layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("hi", "hello"),
            identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        )

        identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
        self.store.clear_blocks(
            WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)
        )

        topic_data = self.store.get_topic_data(identity_scope, topic_id, touch=False)
        assert topic_data is not None
        assert topic_data.blocks == ()
