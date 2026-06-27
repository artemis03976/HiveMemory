import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import Identity, TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.system.config import SemanticFlowPerceptionConfig


@pytest.mark.asyncio
async def test_ingest_payload_uses_identity_agent_id():
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(),
        relay_controller=Mock(),
        short_term_store=ShortTermMemoryStore(),
    )
    captured_blocks = []
    layer._short_term_store.add_block = Mock(
        side_effect=lambda topic_id, block: captured_blocks.append(block)
    )
    payload = InteractionPayload(
        user_message="u",
        assistant_final_text="a",
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content="a",
            )
        ],
        identity=Identity(user_id="u1", agent_id="coder_doll"),
    )

    await layer.ingest_payload(payload, topic_id="topic-1")

    assert len(captured_blocks) == 1
    assert captured_blocks[0].identity.agent_id == "coder_doll"


@pytest.mark.asyncio
async def test_create_new_topic_calls_create_buffer_with_user_id():
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(),
        relay_controller=Mock(),
        short_term_store=ShortTermMemoryStore(),
    )
    fake_buffer = MagicMock()
    fake_buffer.topic_id = "topic-xyz"
    layer._short_term_store.needs_eviction = Mock(return_value=False)
    layer._short_term_store.create_buffer = Mock(return_value=fake_buffer)

    topic_id = await layer.create_new_topic(Identity(user_id="u1", agent_id="a1"))

    assert topic_id == "topic-xyz"
    layer._short_term_store.create_buffer.assert_called_once_with(
            user_id="u1",
            topic_title="新建话题",
            topic_summary=""
        )
