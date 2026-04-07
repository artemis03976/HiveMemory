import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.perception.models import InteractionPayload
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.patchouli.config import SemanticFlowPerceptionConfig


@patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
@pytest.mark.asyncio
async def test_ingest_payload_uses_identity_agent_id(mock_parser_cls):
    mock_parser_cls.parse.return_value = ("clean", [])
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(),
        relay_controller=Mock(),
    )
    captured_blocks = []
    layer._buffer_manager.add_block = Mock(
        side_effect=lambda topic_id, block: captured_blocks.append(block)
    )
    payload = InteractionPayload(
        user_message="u",
        assistant_message="a",
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
    )
    fake_buffer = MagicMock()
    fake_buffer.topic_id = "topic-xyz"
    layer._buffer_manager.needs_eviction = Mock(return_value=False)
    layer._buffer_manager.create_buffer = Mock(return_value=fake_buffer)

    topic_id = await layer.create_new_topic(Identity(user_id="u1", agent_id="a1"))

    assert topic_id == "topic-xyz"
    layer._buffer_manager.create_buffer.assert_called_once_with(
        user_id="u1",
        title="新建话题",
    )
