"""ChatApplicationService / PassiveIngressService 委托测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService


@pytest.fixture
def mock_patchouli():
    p = MagicMock()
    p.chat = AsyncMock(return_value="chat_result")
    p.chat_stream = MagicMock()
    p.cancel_generation = MagicMock(return_value=True)
    p.ingest_event = AsyncMock(return_value={"buffered": True})
    p.flush_observer_session = AsyncMock(return_value=True)
    return p


class TestChatApplicationService:
    @pytest.mark.asyncio
    async def test_chat_passes_all_args(self, mock_patchouli):
        svc = ChatApplicationService(patchouli=mock_patchouli)
        result = await svc.chat(
            user_message="hi",
            user_id="u1",
            agent_id="agent_x",
            session_id="s1",
            enable_memory_retrieval=False,
            generation_options={"max_tokens": 100},
        )
        mock_patchouli.chat.assert_called_once_with(
            user_message="hi",
            user_id="u1",
            agent_id="agent_x",
            session_id="s1",
            enable_memory_retrieval=False,
            generation_options={"max_tokens": 100},
        )
        assert result == "chat_result"

    @pytest.mark.asyncio
    async def test_chat_stream_yields_events(self, mock_patchouli):
        async def fake_stream(**kwargs):
            yield {"event": "token", "data": {"content": "hi"}}
            yield {"event": "done", "data": {}}

        mock_patchouli.chat_stream = fake_stream
        svc = ChatApplicationService(patchouli=mock_patchouli)
        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)
        assert len(events) == 2
        assert events[0]["event"] == "token"

    def test_cancel_generation(self, mock_patchouli):
        svc = ChatApplicationService(patchouli=mock_patchouli)
        assert svc.cancel_generation("gen-1") is True
        mock_patchouli.cancel_generation.assert_called_once_with("gen-1")


class TestPassiveIngressService:
    @pytest.mark.asyncio
    async def test_ingest_event_passes_args(self, mock_patchouli):
        svc = PassiveIngressService(patchouli=mock_patchouli)
        event = MagicMock()
        result = await svc.ingest_event(
            event=event,
            user_id="u1",
            agent_id="agent_y",
            session_id="s2",
        )
        mock_patchouli.ingest_event.assert_called_once_with(
            event=event,
            user_id="u1",
            agent_id="agent_y",
            session_id="s2",
        )
        assert result == {"buffered": True}

    @pytest.mark.asyncio
    async def test_flush_observer_session(self, mock_patchouli):
        svc = PassiveIngressService(patchouli=mock_patchouli)
        result = await svc.flush_observer_session(user_id="u1", agent_id="a", session_id="s")
        mock_patchouli.flush_observer_session.assert_called_once_with(
            user_id="u1", agent_id="a", session_id="s"
        )
        assert result is True
