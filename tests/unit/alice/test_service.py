import asyncio
from unittest.mock import AsyncMock

import pytest

from hivememory.alice.service import AliceService
from hivememory.core.models import Identity, OMNI_DOLL_PROFILE
from hivememory.core.protocol.models import AgentRunContext, AgentRunResult


def _context() -> AgentRunContext:
    return AgentRunContext(
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        topic_id="topic-1",
        user_message="hello",
        agent_profile=OMNI_DOLL_PROFILE,
    )


@pytest.mark.asyncio
async def test_run_agent_delegates_to_runtime():
    runtime = AsyncMock()
    runtime.run_agent.return_value = AgentRunResult(final_text="done")
    service = AliceService(runtime=runtime)
    context = _context()
    cancel_event = asyncio.Event()

    result = await service.run_agent(
        context,
        generation_options={"temperature": 0.1},
        cancel_event=cancel_event,
    )

    assert result.final_text == "done"
    runtime.run_agent.assert_awaited_once_with(
        agent_run_context=context,
        generation_options={"temperature": 0.1},
        cancel_event=cancel_event,
    )


@pytest.mark.asyncio
async def test_run_agent_stream_yields_runtime_events():
    class Runtime:
        async def run_agent_stream(self, **kwargs):
            yield {"event": "token", "data": "a"}
            yield {"event": "done"}

    service = AliceService(runtime=Runtime())

    events = [event async for event in service.run_agent_stream(_context())]

    assert events == [{"event": "token", "data": "a"}, {"event": "done"}]
