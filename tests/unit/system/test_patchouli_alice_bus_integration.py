"""Patchouli 通过 GlobalSystemBus 调用 Alice 的单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity
from hivememory.core.protocol.models import ChatResult
from hivememory.patchouli.service import PatchouliService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


@pytest.mark.asyncio
async def test_chat_uses_global_bus_for_alice_run_and_alias_registration():
    kernel = MagicMock()
    eye = MagicMock()
    bus = GlobalSystemBus()

    gaze_result = MagicMock(
        target_topic="topic_1",
        new_topic_title=None,
        new_topic_summary=None,
    )
    hot_result = MagicMock(
        retrieved_memories=[MagicMock()],
        rewritten="rewritten query",
        worth_saving=True,
    )
    loop_result = ChatResult(final_text="hello")

    eye.gaze = AsyncMock(return_value=gaze_result)
    kernel.load_agent_profile = MagicMock(return_value=MagicMock())
    kernel.get_topic_snapshots = AsyncMock(return_value=[])
    kernel.prepare_topic = AsyncMock(
        return_value=("topic_1", {"topics": []}, {"blocks": []})
    )
    kernel.handle_hot = AsyncMock(return_value=hot_result)

    register_aliases = AsyncMock(return_value=None)
    run_agent = AsyncMock(return_value=loop_result)
    bus.register(GlobalRoutes.ALICE_REGISTER_PRERETRIEVAL_ALIASES, register_aliases)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, run_agent)

    service = PatchouliService(kernel=kernel, eye=eye, global_bus=bus)
    service._assemble_messages_from_context = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    service._chat_post_process = AsyncMock()

    result = await service.chat(
        user_message="hi",
        user_id="u1",
    )

    assert result is loop_result
    register_aliases.assert_awaited_once_with(hot_result.retrieved_memories)
    run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_post_process_reads_interaction_state_via_global_bus():
    kernel = MagicMock()
    kernel.submit_interaction = AsyncMock(return_value=None)
    service = PatchouliService(
        kernel=kernel,
        eye=MagicMock(),
        global_bus=GlobalSystemBus(),
    )

    interaction_state = {
        "mtp_traces": [],
        "write_focus": None,
        "update_focus": None,
    }
    service._global_bus.register(
        GlobalRoutes.ALICE_GET_INTERACTION_STATE,
        AsyncMock(return_value=interaction_state),
    )

    await service._chat_post_process(
        messages=[{"role": "user", "content": "hi"}],
        loop_result=ChatResult(final_text="done"),
        hot_result=MagicMock(rewritten="rewritten", worth_saving=True),
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hi",
    )

    kernel.submit_interaction.assert_awaited_once()
