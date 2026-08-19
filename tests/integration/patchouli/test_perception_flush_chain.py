"""
Perception 感知链路集成测试。

驱动真实组件协作：
    PerceptionFamiliar + SemanticFlowPerceptionLayer + TriggerManager
    + ShortTermMemoryStore + InMemoryInteractionApplyJournal + MemoryLibrary
仅 relay（LLM 摘要）与 bus（下游生成端口）为 fake。

覆盖：
- IDLE 超时结算并驱逐话题
- IDLE 空话题跳过 settlement 提交
- IDLE 释放容量后新话题可入池
- SHUTDOWN 全量结算 + 驱逐
- folding 后的 shutdown 结算保留 state_summary 与 retained block
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from hivememory.core.models import Identity, TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushReason
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_access_context


def _make_identity(user="u1", agent="a1"):
    return Identity(user_id=user, agent_id=agent)


def _make_payload(user_msg="hello", assistant_msg="world", identity=None):
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
        access_context=make_access_context(actor_identity=identity),
    )


def _make_real_familiar(
    *,
    idle_timeout_seconds=1,
    max_resident_topics=5,
    fold_token_threshold=999999,
    fold_retain_recent_blocks=2,
):
    store = ShortTermMemoryStore(max_resident_topics=max_resident_topics)
    relay = Mock()
    relay.should_relay.return_value = None
    relay.generate_summary.side_effect = (
        lambda blocks_to_fold, previous_summary: (
            f"{previous_summary}|folded:{len(blocks_to_fold)}"
            if previous_summary
            else f"folded:{len(blocks_to_fold)}"
        )
    )
    interaction_journal = InMemoryInteractionApplyJournal()
    layer = SemanticFlowPerceptionLayer(
        config=SemanticFlowPerceptionConfig(
            fold_token_threshold=fold_token_threshold,
            fold_retain_recent_blocks=fold_retain_recent_blocks,
        ),
        relay_controller=relay,
        short_term_store=store,
        interaction_journal=interaction_journal,
    )
    bus = Mock()
    bus.request = AsyncMock(return_value=None)
    library = MemoryLibrary(short_term=store, mid_term=Mock(), long_term=Mock())
    familiar = PerceptionFamiliar(
        perception_layer=layer,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=idle_timeout_seconds),
        memory_library=library,
        interaction_journal=interaction_journal,
    )
    return familiar, layer, store, bus


def _fast_forward_idle():
    """把模型层时间源前移，使已驻留话题进入 idle 判定（替代固定 sleep）。"""
    from datetime import datetime, timezone

    real_now = time.time()

    class _ShiftedDatetime(datetime):
        @classmethod
        def now(cls, tz=timezone.utc):
            return datetime.fromtimestamp(real_now + 100, tz)

    return patch(
        "hivememory.core.models.topic.datetime",
        _ShiftedDatetime,
    )


@pytest.mark.asyncio
async def test_idle_flush_swaps_out_topic():
    familiar, _, store, bus = _make_real_familiar(idle_timeout_seconds=1)
    await familiar.submit_interaction(_make_payload("question", "answer"), "NEW_TOPIC")
    assert len(store.list_topic_data()) == 1

    with _fast_forward_idle():
        flushed = await familiar.scan_idle_buffers_once()

    assert len(flushed) == 1
    assert store.list_topic_data() == []
    route, settlement = bus.request.await_args.args
    assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    assert settlement.reason == FlushReason.IDLE_TIMEOUT


@pytest.mark.asyncio
async def test_idle_flush_skips_empty_settlement_submission():
    familiar, layer, store, bus = _make_real_familiar(idle_timeout_seconds=1)
    topic_id = await layer.create_new_topic(_make_identity())
    assert store.get_topic_data(topic_id) is not None

    with _fast_forward_idle():
        flushed = await familiar.scan_idle_buffers_once()

    assert flushed == [topic_id]
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_flush_frees_slot():
    familiar, _, store, _ = _make_real_familiar(
        idle_timeout_seconds=1,
        max_resident_topics=2,
    )
    await familiar.submit_interaction(_make_payload("q1", "a1", _make_identity("u1", "a1")), "NEW_TOPIC")
    await familiar.submit_interaction(_make_payload("q2", "a2", _make_identity("u2", "a2")), "NEW_TOPIC")
    assert len(store.list_topic_data()) == 2

    with _fast_forward_idle():
        assert len(await familiar.scan_idle_buffers_once()) == 2

    await familiar.submit_interaction(_make_payload("q3", "a3", _make_identity("u3", "a3")), "NEW_TOPIC")
    assert len(store.list_topic_data()) == 1


@pytest.mark.asyncio
async def test_shutdown_flush_archives_and_swaps_out_all_topics():
    familiar, _, store, bus = _make_real_familiar(max_resident_topics=4)
    await familiar.submit_interaction(_make_payload("q1", "a1", _make_identity("u1", "a1")), "NEW_TOPIC")
    await familiar.submit_interaction(_make_payload("q2", "a2", _make_identity("u2", "a2")), "NEW_TOPIC")

    result = await familiar.flush_all_for_shutdown()

    assert result.trigger_reason == FlushReason.SHUTDOWN.value
    assert len(result.flushed_topics) == 2
    assert result.archived_blocks == 2
    assert store.list_topic_data() == []
    assert bus.request.await_count == 2
    for call in bus.request.await_args_list:
        assert call.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert call.args[1].reason == FlushReason.SHUTDOWN


@pytest.mark.asyncio
async def test_shutdown_after_folding_settles_summary_and_retained_block():
    familiar, _, store, bus = _make_real_familiar(
        max_resident_topics=2,
        fold_token_threshold=10,
        fold_retain_recent_blocks=1,
    )
    identity = _make_identity("u-fold", "a-fold")
    topic_id = "NEW_TOPIC"

    for i in range(3):
        topic_id = await familiar.submit_interaction(
            _make_payload(
                f"question-{i}-" * 80,
                f"answer-{i}",
                identity,
            ),
            topic_id,
        )

    before_shutdown = store.get_topic_data(topic_id, touch=False)
    assert before_shutdown is not None
    assert before_shutdown.state_summary == "folded:1|folded:1"
    assert [block.user_query for block in before_shutdown.blocks] == [
        "question-2-" * 80
    ]
    assert bus.request.await_count == 0

    result = await familiar.flush_all_for_shutdown()

    assert result.flushed_topics == [topic_id]
    assert result.archived_blocks == 1
    assert store.get_topic_data(topic_id, touch=False) is None
    bus.request.assert_awaited_once()
    route, settlement = bus.request.await_args.args
    assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    assert settlement.reason == FlushReason.SHUTDOWN
    assert settlement.state_summary == "folded:1|folded:1"
    assert [block.user_query for block in settlement.blocks] == [
        "question-2-" * 80
    ]
