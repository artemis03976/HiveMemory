"""
Perception 感知链路集成测试。

驱动真实组件协作：
    PerceptionFamiliar + SemanticFlowPerceptionLayer + TriggerManager
    + ShortTermMemoryStore + InMemoryInteractionApplyJournal + MemoryLibrary
仅 relay（LLM 摘要）与 bus（下游生成端口）为 fake。

覆盖：
- IDLE 超时结算并驱逐话题
- IDLE 空话题跳过 settlement 提交但仍按矩阵 evict
- IDLE 释放容量后新话题可入池
- SHUTDOWN 全量结算 + 驱逐（含真正空 Topic）
- SHUTDOWN 正常 generation skip 与 admission 异常传播
- folding 后的 shutdown 结算保留 state_summary 与 retained block
- summary-only Topic 的列表、路由与 discard 语义
- manual settle / compact / delete 三个互不混杂的用例
- manual settle prepare -> admission -> evict 顺序与失败可重试
- compact 路径 retain_recent_blocks >= 1 的输入边界
"""

import time
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from hivememory.core.models import Identity, LogicalBlock, TurnEvent, TurnRecord, WorkspaceTopicKey
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushEvent, FlushReason
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


def _make_identity(user="u1", agent="a1"):
    return Identity(user_id=user, agent_id=agent)


def _make_payload(user_msg="hello", assistant_msg="world"):
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


def _accept_settlement_task(route, payload):
    """在 generation 边界返回确定性的已接纳任务快照。"""

    return MemoryGenerationTask(
        task_id=f"memtask-{payload.topic_id}",
        topic_id=payload.topic_id,
        label=payload.topic_id,
        source=MemoryGenerationSource.SETTLE,
    )


def _fast_forward_idle():
    """把模型层时间源前移，使已驻留话题进入 idle 判定（替代固定 sleep）。"""
    from datetime import datetime

    real_now = time.time()

    class _ShiftedDatetime(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return datetime.fromtimestamp(real_now + 100, tz)

    return patch(
        "hivememory.core.models.topic.datetime",
        _ShiftedDatetime,
    )


@pytest.mark.asyncio
async def test_idle_flush_swaps_out_topic():
    familiar, _, store, bus = _make_real_familiar(idle_timeout_seconds=1)
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    await familiar.submit_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    assert len(store.list_topic_data(identity_scope)) == 1

    with _fast_forward_idle():
        flushed = await familiar.scan_idle_buffers_once()

    assert len(flushed) == 1
    assert store.list_topic_data(identity_scope) == []
    route, settlement = bus.request.await_args.args
    assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    assert settlement.reason == FlushReason.IDLE_TIMEOUT


@pytest.mark.asyncio
async def test_idle_flush_skips_empty_settlement_submission():
    familiar, layer, store, bus = _make_real_familiar(idle_timeout_seconds=1)
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)
    assert store.get_topic_data(identity_scope, topic_id) is not None

    with _fast_forward_idle():
        flushed = await familiar.scan_idle_buffers_once()

    assert flushed == [topic_id]
    # 真正空 Topic：不提交 settlement，但按 IDLE 矩阵 evict
    bus.request.assert_not_awaited()
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None


@pytest.mark.asyncio
async def test_idle_scan_skips_topic_that_becomes_busy_before_settlement():
    """维护快照之后进入 PROCESSING 的 Topic 留给下一轮扫描。"""
    familiar, layer, store, bus = _make_real_familiar(idle_timeout_seconds=1)
    identity_scope = make_identity_scope(user_id="u-busy", agent_id="a-busy")
    topic_id = await layer.create_new_topic(identity_scope)
    topic_key = WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)
    assert store.reserve_processing(topic_key) is True

    with _fast_forward_idle():
        flushed = await familiar.scan_idle_buffers_once()

    assert flushed == []
    remaining = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert remaining is not None
    assert remaining.state.value == "processing"
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_flush_frees_slot():
    familiar, _, store, _ = _make_real_familiar(
        idle_timeout_seconds=1,
        max_resident_topics=2,
    )
    await familiar.submit_interaction(
        _make_payload("q1", "a1"),
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        target_topic_id="NEW_TOPIC",
    )
    await familiar.submit_interaction(
        _make_payload("q2", "a2"),
        identity_scope=make_identity_scope(user_id="u2", agent_id="a2"),
        target_topic_id="NEW_TOPIC",
    )
    assert len(store.list_topic_data(make_identity_scope(user_id="u1", agent_id="a1"))) == 1
    assert len(store.list_topic_data(make_identity_scope(user_id="u2", agent_id="a2"))) == 1

    with _fast_forward_idle():
        assert len(await familiar.scan_idle_buffers_once()) == 2

    await familiar.submit_interaction(
        _make_payload("q3", "a3"),
        identity_scope=make_identity_scope(user_id="u3", agent_id="a3"),
        target_topic_id="NEW_TOPIC",
    )
    assert len(store.list_topic_data(make_identity_scope(user_id="u3", agent_id="a3"))) == 1


@pytest.mark.asyncio
async def test_lru_reselects_another_idle_topic_when_first_candidate_becomes_busy():
    """LRU 候选选择后的状态竞态不能导致超额创建或误报驱逐成功。"""
    familiar, layer, store, bus = _make_real_familiar(max_resident_topics=2)
    identity_scope = make_identity_scope(user_id="u-lru", agent_id="a-lru")
    first_id = await layer.create_new_topic(identity_scope)
    second_id = await layer.create_new_topic(identity_scope)
    first_candidate_id = store.get_lru_topic(identity_scope)
    assert first_candidate_id in {first_id, second_id}
    fallback_id = second_id if first_candidate_id == first_id else first_id

    original_settle = layer.settle_topic
    first_candidate_key = WorkspaceTopicKey.from_identity_scope(
        identity_scope,
        first_candidate_id,
    )

    async def settle_with_candidate_race(topic_key, reason):
        if topic_key == first_candidate_key:
            assert store.reserve_processing(topic_key) is True
            return await original_settle(topic_key, reason)

        result = await original_settle(topic_key, reason)
        store.release_processing(first_candidate_key)
        return result

    layer.settle_topic = settle_with_candidate_race

    new_id = await familiar.submit_interaction(
        _make_payload("new question", "new answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    resident = store.list_topic_data(identity_scope)
    assert {topic.topic_id for topic in resident} == {first_candidate_id, new_id}
    assert all(topic.state.value == "idle" for topic in resident)
    assert store.get_topic_data(identity_scope, fallback_id, touch=False) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_interaction_retry_rejects_same_content_from_another_workspace():
    """同一 interaction_id 不能跨 Workspace 复用已完成的 apply 记录。"""
    familiar, _, store, _ = _make_real_familiar()
    main_scope = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="main_workspace",
    )
    isolated_scope = make_identity_scope(
        user_id="u1",
        agent_id="a1",
        workspace_id="isolated_workspace",
    )
    payload = _make_payload("same question", "same answer")

    topic_id = await familiar.submit_interaction(
        payload,
        identity_scope=main_scope,
        target_topic_id="NEW_TOPIC",
        interaction_id="interaction-shared",
    )

    with pytest.raises(ValueError, match="different input"):
        await familiar.submit_interaction(
            payload,
            identity_scope=isolated_scope,
            target_topic_id="NEW_TOPIC",
            interaction_id="interaction-shared",
        )

    original = store.get_topic_data(main_scope, topic_id, touch=False)
    assert original is not None
    assert [block.user_query for block in original.blocks] == ["same question"]
    assert store.list_topic_data(isolated_scope) == []


@pytest.mark.asyncio
async def test_shutdown_flush_settles_and_swaps_out_all_topics():
    familiar, _, store, bus = _make_real_familiar(max_resident_topics=4)
    await familiar.submit_interaction(
        _make_payload("q1", "a1"),
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        target_topic_id="NEW_TOPIC",
    )
    await familiar.submit_interaction(
        _make_payload("q2", "a2"),
        identity_scope=make_identity_scope(user_id="u2", agent_id="a2"),
        target_topic_id="NEW_TOPIC",
    )
    bus.request.side_effect = _accept_settlement_task

    result = await familiar.flush_all_for_shutdown()

    assert len(result.settled_topic_ids) == 2
    assert result.generation_skipped_topic_ids == ()
    assert result.resident_block_count == 2
    assert store.list_topic_data(make_identity_scope(user_id="u1", agent_id="a1")) == []
    assert store.list_topic_data(make_identity_scope(user_id="u2", agent_id="a2")) == []
    assert bus.request.await_count == 2
    for call in bus.request.await_args_list:
        assert call.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert call.args[1].reason == FlushReason.SHUTDOWN


@pytest.mark.asyncio
async def test_shutdown_reports_busy_topic_instead_of_marking_generation_skip():
    """drain 后仍 busy 属于关闭顺序缺陷，不能计入 settled 或正常 skip。"""
    familiar, _, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u-busy", agent_id="a-busy")
    topic = store.create_buffer(identity_scope)
    assert store.reserve_processing(topic.topic_key) is True

    with pytest.raises(TopicBusyError, match="正忙"):
        await familiar.flush_all_for_shutdown()

    remaining = store.get_topic_data(identity_scope, topic.topic_id, touch=False)
    assert remaining is not None
    assert remaining.state.value == "processing"
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_after_folding_settles_summary_and_retained_block():
    familiar, _, store, bus = _make_real_familiar(
        max_resident_topics=2,
        fold_token_threshold=10,
        fold_retain_recent_blocks=1,
    )
    identity = _make_identity("u-fold", "a-fold")
    identity_scope = make_identity_scope(actor_identity=identity)
    topic_id = "NEW_TOPIC"

    for i in range(3):
        topic_id = await familiar.submit_interaction(
            _make_payload(
                f"question-{i}-" * 80,
                f"answer-{i}",
            ),
            identity_scope=identity_scope,
            target_topic_id=topic_id,
        )

    identity_scope = make_identity_scope(actor_identity=identity)
    before_shutdown = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert before_shutdown is not None
    assert before_shutdown.state_summary == "folded:1|folded:1"
    assert [block.user_query for block in before_shutdown.blocks] == [
        "question-2-" * 80
    ]
    assert bus.request.await_count == 0
    bus.request.side_effect = _accept_settlement_task

    result = await familiar.flush_all_for_shutdown()

    assert result.settled_topic_ids == (topic_id,)
    assert result.generation_skipped_topic_ids == ()
    assert result.resident_block_count == 1
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None
    bus.request.assert_awaited_once()
    route, settlement = bus.request.await_args.args
    assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    assert settlement.reason == FlushReason.SHUTDOWN
    assert settlement.state_summary == "folded:1|folded:1"
    assert [block.user_query for block in settlement.blocks] == [
        "question-2-" * 80
    ]


# ========== summary-only 内容判空语义 ==========

@pytest.mark.asyncio
async def test_summary_only_topic_stays_in_non_empty_active_list():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)
    store.update_summary(
        WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id),
        "已经折叠的历史内容",
    )

    topics = store.list_topic_data(identity_scope, include_empty=False)

    assert [t.topic_id for t in topics] == [topic_id]
    assert topics[0].blocks == ()
    assert topics[0].is_empty is False


@pytest.mark.asyncio
async def test_discard_if_empty_keeps_summary_only_topic():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)
    store.update_summary(
        WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id),
        "折叠历史",
    )

    discarded = layer.discard_if_empty(identity_scope, topic_id)

    assert discarded is False
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is not None


@pytest.mark.asyncio
async def test_discard_if_empty_evicts_truly_empty_topic():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)

    discarded = layer.discard_if_empty(identity_scope, topic_id)

    assert discarded is True
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None


# ========== 真正空 Topic 仍按矩阵 evict ==========

@pytest.mark.asyncio
async def test_shutdown_evicts_truly_empty_topic_and_marks_generation_skip():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)

    result = await familiar.flush_all_for_shutdown()

    assert topic_id in result.settled_topic_ids
    assert result.generation_skipped_topic_ids == (topic_id,)
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_marks_generation_skip_when_submission_builds_no_task():
    """下游正常返回无任务时，Topic 已结算但 generation 被跳过。"""
    familiar, _, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.submit_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    result = await familiar.flush_all_for_shutdown()

    assert result.settled_topic_ids == (topic_id,)
    assert result.generation_skipped_topic_ids == (topic_id,)
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None


@pytest.mark.asyncio
async def test_shutdown_propagates_submission_failure_instead_of_marking_skip():
    """generation admission 异常必须向上游传播，不能降级为正常 skip。"""
    familiar, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    await familiar.submit_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    bus.request.side_effect = RuntimeError("generation admission failed")

    with pytest.raises(RuntimeError, match="generation admission failed"):
        await familiar.flush_all_for_shutdown()


@pytest.mark.asyncio
async def test_shutdown_marks_filtered_topic_as_generation_skip_and_evicts_it():
    """全部 blocks 被过滤时不能因为 block_count 非零而漏报 skip。"""
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)
    store.add_block(
        WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id),
        LogicalBlock(
            turn=TurnRecord(user_query="q", assistant_final_text="a"),
            worth_saving=False,
        ),
    )

    result = await familiar.flush_all_for_shutdown()

    assert result.settled_topic_ids == (topic_id,)
    assert result.generation_skipped_topic_ids == (topic_id,)
    assert result.resident_block_count == 1
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_settle_evicts_truly_empty_topic():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)

    result = await familiar.manual_settle_topic(identity_scope, topic_id)

    assert result.topic_id == topic_id
    assert result.generation_submitted is False
    assert result.generation_task_id is None
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None
    bus.request.assert_not_awaited()


# ========== manual settle: prepare -> admission -> evict ==========

@pytest.mark.asyncio
async def test_manual_settle_admits_generation_task_before_evicting():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.submit_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    accepted = MemoryGenerationTask(
        task_id="memtask-1",
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.SETTLE,
    )
    bus.request = AsyncMock(return_value=accepted)

    result = await familiar.manual_settle_topic(identity_scope, topic_id)

    assert result.topic_id == topic_id
    assert result.generation_task_id == "memtask-1"
    assert result.generation_submitted is True
    assert bus.request.await_args.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    # 接纳成功后 Topic 才从池中移除
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None


@pytest.mark.asyncio
async def test_manual_settle_admission_failure_keeps_topic_intact_and_allows_retry():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.submit_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    before = store.get_topic_data(identity_scope, topic_id, touch=False)
    bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

    with pytest.raises(TopicSettleAdmissionError, match="可重试"):
        await familiar.manual_settle_topic(identity_scope, topic_id)

    after = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert after is not None
    assert [b.user_query for b in after.blocks] == [b.user_query for b in before.blocks]
    assert after.state_summary == before.state_summary

    # 修复 admission 后可再次重试并成功结束生命周期
    bus.request = AsyncMock(return_value=None)
    result = await familiar.manual_settle_topic(identity_scope, topic_id)
    assert result.topic_id == topic_id
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None


@pytest.mark.asyncio
async def test_manual_settle_with_all_blocks_filtered_evicts_without_task():
    """blocks 均被 worth_saving=False 过滤：无任务但生命周期仍正常结束。"""
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)
    store.add_block(
        WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id),
        LogicalBlock(
            turn=TurnRecord(user_query="q", assistant_final_text="a"),
            worth_saving=False,
        ),
    )

    result = await familiar.manual_settle_topic(identity_scope, topic_id)

    assert result.topic_id == topic_id
    assert result.generation_submitted is False
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None
    bus.request.assert_not_awaited()


# ========== manual delete: 只驱逐，不写记忆 ==========

@pytest.mark.asyncio
async def test_manual_delete_evicts_without_generation_task():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.submit_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    removed = await familiar.evict_topic(identity_scope, topic_id)

    assert removed.topic_id == topic_id
    assert removed.removed is True
    assert store.get_topic_data(identity_scope, topic_id, touch=False) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reserve_method", "expected_state"),
    [
        ("reserve_processing", "processing"),
        ("reserve_flushing", "flushing"),
    ],
)
async def test_manual_delete_rejects_busy_topic_without_removing_it(
    reserve_method,
    expected_state,
):
    """Topic 持有任一种单写者预约时，手动删除必须报告 busy 并保留它。"""
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await layer.create_new_topic(identity_scope)
    topic_key = WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)
    assert getattr(store, reserve_method)(topic_key) is True

    with pytest.raises(TopicBusyError, match="正忙"):
        await familiar.evict_topic(identity_scope, topic_id)

    remaining = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert remaining is not None
    assert remaining.state.value == expected_state
    bus.request.assert_not_awaited()


# ========== manual compact: 只压缩工作集 ==========

@pytest.mark.asyncio
async def test_manual_compact_updates_summary_trims_prefix_keeps_topic():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = "NEW_TOPIC"
    for i in range(3):
        topic_id = await familiar.submit_interaction(
            _make_payload(f"q{i}", f"a{i}"),
            identity_scope=identity_scope,
            target_topic_id=topic_id,
        )
    topic_key = WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)

    payload = await layer._trigger_manager.resolve_topic(
        FlushEvent(topic_key=topic_key, reason=FlushReason.MANUAL_COMPACT),
        retain_recent_blocks=1,
    )

    assert payload is None
    data = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert data is not None
    assert data.state_summary == "folded:2"
    assert [b.user_query for b in data.blocks] == ["q2"]
    # 同一旧前缀不得同时残留在 summary 与 blocks 中
    assert "q0" not in data.state_summary
    assert "q1" not in data.state_summary
    # compact 不触发记忆生成、不驱逐
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_compact_is_noop_when_blocks_not_exceeding_retain():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.submit_interaction(
        _make_payload("q", "a"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    topic_key = WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)

    payload = await layer._trigger_manager.resolve_topic(
        FlushEvent(topic_key=topic_key, reason=FlushReason.MANUAL_COMPACT),
        retain_recent_blocks=5,
    )

    assert payload is None
    data = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert data.state_summary == ""
    assert len(data.blocks) == 1


# ========== compact 输入边界：retain_recent_blocks >= 1 ==========

@pytest.mark.asyncio
async def test_compact_entries_reject_retain_below_one():
    familiar, layer, store, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.submit_interaction(
        _make_payload("q", "a"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    topic_key = WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)

    for reason in (FlushReason.MANUAL_COMPACT, FlushReason.TOKEN_OVERFLOW):
        for bad in (0, -1):
            with pytest.raises(
                ValueError, match="retain_recent_blocks must be >= 1"
            ):
                await layer._trigger_manager.resolve_topic(
                    FlushEvent(topic_key=topic_key, reason=reason),
                    retain_recent_blocks=bad,
                )
    data = store.get_topic_data(identity_scope, topic_id, touch=False)
    assert data is not None
    assert len(data.blocks) == 1
