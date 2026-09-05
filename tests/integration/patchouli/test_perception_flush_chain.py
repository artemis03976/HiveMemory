"""
Perception 感知链路集成测试。

驱动真实组件协作：
    PerceptionFamiliar + MemoryPerceptionEngine + TopicWorkingSet
    + ShortTermMemoryStore + InMemoryInteractionApplyJournal
仅 relay（LLM 摘要）与 bus（下游生成端口）为 fake。

覆盖：
- IDLE 超时结算并驱逐话题
- IDLE 空话题跳过 settlement 提交但仍正常结束生命周期
- IDLE 扫描跳过被占用（lease）的话题
- IDLE 释放容量后新话题可入池
- LRU 候选被占用时改选其他话题
- 跨 Workspace 复用同一 interaction_id 被拒绝
- SHUTDOWN 全量结算 + 驱逐（含真正空 Topic）
- SHUTDOWN busy / admission 异常隔离到 failed_topic_ids
- folding 后的 shutdown 结算保留 state_summary 与 retained block
- summary-only Topic 的列表与 discard 语义
- token 溢出 compact 折叠旧前缀、不触发记忆生成、不驱逐
- manual settle / manual delete 三个互不混杂的用例
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import ActorIdentity, LogicalBlock, TurnEvent, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TriggerReason
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.errors import TopicSettleAdmissionError
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


class _FakeClock:
    """可控单调时钟：由测试手动推进，替代真实时间。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_identity(user="u1", agent="a1"):
    return ActorIdentity(user_id=user, agent_id=agent)


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
    """组装真实感知链路：Engine + Store + WorkingSet + Journal 全真实。"""
    store = ShortTermMemoryStore()
    clock = _FakeClock()
    relay = Mock()
    relay.generate_summary.side_effect = (
        lambda blocks_to_fold, previous_summary=None: (
            f"{previous_summary}|folded:{len(blocks_to_fold)}"
            if previous_summary
            else f"folded:{len(blocks_to_fold)}"
        )
    )
    interaction_journal = InMemoryInteractionApplyJournal()
    engine = MemoryPerceptionEngine(
        config=SemanticFlowPerceptionConfig(
            fold_token_threshold=fold_token_threshold,
            fold_retain_recent_blocks=fold_retain_recent_blocks,
        ),
        relay_controller=relay,
    )
    working_set = TopicWorkingSet(max_resident=max_resident_topics, clock=clock)
    bus = Mock()
    bus.request = AsyncMock(return_value=None)
    familiar = PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=working_set,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=idle_timeout_seconds),
        interaction_journal=interaction_journal,
    )
    return familiar, store, working_set, clock, bus


def _accept_settlement_task(route, payload):
    """在 generation 边界返回确定性的已接纳任务快照。"""

    return MemoryGenerationTask(
        task_id=f"memtask-{payload.topic_id}",
        topic_id=payload.topic_id,
        label=payload.topic_id,
        source=MemoryGenerationSource.SETTLE,
    )


@pytest.mark.asyncio
async def test_idle_flush_swaps_out_topic():
    familiar, store, _, clock, bus = _make_real_familiar(idle_timeout_seconds=1)
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    await familiar.apply_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    assert len(store.list_by_workspace(identity_scope)) == 1

    clock.advance(100)
    flushed = await familiar.scan_idle_buffers_once()

    assert len(flushed) == 1
    assert store.list_by_workspace(identity_scope) == []
    route, settlement = bus.request.await_args.args
    assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    assert settlement.reason == TriggerReason.IDLE_TIMEOUT


@pytest.mark.asyncio
async def test_idle_flush_skips_empty_settlement_submission():
    familiar, store, _, clock, bus = _make_real_familiar(idle_timeout_seconds=1)
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    assert store.get(identity_scope, topic_id) is not None

    clock.advance(100)
    flushed = await familiar.scan_idle_buffers_once()

    assert flushed == [topic_id]
    # 真正空 Topic：不提交 settlement，但生命周期正常结束
    bus.request.assert_not_awaited()
    assert store.get(identity_scope, topic_id) is None


@pytest.mark.asyncio
async def test_idle_scan_skips_topic_that_becomes_busy_before_settlement():
    """维护快照之后被占用（lease）的 Topic 留给下一轮扫描。"""
    familiar, store, working_set, clock, bus = _make_real_familiar(
        idle_timeout_seconds=1
    )
    identity_scope = make_identity_scope(user_id="u-busy", agent_id="a-busy")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    assert working_set.acquire(identity_scope, topic_id) is not None

    clock.advance(100)
    flushed = await familiar.scan_idle_buffers_once()

    assert flushed == []
    remaining = store.get(identity_scope, topic_id)
    assert remaining is not None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_flush_frees_slot():
    familiar, store, _, clock, _ = _make_real_familiar(
        idle_timeout_seconds=1,
        max_resident_topics=2,
    )
    await familiar.apply_interaction(
        _make_payload("q1", "a1"),
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        target_topic_id="NEW_TOPIC",
    )
    await familiar.apply_interaction(
        _make_payload("q2", "a2"),
        identity_scope=make_identity_scope(user_id="u2", agent_id="a2"),
        target_topic_id="NEW_TOPIC",
    )
    assert len(store.list_by_workspace(make_identity_scope(user_id="u1", agent_id="a1"))) == 1
    assert len(store.list_by_workspace(make_identity_scope(user_id="u2", agent_id="a2"))) == 1

    clock.advance(100)
    assert len(await familiar.scan_idle_buffers_once()) == 2

    await familiar.apply_interaction(
        _make_payload("q3", "a3"),
        identity_scope=make_identity_scope(user_id="u3", agent_id="a3"),
        target_topic_id="NEW_TOPIC",
    )
    assert len(store.list_by_workspace(make_identity_scope(user_id="u3", agent_id="a3"))) == 1


@pytest.mark.asyncio
async def test_lru_reselects_another_topic_when_first_candidate_becomes_busy():
    """LRU 候选被占用时改选其他话题，不能导致超额创建或误报驱逐成功。"""
    familiar, store, working_set, clock, bus = _make_real_familiar(
        max_resident_topics=2
    )
    identity_scope = make_identity_scope(user_id="u-lru", agent_id="a-lru")
    first_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    clock.advance(10)
    second_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    clock.advance(10)
    # 占住最久未访问的候选，迫使驱逐改选
    lease = working_set.acquire(identity_scope, first_id)
    assert lease is not None

    new_id = await familiar.apply_interaction(
        _make_payload("new question", "new answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    resident = store.list_by_workspace(identity_scope)
    assert {topic.topic_id for topic in resident} == {first_id, new_id}
    assert store.get(identity_scope, second_id) is None
    # 空话题结算不产生材料，不触发生成 admission
    bus.request.assert_not_awaited()
    working_set.release(lease)


@pytest.mark.asyncio
async def test_interaction_retry_rejects_same_content_from_another_workspace():
    """同一 interaction_id 不能跨 Workspace 复用已完成的 apply 记录。"""
    familiar, store, _, _, _ = _make_real_familiar()
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

    topic_id = await familiar.apply_interaction(
        payload,
        identity_scope=main_scope,
        target_topic_id="NEW_TOPIC",
        interaction_id="interaction-shared",
    )

    with pytest.raises(ValueError, match="different input"):
        await familiar.apply_interaction(
            payload,
            identity_scope=isolated_scope,
            target_topic_id="NEW_TOPIC",
            interaction_id="interaction-shared",
        )

    original = store.get(main_scope, topic_id)
    assert original is not None
    assert [block.user_query for block in original.blocks] == ["same question"]
    assert store.list_by_workspace(isolated_scope) == []


@pytest.mark.asyncio
async def test_shutdown_flush_settles_and_swaps_out_all_topics():
    familiar, store, _, _, bus = _make_real_familiar(max_resident_topics=4)
    await familiar.apply_interaction(
        _make_payload("q1", "a1"),
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        target_topic_id="NEW_TOPIC",
    )
    await familiar.apply_interaction(
        _make_payload("q2", "a2"),
        identity_scope=make_identity_scope(user_id="u2", agent_id="a2"),
        target_topic_id="NEW_TOPIC",
    )
    bus.request.side_effect = _accept_settlement_task

    result = await familiar.flush_all_for_shutdown()

    assert len(result.settled_topic_ids) == 2
    assert result.generation_skipped_topic_ids == ()
    assert result.resident_block_count == 2
    assert store.list_by_workspace(make_identity_scope(user_id="u1", agent_id="a1")) == []
    assert store.list_by_workspace(make_identity_scope(user_id="u2", agent_id="a2")) == []
    assert bus.request.await_count == 2
    for call in bus.request.await_args_list:
        assert call.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert call.args[1].reason == TriggerReason.SHUTDOWN


@pytest.mark.asyncio
async def test_shutdown_reports_busy_topic_in_failed_ids_without_generation():
    """drain 后仍被占用的话题属于关闭顺序缺陷，隔离记录为 failed。"""
    familiar, store, working_set, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u-busy", agent_id="a-busy")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    assert working_set.acquire(identity_scope, topic_id) is not None

    result = await familiar.flush_all_for_shutdown()

    assert result.failed_topic_ids == (topic_id,)
    assert result.settled_topic_ids == ()
    remaining = store.get(identity_scope, topic_id)
    assert remaining is not None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_after_folding_settles_summary_and_retained_block():
    familiar, store, _, _, bus = _make_real_familiar(
        max_resident_topics=2,
        fold_token_threshold=10,
        fold_retain_recent_blocks=1,
    )
    identity_scope = make_identity_scope(user_id="u-fold", agent_id="a-fold")
    topic_id = "NEW_TOPIC"

    for i in range(3):
        topic_id = await familiar.apply_interaction(
            _make_payload(
                f"question-{i}-" * 80,
                f"answer-{i}",
            ),
            identity_scope=identity_scope,
            target_topic_id=topic_id,
        )

    before_shutdown = store.get(identity_scope, topic_id)
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
    assert store.get(identity_scope, topic_id) is None
    bus.request.assert_awaited_once()
    route, settlement = bus.request.await_args.args
    assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
    assert settlement.reason == TriggerReason.SHUTDOWN
    assert settlement.state_summary == "folded:1|folded:1"
    assert [block.user_query for block in settlement.blocks] == [
        "question-2-" * 80
    ]


# ========== summary-only 内容判空语义 ==========

@pytest.mark.asyncio
async def test_summary_only_topic_stays_in_non_empty_active_list():
    familiar, store, _, _, _ = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    topic = store.get(identity_scope, topic_id)
    store.put(topic.model_copy(update={"state_summary": "已经折叠的历史内容"}))

    topics = store.list_by_workspace(identity_scope, include_empty=False)

    assert [t.topic_id for t in topics] == [topic_id]
    assert topics[0].blocks == ()
    assert topics[0].is_empty is False


@pytest.mark.asyncio
async def test_discard_if_empty_keeps_summary_only_topic():
    familiar, store, _, _, _ = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    topic = store.get(identity_scope, topic_id)
    store.put(topic.model_copy(update={"state_summary": "折叠历史"}))

    discarded = familiar.discard_if_empty(identity_scope, topic_id)

    assert discarded is False
    assert store.get(identity_scope, topic_id) is not None


@pytest.mark.asyncio
async def test_discard_if_empty_evicts_truly_empty_topic():
    familiar, store, _, _, _ = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)

    discarded = familiar.discard_if_empty(identity_scope, topic_id)

    assert discarded is True
    assert store.get(identity_scope, topic_id) is None


# ========== 真正空 Topic 仍正常结束生命周期 ==========

@pytest.mark.asyncio
async def test_shutdown_evicts_truly_empty_topic_and_marks_generation_skip():
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)

    result = await familiar.flush_all_for_shutdown()

    assert topic_id in result.settled_topic_ids
    assert result.generation_skipped_topic_ids == (topic_id,)
    assert store.get(identity_scope, topic_id) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_marks_generation_skip_when_submission_builds_no_task():
    """下游正常返回无任务时，Topic 已结算但 generation 被跳过。"""
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    result = await familiar.flush_all_for_shutdown()

    assert result.settled_topic_ids == (topic_id,)
    assert result.generation_skipped_topic_ids == (topic_id,)
    assert store.get(identity_scope, topic_id) is None


@pytest.mark.asyncio
async def test_shutdown_isolates_admission_failure_in_failed_ids():
    """generation admission 异常被隔离到 failed_topic_ids，话题内容保留。"""
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    bus.request.side_effect = RuntimeError("generation admission failed")

    result = await familiar.flush_all_for_shutdown()

    assert result.failed_topic_ids == (topic_id,)
    assert result.settled_topic_ids == ()
    # admission 失败：话题内容保留，可重试
    assert store.get(identity_scope, topic_id) is not None


@pytest.mark.asyncio
async def test_shutdown_marks_filtered_topic_as_generation_skip_and_evicts_it():
    """全部 blocks 被过滤时不能因为 block_count 非零而漏报 skip。"""
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    topic = store.get(identity_scope, topic_id)
    store.put(
        topic.model_copy(
            update={
                "blocks": (
                    LogicalBlock(
                        turn=TurnRecord(user_query="q", assistant_final_text="a"),
                        worth_saving=False,
                    ),
                )
            }
        )
    )

    result = await familiar.flush_all_for_shutdown()

    assert result.settled_topic_ids == (topic_id,)
    assert result.generation_skipped_topic_ids == (topic_id,)
    assert result.resident_block_count == 1
    assert store.get(identity_scope, topic_id) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_settle_evicts_truly_empty_topic():
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)

    result = await familiar.manual_settle_topic(identity_scope, topic_id)

    assert result.topic_id == topic_id
    assert result.generation_submitted is False
    assert result.generation_task_id is None
    assert store.get(identity_scope, topic_id) is None
    bus.request.assert_not_awaited()


# ========== manual settle: admission -> evict 顺序 ==========

@pytest.mark.asyncio
async def test_manual_settle_admits_generation_task_before_evicting():
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
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
    assert store.get(identity_scope, topic_id) is None


@pytest.mark.asyncio
async def test_manual_settle_admission_failure_keeps_topic_intact_and_allows_retry():
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    before = store.get(identity_scope, topic_id)
    bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

    with pytest.raises(TopicSettleAdmissionError, match="可重试"):
        await familiar.manual_settle_topic(identity_scope, topic_id)

    after = store.get(identity_scope, topic_id)
    assert after is not None
    assert [b.user_query for b in after.blocks] == [b.user_query for b in before.blocks]
    assert after.state_summary == before.state_summary

    # 修复 admission 后可再次重试并成功结束生命周期
    bus.request = AsyncMock(return_value=None)
    result = await familiar.manual_settle_topic(identity_scope, topic_id)
    assert result.topic_id == topic_id
    assert store.get(identity_scope, topic_id) is None


@pytest.mark.asyncio
async def test_manual_settle_with_all_blocks_filtered_evicts_without_task():
    """blocks 均被 worth_saving=False 过滤：无任务但生命周期仍正常结束。"""
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, identity_scope)
    topic = store.get(identity_scope, topic_id)
    store.put(
        topic.model_copy(
            update={
                "blocks": (
                    LogicalBlock(
                        turn=TurnRecord(user_query="q", assistant_final_text="a"),
                        worth_saving=False,
                    ),
                )
            }
        )
    )

    result = await familiar.manual_settle_topic(identity_scope, topic_id)

    assert result.topic_id == topic_id
    assert result.generation_submitted is False
    assert store.get(identity_scope, topic_id) is None
    bus.request.assert_not_awaited()


# ========== manual delete: 只驱逐，不写记忆 ==========

@pytest.mark.asyncio
async def test_manual_delete_evicts_without_generation_task():
    familiar, store, _, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    removed = await familiar.evict_topic(identity_scope, topic_id)

    assert removed.topic_id == topic_id
    assert removed.removed is True
    assert store.get(identity_scope, topic_id) is None
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_delete_reports_busy_topic_without_removing_it():
    """话题正被占用时，手动删除报告 removed=False 并保留它。"""
    familiar, store, working_set, _, bus = _make_real_familiar()
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _make_payload("question", "answer"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )
    lease = working_set.acquire(identity_scope, topic_id)
    assert lease is not None

    result = await familiar.evict_topic(identity_scope, topic_id)

    assert result.removed is False
    assert store.get(identity_scope, topic_id) is not None
    bus.request.assert_not_awaited()
    working_set.release(lease)


# ========== token 溢出 compact：只压缩工作集 ==========

@pytest.mark.asyncio
async def test_token_overflow_compact_trims_prefix_and_keeps_topic():
    familiar, store, _, _, bus = _make_real_familiar(
        fold_token_threshold=1, fold_retain_recent_blocks=1
    )
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = "NEW_TOPIC"
    for i in range(3):
        topic_id = await familiar.apply_interaction(
            _make_payload(f"question-{i}-" * 80, f"answer-{i}"),
            identity_scope=identity_scope,
            target_topic_id=topic_id,
        )

    data = store.get(identity_scope, topic_id)
    assert data is not None
    assert data.state_summary == "folded:1|folded:1"
    assert [b.user_query for b in data.blocks] == ["question-2-" * 80]
    # compact 不触发记忆生成、不驱逐
    bus.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_compact_is_noop_when_blocks_not_exceeding_retain():
    familiar, store, _, _, _ = _make_real_familiar(
        fold_token_threshold=10, fold_retain_recent_blocks=5
    )
    identity_scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _make_payload("q", "a"),
        identity_scope=identity_scope,
        target_topic_id="NEW_TOPIC",
    )

    data = store.get(identity_scope, topic_id)
    assert data.state_summary == ""
    assert len(data.blocks) == 1
