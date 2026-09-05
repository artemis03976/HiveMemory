"""PerceptionFamiliar 单元测试（lease 编排架构）。

测试覆盖:
- apply_interaction: 新建/指定话题写入、binding 幂等、busy 拒绝、未知目标、
  retry 幂等与等价性冲突
- LRU 驱逐: 池满驱逐最久未访问话题、busy 候选改选、admission 失败传播
- token 溢出 compact: 折叠摘要写回、relay 失败后 retry 续跑不重复写块
- manual_settle_topic / evict_topic / discard_if_empty 具名用例
- scan_idle_buffers_once / flush_all_for_shutdown 维护投影

真实协作：Engine + Store + WorkingSet + Journal 均为真实对象；
Relay（LLM 摘要）与 Bus（Generation 队列）是边界之外协作者，使用 Mock。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import TurnEvent, WorkspaceAssetRef
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TriggerReason
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
    InteractionApplyStage,
)
from hivememory.patchouli.control.interaction_submission import (
    TransientInteractionSubmissionError,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1", agent_id="a1"):
    return make_identity_scope(user_id=user_id, agent_id=agent_id)


def _payload(message: str = "hello") -> InteractionPayload:
    return InteractionPayload(
        user_message=message,
        assistant_final_text=f"answer:{message}",
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=f"answer:{message}",
            )
        ],
    )


def _generation_task(task_id="task-1", topic_id="t1") -> MemoryGenerationTask:
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.SETTLE,
    )


class _FakeClock:
    """可控单调时钟：由测试手动推进，替代真实时间。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_familiar(
    *,
    max_resident=5,
    idle_timeout=30,
    engine_config=None,
    relay=None,
    bus=None,
    clock=None,
    engine="default",
):
    """构造 (familiar, store, working_set, bus, journal) 测试组合。

    Engine（含 RelayController）/ Store / WorkingSet / Journal 为真实对象；
    时间使用可控时钟。
    """
    clock = clock or _FakeClock()
    store = ShortTermMemoryStore()
    working_set = TopicWorkingSet(max_resident=max_resident, clock=clock)
    if relay is None:
        relay = Mock()
        relay.generate_summary = Mock(return_value="test summary")
    bus = bus or Mock()
    bus.request = AsyncMock(return_value=None)
    if engine == "default":
        engine = MemoryPerceptionEngine(
            config=engine_config or SemanticFlowPerceptionConfig(fold_token_threshold=999999),
            relay_controller=relay,
        )
    journal = InMemoryInteractionApplyJournal()
    familiar = PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=working_set,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=idle_timeout),
        interaction_journal=journal,
    )
    return familiar, store, working_set, bus, journal


# ========== apply_interaction：核心摄入用例 ==========


class TestApplyInteraction:
    @pytest.mark.asyncio
    async def test_new_topic_creates_topic_and_completes_journal(self):
        familiar, store, working_set, bus, journal = _make_familiar()
        scope = _identity_scope()

        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )

        topic = store.get(scope, topic_id)
        assert topic is not None
        assert topic.block_count == 1
        assert topic.topic_title == "新建话题"
        assert journal.get("i-1").stage is InteractionApplyStage.COMPLETED
        # 驻留工作集记录了话题（shutdown 候选可观察）
        assert {tid for _, tid in working_set.list_shutdown_candidates()} == {topic_id}

    @pytest.mark.asyncio
    async def test_existing_topic_appends_blocks_and_dedupes_bindings(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope, topic_id="t-fixed", topic_title="Fixed")
        ref = WorkspaceAssetRef(token="token-1")

        first = await familiar.apply_interaction(
            _payload("a"),
            identity_scope=scope,
            target_topic_id="t-fixed",
            interaction_id="i-1",
            asset_id_and_refs=(("asset-1", ref),),
        )
        second = await familiar.apply_interaction(
            _payload("b"),
            identity_scope=scope,
            target_topic_id="t-fixed",
            interaction_id="i-2",
            asset_id_and_refs=(("asset-1", WorkspaceAssetRef(token="token-1")),),
        )

        assert first == second == "t-fixed"
        topic = store.get(scope, "t-fixed")
        assert topic.block_count == 2
        # 同一资产重复使用只保留首次交互事实
        assert len(topic.bindings) == 1
        assert topic.bindings[0].first_bound_interaction_id == "i-1"

    @pytest.mark.asyncio
    async def test_empty_turn_events_rejected_without_legacy_fallback(self):
        familiar, store, _, _, _ = _make_familiar()
        payload = InteractionPayload(user_message="hello", turn_events=[])

        with pytest.raises(ValueError, match="turn_events is required"):
            await familiar.apply_interaction(
                payload, identity_scope=_identity_scope(), interaction_id="i-1"
            )
        assert store.list_all() == []  # 未产生任何话题写入

    @pytest.mark.asyncio
    async def test_busy_topic_rejects_interaction_without_writing(self):
        familiar, store, working_set, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope, topic_id="t1")
        lease = working_set.acquire(scope, "t1")
        assert lease is not None

        with pytest.raises(TopicBusyError, match="正忙"):
            await familiar.apply_interaction(_payload(), identity_scope=scope, target_topic_id="t1")

        assert store.get(scope, "t1").block_count == 0

    @pytest.mark.asyncio
    async def test_unknown_target_is_rejected(self):
        familiar, _, _, _, _ = _make_familiar()

        with pytest.raises(KeyError, match="does not exist"):
            await familiar.apply_interaction(
                _payload(), identity_scope=_identity_scope(), target_topic_id="missing"
            )

    @pytest.mark.asyncio
    async def test_retry_with_same_payload_is_idempotent(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope, topic_id="t1")
        payload = _payload("m")

        first = await familiar.apply_interaction(
            payload, identity_scope=scope, target_topic_id="t1", interaction_id="i-1"
        )
        replayed = await familiar.apply_interaction(
            payload, identity_scope=scope, target_topic_id="t1", interaction_id="i-1"
        )

        assert replayed == first
        assert store.get(scope, "t1").block_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_different_payload_conflicts(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope, topic_id="t1")

        await familiar.apply_interaction(
            _payload("m"), identity_scope=scope, target_topic_id="t1", interaction_id="i-1"
        )
        with pytest.raises(ValueError, match="different input"):
            await familiar.apply_interaction(
                _payload("other"),
                identity_scope=scope,
                target_topic_id="t1",
                interaction_id="i-1",
            )
        assert store.get(scope, "t1").block_count == 1

    @pytest.mark.asyncio
    async def test_default_manual_settle_targets_last_active_topic(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )

        result = await familiar.manual_settle_topic(scope)

        assert result.topic_id == topic_id
        assert store.get(scope, topic_id) is None


# ========== LRU 驱逐 ==========


class TestLruEviction:
    @pytest.mark.asyncio
    async def test_pool_full_evicts_least_recently_used_topic(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(max_resident=2, clock=clock)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(_payload("1"), identity_scope=scope)
        clock.advance(10)
        t2 = await familiar.apply_interaction(_payload("2"), identity_scope=scope)
        clock.advance(10)

        t3 = await familiar.apply_interaction(_payload("3"), identity_scope=scope)

        assert store.get(scope, t1) is None  # 最久未访问被驱逐
        assert store.get(scope, t2) is not None
        assert store.get(scope, t3) is not None
        # 驱逐走统一 settle 时序并向 Generation 提交材料
        assert bus.request.await_args.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        task = bus.request.await_args.args[1]
        assert task.topic_id == t1
        assert task.reason is TriggerReason.LRU_EVICTION

    @pytest.mark.asyncio
    async def test_busy_candidate_is_skipped_for_older_leased_topic(self):
        clock = _FakeClock()
        familiar, store, working_set, _, _ = _make_familiar(max_resident=2, clock=clock)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(_payload("1"), identity_scope=scope)
        clock.advance(10)
        t2 = await familiar.apply_interaction(_payload("2"), identity_scope=scope)
        clock.advance(10)
        lease = working_set.acquire(scope, t1)
        assert lease is not None

        t3 = await familiar.apply_interaction(_payload("3"), identity_scope=scope)

        # t1 被占用跳过，改选 t2 驱逐
        assert store.get(scope, t1) is not None
        assert store.get(scope, t2) is None
        assert store.get(scope, t3) is not None
        working_set.release(lease)

    @pytest.mark.asyncio
    async def test_admission_failure_preserves_topic_and_propagates(self):
        familiar, store, _, bus, _ = _make_familiar(max_resident=1)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(_payload("1"), identity_scope=scope)
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(RuntimeError, match="admission boom"):
            await familiar.apply_interaction(_payload("2"), identity_scope=scope)

        # admission 失败：话题内容完整保留，可重试
        assert store.get(scope, t1) is not None
        assert store.get(scope, t1).block_count == 1

    @pytest.mark.asyncio
    async def test_pool_full_with_all_candidates_busy_raises(self):
        familiar, _, working_set, _, _ = _make_familiar(max_resident=1)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(_payload("1"), identity_scope=scope)
        working_set.acquire(scope, t1)  # 唯一候选正被占用

        with pytest.raises(TopicBusyError, match="无可占用候选"):
            await familiar.apply_interaction(_payload("2"), identity_scope=scope)


# ========== token 溢出 compact ==========


class TestCompact:
    @pytest.mark.asyncio
    async def test_token_overflow_folds_old_blocks_and_writes_cumulative_summary(self):
        engine_config = SemanticFlowPerceptionConfig(
            fold_token_threshold=1, fold_retain_recent_blocks=1
        )
        mock_relay = Mock()
        # 摘要在旧摘要之上累积（Relay 的 previous_summary 契约）
        mock_relay.generate_summary.side_effect = (
            lambda blocks_to_fold, previous_summary=None: (previous_summary or "") + "---folded"
        )
        familiar, store, _, _, _ = _make_familiar(engine_config=engine_config, relay=mock_relay)
        scope = _identity_scope()
        store.create(scope, topic_id="t1")

        await familiar.apply_interaction(
            _payload("a"), identity_scope=scope, target_topic_id="t1", interaction_id="i-1"
        )
        await familiar.apply_interaction(
            _payload("b"), identity_scope=scope, target_topic_id="t1", interaction_id="i-2"
        )
        await familiar.apply_interaction(
            _payload("c"), identity_scope=scope, target_topic_id="t1", interaction_id="i-3"
        )

        topic = store.get(scope, "t1")
        assert topic.block_count == 1  # 每轮折叠后只保留最近 1 块
        # 第二次折叠收到第一次的摘要（"---folded"）作为 previous_summary，累积成链
        assert topic.state_summary == "---folded---folded"
        assert topic.total_tokens == topic.blocks[0].total_tokens

    @pytest.mark.asyncio
    async def test_compact_failure_after_write_resumes_on_retry(self):
        engine_config = SemanticFlowPerceptionConfig(
            fold_token_threshold=1, fold_retain_recent_blocks=1
        )
        mock_relay = Mock()
        mock_relay.generate_summary.side_effect = [
            TransientInteractionSubmissionError("caller missed apply result"),
            "folded-summary",
        ]
        familiar, store, _, _, journal = _make_familiar(engine_config=engine_config, relay=mock_relay)
        scope = _identity_scope()
        store.create(scope, topic_id="t1")
        payload = _payload("b")

        await familiar.apply_interaction(
            _payload("a"), identity_scope=scope, target_topic_id="t1", interaction_id="i-1"
        )
        with pytest.raises(TransientInteractionSubmissionError):
            await familiar.apply_interaction(
                payload, identity_scope=scope, target_topic_id="t1", interaction_id="i-2"
            )
        # block 已写入、journal 停在 INTERACTION_APPLIED
        assert store.get(scope, "t1").block_count == 2
        assert journal.get("i-2").stage is InteractionApplyStage.INTERACTION_APPLIED

        # 同 payload retry：不重复写块，compact 后置义务补跑完成
        await familiar.apply_interaction(
            payload, identity_scope=scope, target_topic_id="t1", interaction_id="i-2"
        )
        topic = store.get(scope, "t1")
        assert topic.block_count == 1
        assert topic.state_summary == "folded-summary"
        assert journal.get("i-2").stage is InteractionApplyStage.COMPLETED


# ========== manual_settle / evict / discard 具名用例 ==========


class TestNamedUseCases:
    @pytest.mark.asyncio
    async def test_manual_settle_admits_task_then_deletes_topic(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )
        accepted = _generation_task("task-1", topic_id)
        bus.request = AsyncMock(return_value=accepted)

        result = await familiar.manual_settle_topic(scope, topic_id)

        assert result.topic_id == topic_id
        assert result.generation_task_id == "task-1"
        assert result.generation_submitted is True
        assert store.get(scope, topic_id) is None
        assert bus.request.await_args.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT

    @pytest.mark.asyncio
    async def test_manual_settle_empty_topic_skips_generation(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", "标题", "摘要", scope)

        result = await familiar.manual_settle_topic(scope, topic_id)

        assert result.generation_submitted is False
        assert result.generation_task_id is None
        bus.request.assert_not_awaited()
        assert store.get(scope, topic_id) is None

    @pytest.mark.asyncio
    async def test_manual_settle_admission_failure_keeps_topic(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(TopicSettleAdmissionError, match="可重试"):
            await familiar.manual_settle_topic(scope, topic_id)

        assert store.get(scope, topic_id) is not None

    @pytest.mark.asyncio
    async def test_manual_settle_busy_topic_raises_busy_error(self):
        familiar, store, working_set, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )
        working_set.acquire(scope, topic_id)

        with pytest.raises(TopicBusyError):
            await familiar.manual_settle_topic(scope, topic_id)

        assert store.get(scope, topic_id) is not None

    @pytest.mark.asyncio
    async def test_manual_settle_without_topic_or_active_history_raises(self):
        familiar, _, _, _, _ = _make_familiar()

        with pytest.raises(ValueError, match="未指定"):
            await familiar.manual_settle_topic(_identity_scope())

    @pytest.mark.asyncio
    async def test_evict_topic_removes_without_settlement(self):
        familiar, store, working_set, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )

        result = await familiar.evict_topic(scope, topic_id)

        assert result.removed is True
        assert store.get(scope, topic_id) is None
        assert working_set.list_shutdown_candidates() == []
        bus.request.assert_not_awaited()  # evict 不触发结算

    @pytest.mark.asyncio
    async def test_evict_busy_topic_reports_not_removed(self):
        familiar, store, working_set, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )
        working_set.acquire(scope, topic_id)

        result = await familiar.evict_topic(scope, topic_id)

        assert result.removed is False
        assert store.get(scope, topic_id) is not None

    @pytest.mark.asyncio
    async def test_discard_if_empty_removes_only_truly_empty_topic(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        empty_id = await familiar.prepare_topic("NEW_TOPIC", "标题", "摘要", scope)
        content_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )

        assert familiar.discard_if_empty(scope, empty_id) is True
        assert familiar.discard_if_empty(scope, content_id) is False
        assert store.get(scope, empty_id) is None
        assert store.get(scope, content_id) is not None


# ========== 维护与 shutdown ==========


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_scan_idle_settles_only_timed_out_topics(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(idle_timeout=30, clock=clock)
        scope = _identity_scope()
        stale_id = await familiar.apply_interaction(
            _payload("1"), identity_scope=scope, interaction_id="i-1"
        )
        clock.advance(100)
        fresh_id = await familiar.apply_interaction(
            _payload("2"), identity_scope=scope, interaction_id="i-2"
        )

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == [stale_id]
        assert store.get(scope, stale_id) is None
        assert store.get(scope, fresh_id) is not None

    @pytest.mark.asyncio
    async def test_scan_idle_skips_leased_topic(self):
        clock = _FakeClock()
        familiar, store, working_set, _, _ = _make_familiar(idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )
        clock.advance(100)
        working_set.acquire(scope, topic_id)

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == []
        assert store.get(scope, topic_id) is not None

    @pytest.mark.asyncio
    async def test_scan_idle_admission_failure_skips_and_preserves(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )
        clock.advance(100)
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        flushed = await familiar.scan_idle_buffers_once()

        # idle 维护不向上传播；话题保留等待下一轮
        assert flushed == []
        assert store.get(scope, topic_id) is not None

    @pytest.mark.asyncio
    async def test_shutdown_flush_classifies_settled_skipped_and_failed(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        ok_id = await familiar.apply_interaction(
            _payload("1"), identity_scope=scope, interaction_id="i-1"
        )
        empty_id = await familiar.prepare_topic("NEW_TOPIC", "标题", "摘要", scope)
        bad_id = await familiar.apply_interaction(
            _payload("3"), identity_scope=scope, interaction_id="i-3"
        )
        bus.request = AsyncMock(
            side_effect=[_generation_task("task-1", ok_id), RuntimeError("admission boom")]
        )

        report = await familiar.flush_all_for_shutdown()

        assert report.settled_topic_ids == (ok_id, empty_id)
        assert report.generation_skipped_topic_ids == (empty_id,)
        assert report.failed_topic_ids == (bad_id,)
        assert report.resident_block_count == 2  # ok + bad 各 1 块
        assert store.get(scope, ok_id) is None
        assert store.get(scope, empty_id) is None
        # admission 失败：话题内容保留（lease 释放后可重试），仅计入 failed
        assert store.get(scope, bad_id) is not None


# ========== 感知关闭（engine=None）==========


class TestDisabledPerception:
    @pytest.mark.asyncio
    async def test_disabled_engine_has_no_side_effects(self):
        familiar, store, _, _, journal = _make_familiar(engine=None)
        scope = _identity_scope()

        topic_id = await familiar.apply_interaction(
            _payload(), identity_scope=scope, interaction_id="i-1"
        )

        assert topic_id == "NEW_TOPIC"
        assert store.list_all() == []
        assert journal.get("i-1") is None
