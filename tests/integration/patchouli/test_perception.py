"""PerceptionFamiliar 感知链路集成测试（lease 编排架构）。

真实协作：PerceptionFamiliar + MemoryPerceptionEngine + TopicWorkingSet
+ ShortTermMemoryStore + InMemoryInteractionApplyJournal；Relay（LLM 摘要）与
Bus（Generation 队列）是边界之外协作者，使用 Mock，时间使用可控时钟。

测试覆盖:
- apply_interaction: 新建/指定话题写入、binding 幂等、busy 拒绝、未知目标、
  retry 幂等与等价性冲突、跨 Workspace 复用 interaction_id 被拒
- LRU 驱逐: 池满驱逐最久未访问话题、busy 候选改选、admission 失败传播
- token 溢出 compact: 折叠摘要写回、relay 失败后 retry 续跑不重复写块、未超保留块数不折叠
- manual_settle_topic / evict_topic / discard_if_empty 具名用例，含 admission 失败可重试、
  全部 block 被过滤与 summary-only 话题
- scan_idle_buffers_once / flush_all_for_shutdown 维护投影，含跨 Workspace 结算、
  折叠后结算、占用话题记为 failed 与生成跳过
- 感知关闭时无副作用且重放不依赖 apply journal
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.config.patchouli import SemanticFlowPerceptionConfig
from hivememory.core.models import (
    LogicalBlock,
    TurnEvent,
    TurnRecord,
    WorkspaceAssetRef,
    system_actor_for_workspace,
)
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
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1", agent_id="a1"):
    return make_identity_scope(user_id=user_id, agent_id=agent_id)


def _payload(message: str = "hello", answer: str | None = None) -> InteractionPayload:
    answer = answer if answer is not None else f"answer:{message}"
    return InteractionPayload(
        user_message=message,
        assistant_final_text=answer,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=answer,
            )
        ],
    )


def _generation_task(task_id="task-1", topic_id="t1") -> MemoryGenerationTask:
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.SETTLE,
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
    )


def _accept_settlement_task(route, payload):
    """在 generation 边界返回确定性的已接纳任务快照。"""
    return MemoryGenerationTask(
        task_id=f"memtask-{payload.topic_id}",
        topic_id=payload.topic_id,
        label=payload.topic_id,
        source=MemoryGenerationSource.SETTLE,
        belong_to=payload.belong_to,
        from_actor=system_actor_for_workspace(payload.belong_to),
    )


def _folding_relay() -> Mock:
    """摘要记录每次折叠的块数并在旧摘要上累积，便于断言折叠次数。"""
    relay = Mock()
    relay.generate_summary.side_effect = lambda blocks_to_fold, previous_summary=None: (
        f"{previous_summary}|folded:{len(blocks_to_fold)}"
        if previous_summary
        else f"folded:{len(blocks_to_fold)}"
    )
    return relay


def _filtered_block() -> LogicalBlock:
    """worth_saving=False 的块：保留在工作集，但不进入结算材料。"""
    return LogicalBlock(
        turn=TurnRecord(user_query="q", assistant_final_text="a"),
        worth_saving=False,
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
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        topic = store.get(scope.workspace_identity, topic_id)
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
        store.create(scope.workspace_identity, topic_id="t-fixed", topic_title="Fixed")
        ref = WorkspaceAssetRef(token="token-1", asset_id="asset-1")

        first = await familiar.apply_interaction(
            _payload("a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t-fixed",
            interaction_id="i-1",
            asset_refs=(ref,),
        )
        second = await familiar.apply_interaction(
            _payload("b"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t-fixed",
            interaction_id="i-2",
            asset_refs=(WorkspaceAssetRef(token="token-1", asset_id="asset-1"),),
        )

        assert first == second == "t-fixed"
        topic = store.get(scope.workspace_identity, "t-fixed")
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
                payload,
                belong_to=_identity_scope().workspace_identity,
                from_actor=_identity_scope().actor_identity,
                interaction_id="i-1",
            )
        assert store.list_all() == []  # 未产生任何话题写入

    @pytest.mark.asyncio
    async def test_busy_topic_rejects_interaction_without_writing(self):
        familiar, store, working_set, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        lease = working_set.acquire(scope.workspace_identity, "t1")
        assert lease is not None

        with pytest.raises(TopicBusyError, match="正忙"):
            await familiar.apply_interaction(
                _payload(),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id="t1",
            )

        assert store.get(scope.workspace_identity, "t1").block_count == 0

    @pytest.mark.asyncio
    async def test_unknown_target_is_rejected(self):
        familiar, _, _, _, _ = _make_familiar()

        with pytest.raises(KeyError, match="does not exist"):
            await familiar.apply_interaction(
                _payload(),
                belong_to=_identity_scope().workspace_identity,
                from_actor=_identity_scope().actor_identity,
                target_topic_id="missing",
            )

    @pytest.mark.asyncio
    async def test_retry_with_same_payload_is_idempotent(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        payload = _payload("m")

        first = await familiar.apply_interaction(
            payload,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-1",
        )
        replayed = await familiar.apply_interaction(
            payload,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-1",
        )

        assert replayed == first
        assert store.get(scope.workspace_identity, "t1").block_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_different_payload_conflicts(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        await familiar.apply_interaction(
            _payload("m"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-1",
        )
        with pytest.raises(ValueError, match="different input"):
            await familiar.apply_interaction(
                _payload("other"),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id="t1",
                interaction_id="i-1",
            )
        assert store.get(scope.workspace_identity, "t1").block_count == 1

    @pytest.mark.asyncio
    async def test_default_manual_settle_targets_last_active_topic(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        result = await familiar.manual_settle_topic(scope.workspace_identity)

        assert result.topic_id == topic_id
        assert store.get(scope.workspace_identity, topic_id) is None

    @pytest.mark.asyncio
    async def test_interaction_retry_rejects_same_content_from_another_workspace(self):
        """同一 interaction_id 不能跨 Workspace 复用已完成的 apply 记录。"""
        familiar, store, _, _, _ = _make_familiar()
        main_scope = make_identity_scope(user_id="u1", agent_id="a1", workspace_id="main_workspace")
        isolated_scope = make_identity_scope(
            user_id="u1", agent_id="a1", workspace_id="isolated_workspace"
        )
        payload = _payload("same question", "same answer")

        topic_id = await familiar.apply_interaction(
            payload,
            belong_to=main_scope.workspace_identity,
            from_actor=main_scope.actor_identity,
            target_topic_id="NEW_TOPIC",
            interaction_id="interaction-shared",
        )

        with pytest.raises(ValueError, match="different input"):
            await familiar.apply_interaction(
                payload,
                belong_to=isolated_scope.workspace_identity,
                from_actor=isolated_scope.actor_identity,
                target_topic_id="NEW_TOPIC",
                interaction_id="interaction-shared",
            )

        original = store.get(main_scope.workspace_identity, topic_id)
        assert original is not None
        assert [block.user_query for block in original.blocks] == ["same question"]
        assert store.list_by_workspace(isolated_scope.workspace_identity) == []


# ========== LRU 驱逐 ==========


class TestLruEviction:
    @pytest.mark.asyncio
    async def test_pool_full_evicts_least_recently_used_topic(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(max_resident=2, clock=clock)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)
        t2 = await familiar.apply_interaction(
            _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)

        t3 = await familiar.apply_interaction(
            _payload("3"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )

        assert store.get(scope.workspace_identity, t1) is None  # 最久未访问被驱逐
        assert store.get(scope.workspace_identity, t2) is not None
        assert store.get(scope.workspace_identity, t3) is not None
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
        t1 = await familiar.apply_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)
        t2 = await familiar.apply_interaction(
            _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)
        lease = working_set.acquire(scope.workspace_identity, t1)
        assert lease is not None

        t3 = await familiar.apply_interaction(
            _payload("3"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )

        # t1 被占用跳过，改选 t2 驱逐
        assert store.get(scope.workspace_identity, t1) is not None
        assert store.get(scope.workspace_identity, t2) is None
        assert store.get(scope.workspace_identity, t3) is not None
        working_set.release(lease)

    @pytest.mark.asyncio
    async def test_admission_failure_preserves_topic_and_propagates(self):
        familiar, store, _, bus, _ = _make_familiar(max_resident=1)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(RuntimeError, match="admission boom"):
            await familiar.apply_interaction(
                _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
            )

        # admission 失败：话题内容完整保留，可重试
        assert store.get(scope.workspace_identity, t1) is not None
        assert store.get(scope.workspace_identity, t1).block_count == 1

    @pytest.mark.asyncio
    async def test_pool_full_with_all_candidates_busy_raises(self):
        familiar, _, working_set, _, _ = _make_familiar(max_resident=1)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        working_set.acquire(scope.workspace_identity, t1)  # 唯一候选正被占用

        with pytest.raises(TopicBusyError, match="无可占用候选"):
            await familiar.apply_interaction(
                _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
            )


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
        familiar, store, _, bus, _ = _make_familiar(engine_config=engine_config, relay=mock_relay)
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        await familiar.apply_interaction(
            _payload("a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-1",
        )
        await familiar.apply_interaction(
            _payload("b"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-2",
        )
        await familiar.apply_interaction(
            _payload("c"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-3",
        )

        topic = store.get(scope.workspace_identity, "t1")
        assert topic.block_count == 1  # 每轮折叠后只保留最近 1 块
        # 第二次折叠收到第一次的摘要（"---folded"）作为 previous_summary，累积成链
        assert topic.state_summary == "---folded---folded"
        assert topic.total_tokens == topic.blocks[0].total_tokens
        # 折叠只压缩上下文，不触发记忆生成，也不驱逐话题
        bus.request.assert_not_awaited()

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
        familiar, store, _, _, journal = _make_familiar(
            engine_config=engine_config, relay=mock_relay
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        payload = _payload("b")

        await familiar.apply_interaction(
            _payload("a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-1",
        )
        with pytest.raises(TransientInteractionSubmissionError):
            await familiar.apply_interaction(
                payload,
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id="t1",
                interaction_id="i-2",
            )
        # block 已写入、journal 停在 INTERACTION_APPLIED
        assert store.get(scope.workspace_identity, "t1").block_count == 2
        assert journal.get("i-2").stage is InteractionApplyStage.INTERACTION_APPLIED

        # 同 payload retry：不重复写块，compact 后置义务补跑完成
        await familiar.apply_interaction(
            payload,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t1",
            interaction_id="i-2",
        )
        topic = store.get(scope.workspace_identity, "t1")
        assert topic.block_count == 1
        assert topic.state_summary == "folded-summary"
        assert journal.get("i-2").stage is InteractionApplyStage.COMPLETED

    @pytest.mark.asyncio
    async def test_compact_is_noop_when_blocks_not_exceeding_retain(self):
        familiar, store, _, _, _ = _make_familiar(
            engine_config=SemanticFlowPerceptionConfig(
                fold_token_threshold=10, fold_retain_recent_blocks=5
            )
        )
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload("q", "a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )

        data = store.get(scope.workspace_identity, topic_id)
        assert data.state_summary == ""
        assert len(data.blocks) == 1


# ========== manual_settle / evict / discard 具名用例 ==========


class TestNamedUseCases:
    @pytest.mark.asyncio
    async def test_manual_settle_admits_task_then_deletes_topic(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        accepted = _generation_task("task-1", topic_id)
        bus.request = AsyncMock(return_value=accepted)

        result = await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

        assert result.topic_id == topic_id
        assert result.generation_task_id == "task-1"
        assert result.generation_submitted is True
        assert store.get(scope.workspace_identity, topic_id) is None
        assert bus.request.await_args.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT

    @pytest.mark.asyncio
    async def test_manual_settle_empty_topic_skips_generation(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic(
            "NEW_TOPIC", "标题", "摘要", scope.workspace_identity
        )

        result = await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

        assert result.generation_submitted is False
        assert result.generation_task_id is None
        bus.request.assert_not_awaited()
        assert store.get(scope.workspace_identity, topic_id) is None

    @pytest.mark.asyncio
    async def test_manual_settle_busy_topic_raises_busy_error(self):
        familiar, store, working_set, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        working_set.acquire(scope.workspace_identity, topic_id)

        with pytest.raises(TopicBusyError):
            await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_manual_settle_without_topic_or_active_history_raises(self):
        familiar, _, _, _, _ = _make_familiar()

        with pytest.raises(ValueError, match="未指定"):
            await familiar.manual_settle_topic(_identity_scope().workspace_identity)

    @pytest.mark.asyncio
    async def test_evict_topic_removes_without_settlement(self):
        familiar, store, working_set, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        result = await familiar.evict_topic(scope.workspace_identity, topic_id)

        assert result.removed is True
        assert store.get(scope.workspace_identity, topic_id) is None
        assert working_set.list_shutdown_candidates() == []
        bus.request.assert_not_awaited()  # evict 不触发结算

    @pytest.mark.asyncio
    async def test_evict_busy_topic_reports_not_removed(self):
        familiar, store, working_set, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        working_set.acquire(scope.workspace_identity, topic_id)

        result = await familiar.evict_topic(scope.workspace_identity, topic_id)

        assert result.removed is False
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_discard_if_empty_removes_only_truly_empty_topic(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        empty_id = await familiar.prepare_topic(
            "NEW_TOPIC", "标题", "摘要", scope.workspace_identity
        )
        content_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        assert familiar.discard_if_empty(scope.workspace_identity, empty_id) is True
        assert familiar.discard_if_empty(scope.workspace_identity, content_id) is False
        assert store.get(scope.workspace_identity, empty_id) is None
        assert store.get(scope.workspace_identity, content_id) is not None

    @pytest.mark.asyncio
    async def test_discard_if_empty_keeps_summary_only_topic(self):
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)
        topic = store.get(scope.workspace_identity, topic_id)
        store.put(topic.model_copy(update={"state_summary": "折叠历史"}))

        discarded = familiar.discard_if_empty(scope.workspace_identity, topic_id)

        assert discarded is False
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_manual_settle_admission_failure_keeps_topic_intact_and_allows_retry(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload("question", "answer"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )
        before = store.get(scope.workspace_identity, topic_id)
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(TopicSettleAdmissionError, match="可重试"):
            await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

        after = store.get(scope.workspace_identity, topic_id)
        assert after is not None
        assert [b.user_query for b in after.blocks] == [b.user_query for b in before.blocks]
        assert after.state_summary == before.state_summary

        # 修复 admission 后可再次重试并成功结束生命周期
        bus.request = AsyncMock(return_value=None)
        result = await familiar.manual_settle_topic(scope.workspace_identity, topic_id)
        assert result.topic_id == topic_id
        assert store.get(scope.workspace_identity, topic_id) is None

    @pytest.mark.asyncio
    async def test_manual_settle_with_all_blocks_filtered_evicts_without_task(self):
        """blocks 均被 worth_saving=False 过滤：无任务但生命周期仍正常结束。"""
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)
        topic = store.get(scope.workspace_identity, topic_id)
        store.put(topic.model_copy(update={"blocks": (_filtered_block(),)}))

        result = await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

        assert result.topic_id == topic_id
        assert result.generation_submitted is False
        assert store.get(scope.workspace_identity, topic_id) is None
        bus.request.assert_not_awaited()


# ========== 维护与 shutdown ==========


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_scan_idle_settles_only_timed_out_topics(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(idle_timeout=30, clock=clock)
        scope = _identity_scope()
        stale_id = await familiar.apply_interaction(
            _payload("1"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        fresh_id = await familiar.apply_interaction(
            _payload("2"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-2",
        )

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == [stale_id]
        assert store.get(scope.workspace_identity, stale_id) is None
        assert store.get(scope.workspace_identity, fresh_id) is not None
        route, settlement = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert settlement.reason == TriggerReason.IDLE_TIMEOUT

    @pytest.mark.asyncio
    async def test_scan_idle_skips_leased_topic(self):
        clock = _FakeClock()
        familiar, store, working_set, _, _ = _make_familiar(idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        working_set.acquire(scope.workspace_identity, topic_id)

        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == []
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_scan_idle_admission_failure_skips_and_preserves(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        flushed = await familiar.scan_idle_buffers_once()

        # idle 维护不向上传播；话题保留等待下一轮
        assert flushed == []
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_shutdown_flush_classifies_settled_skipped_and_failed(self):
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        ok_id = await familiar.apply_interaction(
            _payload("1"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        empty_id = await familiar.prepare_topic(
            "NEW_TOPIC", "标题", "摘要", scope.workspace_identity
        )
        bad_id = await familiar.apply_interaction(
            _payload("3"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-3",
        )
        bus.request = AsyncMock(
            side_effect=[_generation_task("task-1", ok_id), RuntimeError("admission boom")]
        )

        report = await familiar.flush_all_for_shutdown()

        assert report.settled_topic_ids == (ok_id, empty_id)
        assert report.generation_skipped_topic_ids == (empty_id,)
        assert report.failed_topic_ids == (bad_id,)
        assert report.resident_block_count == 2  # ok + bad 各 1 块
        assert store.get(scope.workspace_identity, ok_id) is None
        assert store.get(scope.workspace_identity, empty_id) is None
        # admission 失败：话题内容保留（lease 释放后可重试），仅计入 failed
        assert store.get(scope.workspace_identity, bad_id) is not None

    @pytest.mark.asyncio
    async def test_idle_flush_skips_empty_settlement_submission(self):
        clock = _FakeClock()
        familiar, store, _, bus, _ = _make_familiar(idle_timeout=1, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)
        assert store.get(scope.workspace_identity, topic_id) is not None

        clock.advance(100)
        flushed = await familiar.scan_idle_buffers_once()

        assert flushed == [topic_id]
        # 真正空 Topic：不提交 settlement，但生命周期正常结束
        bus.request.assert_not_awaited()
        assert store.get(scope.workspace_identity, topic_id) is None

    @pytest.mark.asyncio
    async def test_idle_flush_frees_slot(self):
        clock = _FakeClock()
        familiar, store, _, _, _ = _make_familiar(idle_timeout=1, max_resident=2, clock=clock)
        first = _identity_scope(user_id="u1", agent_id="a1")
        second = _identity_scope(user_id="u2", agent_id="a2")
        third = _identity_scope(user_id="u3", agent_id="a3")
        for scope, message in ((first, "q1"), (second, "q2")):
            await familiar.apply_interaction(
                _payload(message),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id="NEW_TOPIC",
            )
        assert len(store.list_by_workspace(first.workspace_identity)) == 1
        assert len(store.list_by_workspace(second.workspace_identity)) == 1

        clock.advance(100)
        assert len(await familiar.scan_idle_buffers_once()) == 2

        await familiar.apply_interaction(
            _payload("q3"),
            belong_to=third.workspace_identity,
            from_actor=third.actor_identity,
            target_topic_id="NEW_TOPIC",
        )
        assert len(store.list_by_workspace(third.workspace_identity)) == 1

    @pytest.mark.asyncio
    async def test_shutdown_flush_settles_and_swaps_out_all_topics(self):
        familiar, store, _, bus, _ = _make_familiar(max_resident=4)
        first = _identity_scope(user_id="u1", agent_id="a1")
        second = _identity_scope(user_id="u2", agent_id="a2")
        for scope, message in ((first, "q1"), (second, "q2")):
            await familiar.apply_interaction(
                _payload(message),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id="NEW_TOPIC",
            )
        bus.request.side_effect = _accept_settlement_task

        result = await familiar.flush_all_for_shutdown()

        assert len(result.settled_topic_ids) == 2
        assert result.generation_skipped_topic_ids == ()
        assert result.resident_block_count == 2
        assert store.list_by_workspace(first.workspace_identity) == []
        assert store.list_by_workspace(second.workspace_identity) == []
        assert bus.request.await_count == 2
        for call in bus.request.await_args_list:
            assert call.args[0] == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
            assert call.args[1].reason == TriggerReason.SHUTDOWN

    @pytest.mark.asyncio
    async def test_shutdown_reports_busy_topic_in_failed_ids_without_generation(self):
        """drain 后仍被占用的话题属于关闭顺序缺陷，隔离记录为 failed。"""
        familiar, store, working_set, bus, _ = _make_familiar()
        scope = _identity_scope(user_id="u-busy", agent_id="a-busy")
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)
        assert working_set.acquire(scope.workspace_identity, topic_id) is not None

        result = await familiar.flush_all_for_shutdown()

        assert result.failed_topic_ids == (topic_id,)
        assert result.settled_topic_ids == ()
        assert store.get(scope.workspace_identity, topic_id) is not None
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_after_folding_settles_summary_and_retained_block(self):
        familiar, store, _, bus, _ = _make_familiar(
            max_resident=2,
            engine_config=SemanticFlowPerceptionConfig(
                fold_token_threshold=10, fold_retain_recent_blocks=1
            ),
            relay=_folding_relay(),
        )
        scope = _identity_scope(user_id="u-fold", agent_id="a-fold")
        topic_id = "NEW_TOPIC"

        for i in range(3):
            topic_id = await familiar.apply_interaction(
                _payload(f"question-{i}-" * 80, f"answer-{i}"),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id=topic_id,
            )

        before_shutdown = store.get(scope.workspace_identity, topic_id)
        assert before_shutdown is not None
        assert before_shutdown.state_summary == "folded:1|folded:1"
        assert [block.user_query for block in before_shutdown.blocks] == ["question-2-" * 80]
        assert bus.request.await_count == 0
        bus.request.side_effect = _accept_settlement_task

        result = await familiar.flush_all_for_shutdown()

        assert result.settled_topic_ids == (topic_id,)
        assert result.generation_skipped_topic_ids == ()
        assert result.resident_block_count == 1
        assert store.get(scope.workspace_identity, topic_id) is None
        bus.request.assert_awaited_once()
        route, settlement = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert settlement.reason == TriggerReason.SHUTDOWN
        assert settlement.state_summary == "folded:1|folded:1"
        assert [block.user_query for block in settlement.blocks] == ["question-2-" * 80]

    @pytest.mark.asyncio
    async def test_shutdown_marks_generation_skip_when_submission_builds_no_task(self):
        """下游正常返回无任务时，Topic 已结算但 generation 被跳过。"""
        familiar, store, _, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload("question", "answer"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )

        result = await familiar.flush_all_for_shutdown()

        assert result.settled_topic_ids == (topic_id,)
        assert result.generation_skipped_topic_ids == (topic_id,)
        assert store.get(scope.workspace_identity, topic_id) is None

    @pytest.mark.asyncio
    async def test_shutdown_marks_filtered_topic_as_generation_skip_and_evicts_it(self):
        """全部 blocks 被过滤时不能因为 block_count 非零而漏报 skip。"""
        familiar, store, _, bus, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)
        topic = store.get(scope.workspace_identity, topic_id)
        store.put(topic.model_copy(update={"blocks": (_filtered_block(),)}))

        result = await familiar.flush_all_for_shutdown()

        assert result.settled_topic_ids == (topic_id,)
        assert result.generation_skipped_topic_ids == (topic_id,)
        assert result.resident_block_count == 1
        assert store.get(scope.workspace_identity, topic_id) is None
        bus.request.assert_not_awaited()


# ========== 感知关闭（engine=None）==========


class TestDisabledPerception:
    @pytest.mark.asyncio
    async def test_disabled_engine_has_no_side_effects(self):
        familiar, store, _, _, journal = _make_familiar(engine=None)
        scope = _identity_scope()

        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        replayed_topic_id = await asyncio.wait_for(
            familiar.apply_interaction(
                _payload(),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                interaction_id="i-1",
            ),
            timeout=1,
        )

        assert topic_id == "NEW_TOPIC"
        assert replayed_topic_id == "NEW_TOPIC"
        assert store.list_all() == []
        assert journal.get("i-1") is None
