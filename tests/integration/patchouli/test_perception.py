"""PerceptionFamiliar 感知链路集成测试（lease 编排架构）。

真实协作：PerceptionFamiliar + MemoryPerceptionEngine + TopicWorkingSet
+ ShortTermMemoryStore；Relay（LLM 摘要）与
Bus（Generation 队列）是边界之外协作者，使用 Mock，时间使用可控时钟。

测试覆盖:
- apply_interaction: 新建（含提交携带的标题与摘要）/指定话题写入、binding 幂等、busy 拒绝、未知目标
- LRU 驱逐: 池满驱逐最久未访问话题、busy 候选改选、admission 失败传播
- 折叠: 应用不在内部折叠、needs_compact 判断、折叠摘要累积写回、失败后话题完整且可再次折叠、
  占用时 busy 拒绝、话题已结算时直接结束、未超保留块数不折叠
- manual_settle_topic / evict_topic / discard_if_empty 具名用例，含 admission 失败可重试、
  全部 block 被过滤与 summary-only 话题
- scan_idle_buffers_once / flush_all_for_shutdown 维护投影，含跨 Workspace 结算、
  折叠后结算、占用话题记为 failed 与生成跳过
- 感知关闭时无副作用
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.components.work_queue import WorkQueueCapacityError, WorkQueueStoppedError
from hivememory.config.patchouli import SemanticFlowPerceptionConfig
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import (
    WorkspaceAssetRef,
    system_actor_for_workspace,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TriggerReason
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1", agent_id="a1"):
    return make_identity_scope(user_id=user_id, agent_id=agent_id)


def _payload(message: str = "hello", answer: str | None = None) -> InteractionPayload:
    return make_interaction_payload(message, answer if answer is not None else f"answer:{message}")


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
    relay.generate_summary = AsyncMock(
        side_effect=lambda blocks_to_fold, previous_summary=None: (
            f"{previous_summary}|folded:{len(blocks_to_fold)}"
            if previous_summary
            else f"folded:{len(blocks_to_fold)}"
        )
    )
    return relay


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
    """构造 (familiar, store, working_set, bus) 测试组合。

    Engine（含 RelayController）/ Store / WorkingSet / Journal 为真实对象；
    时间使用可控时钟。
    """
    clock = clock or _FakeClock()
    store = ShortTermMemoryStore()
    working_set = TopicWorkingSet(max_resident=max_resident, clock=clock)
    if relay is None:
        relay = Mock()
        relay.generate_summary = AsyncMock(return_value="test summary")
    bus = bus or Mock()
    bus.request = AsyncMock(return_value=None)
    if engine == "default":
        engine = MemoryPerceptionEngine(
            config=engine_config or SemanticFlowPerceptionConfig(fold_token_threshold=999999),
            relay_controller=relay,
        )
    familiar = PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=working_set,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=idle_timeout),
    )
    return familiar, store, working_set, bus


# ========== apply_interaction：核心摄入用例 ==========


class TestApplyInteraction:
    @pytest.mark.asyncio
    async def test_new_topic_creates_topic_and_tracks_residency(self):
        familiar, store, working_set, bus = _make_familiar()
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
        # 驻留工作集记录了话题（shutdown 候选可观察）
        assert {tid for _, tid in working_set.list_shutdown_candidates()} == {topic_id}

    @pytest.mark.asyncio
    async def test_new_topic_uses_submitted_title_and_summary(self):
        """提交携带的新话题标题与摘要用于建立话题，后续落位依赖它们。"""
        familiar, store, _, _ = _make_familiar()
        scope = _identity_scope()

        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-titled",
            new_topic_title="部署贪吃蛇",
            new_topic_summary="讨论贪吃蛇游戏的部署步骤",
        )

        topic = store.get(scope.workspace_identity, topic_id)
        assert topic.topic_title == "部署贪吃蛇"
        assert topic.topic_summary == "讨论贪吃蛇游戏的部署步骤"

    @pytest.mark.asyncio
    async def test_existing_topic_appends_blocks_and_dedupes_bindings(self):
        familiar, store, _, _ = _make_familiar()
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
    async def test_busy_topic_rejects_interaction_without_writing(self):
        familiar, store, working_set, _ = _make_familiar()
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
        familiar, _, _, _ = _make_familiar()

        with pytest.raises(ResourceNotFoundError):
            await familiar.apply_interaction(
                _payload(),
                belong_to=_identity_scope().workspace_identity,
                from_actor=_identity_scope().actor_identity,
                target_topic_id="missing",
            )


# ========== LRU 驱逐 ==========


class TestLruEviction:
    @pytest.mark.asyncio
    async def test_pool_full_evicts_least_recently_used_topic(self):
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(max_resident=2, clock=clock)
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
        familiar, store, working_set, _ = _make_familiar(max_resident=2, clock=clock)
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
        familiar, store, _, bus = _make_familiar(max_resident=1)
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
        familiar, _, working_set, _ = _make_familiar(max_resident=1)
        scope = _identity_scope()
        t1 = await familiar.apply_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        working_set.acquire(scope.workspace_identity, t1)  # 唯一候选正被占用

        with pytest.raises(TopicBusyError, match="无可占用候选"):
            await familiar.apply_interaction(
                _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
            )


# ========== 折叠（compact） ==========


def _fold_every_block_config() -> SemanticFlowPerceptionConfig:
    """每个 block 都超过阈值、只保留最近 1 块：第二块写入后即有可折叠前缀。"""
    return SemanticFlowPerceptionConfig(fold_token_threshold=1, fold_retain_recent_blocks=1)


async def _apply_to_topic(familiar, scope, topic_id: str, *messages: str) -> None:
    for message in messages:
        await familiar.apply_interaction(
            _payload(message),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id=topic_id,
            interaction_id=f"i-{topic_id}-{message}",
        )


class TestCompact:
    @pytest.mark.asyncio
    async def test_apply_writes_block_without_compacting_over_threshold_topic(self):
        """应用只写入 block；话题超过阈值也不在应用内折叠。"""
        familiar, store, _, _ = _make_familiar(
            engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        await _apply_to_topic(familiar, scope, "t1", "a", "b")

        topic = store.get(scope.workspace_identity, "t1")
        assert [block.user_query for block in topic.blocks] == ["a", "b"]
        assert topic.state_summary == ""
        assert familiar.needs_compact(scope.workspace_identity, "t1") is True

    @pytest.mark.asyncio
    async def test_compact_topic_folds_old_blocks_into_cumulative_summary(self):
        """折叠保留最近的 block，旧前缀并入在上一次摘要之上累积的新摘要。"""
        familiar, store, _, bus = _make_familiar(
            engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        for message in ("a", "b", "c"):
            await _apply_to_topic(familiar, scope, "t1", message)
            await familiar.compact_topic(scope.workspace_identity, "t1")

        topic = store.get(scope.workspace_identity, "t1")
        assert [block.user_query for block in topic.blocks] == ["c"]
        # 第二次折叠收到第一次的摘要作为 previous_summary，累积成链
        assert topic.state_summary == "folded:1|folded:1"
        assert topic.total_tokens == topic.blocks[0].total_tokens
        assert familiar.needs_compact(scope.workspace_identity, "t1") is False
        # 折叠只压缩上下文，不触发记忆生成，也不驱逐话题
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_compaction_leaves_topic_intact_and_can_compact_again(self):
        """摘要失败时话题原样保留并释放占用，之后的折叠照常完成。"""
        relay = Mock()
        relay.generate_summary = AsyncMock(
            side_effect=[RuntimeError("relay down"), "folded-summary"]
        )
        familiar, store, _, _ = _make_familiar(
            engine_config=_fold_every_block_config(), relay=relay
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")
        before = store.get(scope.workspace_identity, "t1")

        with pytest.raises(RuntimeError, match="relay down"):
            await familiar.compact_topic(scope.workspace_identity, "t1")

        assert store.get(scope.workspace_identity, "t1") == before
        await familiar.compact_topic(scope.workspace_identity, "t1")
        topic = store.get(scope.workspace_identity, "t1")
        assert [block.user_query for block in topic.blocks] == ["b"]
        assert topic.state_summary == "folded-summary"

    @pytest.mark.asyncio
    async def test_compact_topic_rejects_topic_held_by_another_use_case(self):
        """话题正被占用（例如结算中）时折叠以 busy 拒绝，留待重试，不改动话题。"""
        familiar, store, working_set, _ = _make_familiar(
            engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")
        lease = working_set.acquire(scope.workspace_identity, "t1")
        assert lease is not None

        with pytest.raises(TopicBusyError, match="正忙"):
            await familiar.compact_topic(scope.workspace_identity, "t1")

        working_set.release(lease)
        assert store.get(scope.workspace_identity, "t1").state_summary == ""

    @pytest.mark.asyncio
    async def test_compact_topic_ends_without_reviving_settled_topic(self):
        """折叠开始前话题已被结算删除：直接结束，不把话题写回。"""
        familiar, store, _, bus = _make_familiar(
            engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")
        bus.request.side_effect = _accept_settlement_task
        await familiar.manual_settle_topic(scope.workspace_identity, "t1")

        await familiar.compact_topic(scope.workspace_identity, "t1")

        assert store.get(scope.workspace_identity, "t1") is None
        assert familiar.needs_compact(scope.workspace_identity, "t1") is False

    @pytest.mark.asyncio
    async def test_blocks_within_retain_count_need_no_compaction(self):
        """超过阈值但块数不多于保留数时没有可折叠前缀：不需要折叠，折叠也不改动话题。"""
        familiar, store, _, _ = _make_familiar(
            engine_config=SemanticFlowPerceptionConfig(
                fold_token_threshold=1, fold_retain_recent_blocks=5
            )
        )
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload("q", "a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )

        assert familiar.needs_compact(scope.workspace_identity, topic_id) is False
        await familiar.compact_topic(scope.workspace_identity, topic_id)
        data = store.get(scope.workspace_identity, topic_id)
        assert data.state_summary == ""
        assert len(data.blocks) == 1


# ========== manual_settle / evict / discard 具名用例 ==========


class TestNamedUseCases:
    @pytest.mark.asyncio
    async def test_manual_settle_admits_task_then_deletes_topic(self):
        familiar, store, _, bus = _make_familiar()
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
        familiar, store, _, bus = _make_familiar()
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
        familiar, store, working_set, _ = _make_familiar()
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
    async def test_manual_settle_unknown_topic_raises_not_found(self):
        familiar, _, _, _ = _make_familiar()

        with pytest.raises(ResourceNotFoundError):
            await familiar.manual_settle_topic(_identity_scope().workspace_identity, "missing")

    @pytest.mark.asyncio
    async def test_evict_topic_removes_without_settlement(self):
        familiar, store, working_set, bus = _make_familiar()
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
        familiar, store, working_set, _ = _make_familiar()
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
        familiar, store, _, _ = _make_familiar()
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
        familiar, store, _, _ = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)
        topic = store.get(scope.workspace_identity, topic_id)
        store.put(topic.model_copy(update={"state_summary": "折叠历史"}))

        discarded = familiar.discard_if_empty(scope.workspace_identity, topic_id)

        assert discarded is False
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure", [WorkQueueCapacityError("memory", 1), WorkQueueStoppedError("stopped")]
    )
    async def test_manual_settle_admission_failure_keeps_topic_intact_and_allows_retry(
        self, failure
    ):
        familiar, store, _, bus = _make_familiar()
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload("question", "answer"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )
        before = store.get(scope.workspace_identity, topic_id)
        bus.request = AsyncMock(side_effect=failure)

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


# ========== 维护与 shutdown ==========


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_scan_idle_settles_only_timed_out_topics(self):
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(idle_timeout=30, clock=clock)
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
        familiar, store, working_set, _ = _make_familiar(idle_timeout=30, clock=clock)
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
        familiar, store, _, bus = _make_familiar(idle_timeout=30, clock=clock)
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
        familiar, store, _, bus = _make_familiar()
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
        familiar, store, _, bus = _make_familiar(idle_timeout=1, clock=clock)
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
        familiar, store, _, _ = _make_familiar(idle_timeout=1, max_resident=2, clock=clock)
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
        familiar, store, _, bus = _make_familiar(max_resident=4)
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
        familiar, store, working_set, bus = _make_familiar()
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
        familiar, store, _, bus = _make_familiar(
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
            await familiar.compact_topic(scope.workspace_identity, topic_id)

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
        familiar, store, _, _ = _make_familiar()
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


# ========== 感知关闭（engine=None）==========


class TestDisabledPerception:
    @pytest.mark.asyncio
    async def test_disabled_engine_has_no_side_effects(self):
        familiar, store, _, _ = _make_familiar(engine=None)
        scope = _identity_scope()

        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        assert topic_id == "NEW_TOPIC"
        assert store.list_all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [KeyError("handler bug"), asyncio.CancelledError()])
async def test_manual_settle_unknown_failure_keeps_topic_and_propagates(failure):
    """未知接纳结果与取消不改判明确拒绝；保留材料并释放占用，供后续核实。"""
    familiar, store, _, bus = _make_familiar()
    scope = _identity_scope()
    topic_id = await familiar.apply_interaction(
        _payload("question", "answer"),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        target_topic_id="NEW_TOPIC",
    )
    before = store.get(scope.workspace_identity, topic_id)
    bus.request = AsyncMock(side_effect=failure)

    with pytest.raises(type(failure)) as caught:
        await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

    assert caught.value is failure
    assert store.get(scope.workspace_identity, topic_id) == before
    # 释放占用后仍可由拥有者操作话题；失败不能留下永久 busy 状态。
    result = await familiar.evict_topic(scope.workspace_identity, topic_id)
    assert result.removed is True
    assert store.get(scope.workspace_identity, topic_id) is None
