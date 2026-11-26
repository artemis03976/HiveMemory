"""PerceptionFamiliar 感知链路集成测试（lease 编排架构）。

真实协作：PerceptionFamiliar + MemoryPerceptionEngine + TopicWorkingSet
+ ShortTermMemoryStore（话题日志写在临时目录）；Relay（LLM 摘要）与
Bus（Generation 队列）是边界之外协作者，使用 Mock，时间使用可控时钟。

测试覆盖:
- apply_interaction: 新建（含提交携带的标题与摘要）/指定话题写入、binding 幂等、busy 拒绝、未知目标
- LRU 驱逐: 池满驱逐最久未访问话题、busy 候选改选、admission 失败传播
- 折叠: 应用不在内部折叠、needs_compact 判断、折叠追加折叠记录且不移除交互记录、摘要累积、
  失败后话题完整且可再次折叠、占用时 busy 拒绝、话题已结算时直接结束、未超保留块数不折叠
- manual_settle_topic / evict_topic 具名用例，含 admission 失败可重试、已沉淀的话题退役不生成
- scan_idle_topics_once：空闲只沉淀，话题留在话题池、访问时间不变，没有新交互不再沉淀；
  从没接收过交互的空话题空闲时直接退役；折叠后沉淀携带被折叠的轮次
- 沉淀进度在生成完成时推进：接纳后只记在途，在途期间不重复沉淀同一段，完成后日志记下进度；
  协调器没有建任务时立即记下
- 退役与删除：退役话题的日志保留到最后一次沉淀完成再删除，其间照样记下与交出本话题生成的记忆；
  没有未完成沉淀的退役与手动删除立即删除日志
- 本话题生成的记忆不取 lease：折叠等待摘要、沉淀等待接纳期间记下的生成的记忆在写回之后仍在
- 感知关闭时无副作用
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.work_queue import WorkQueueCapacityError, WorkQueueStoppedError
from hivememory.config.patchouli import PerceptionEngineConfig
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import WorkspaceAssetRef, system_actor_for_workspace
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TriggerReason
from hivememory.memory_library.adapters.short_term import JsonlShortTermStorage
from hivememory.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
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
    tmp_path,
    *,
    max_resident=5,
    idle_timeout=30,
    engine_config=None,
    relay=None,
    bus=None,
    clock=None,
    engine="default",
    events: RecordingRuntimeEventSink | None = None,
):
    """构造 (familiar, store, working_set, bus) 测试组合。

    Engine（含 RelayController）/ Store / WorkingSet / Journal 为真实对象；
    时间使用可控时钟。
    """
    clock = clock or _FakeClock()
    store = ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))
    working_set = TopicWorkingSet(max_resident=max_resident, clock=clock)
    if relay is None:
        relay = Mock()
        relay.generate_summary = AsyncMock(return_value="test summary")
    bus = bus or Mock()
    bus.request = AsyncMock(return_value=None)
    if engine == "default":
        engine = MemoryPerceptionEngine(
            config=engine_config or PerceptionEngineConfig(fold_token_threshold=999999),
            relay_controller=relay,
        )
    familiar = PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=working_set,
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=idle_timeout),
        runtime_events=RuntimeEventPublisher(events) if events is not None else None,
    )
    return familiar, store, working_set, bus


def _log_exists(tmp_path, topic_id: str) -> bool:
    return (tmp_path / "topics" / f"{topic_id}.jsonl").exists()


def _topic_events(events: RecordingRuntimeEventSink) -> list[tuple[str, str | None, dict]]:
    """话题事件的类型、话题与 data，按发出顺序。"""
    return [
        (event.event_type, event.topic_id, event.data)
        for event in events.events
        if event.event_type.startswith("topic.")
    ]


# ========== apply_interaction：核心摄入用例 ==========


class TestApplyInteraction:
    @pytest.mark.asyncio
    async def test_new_topic_creates_topic_and_tracks_residency(self, tmp_path):
        familiar, store, working_set, bus = _make_familiar(tmp_path)
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
        # 驻留工作集记录了话题
        assert working_set.select_lru_candidate(scope.workspace_identity) == topic_id

    @pytest.mark.asyncio
    async def test_new_topic_uses_submitted_title_and_summary(self, tmp_path):
        """提交携带的新话题标题与摘要用于建立话题，后续路由依赖它们。"""
        familiar, store, _, _ = _make_familiar(tmp_path)
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
    async def test_existing_topic_appends_blocks_and_dedupes_bindings(self, tmp_path):
        familiar, store, _, _ = _make_familiar(tmp_path)
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
    async def test_busy_topic_rejects_interaction_without_writing(self, tmp_path):
        familiar, store, working_set, _ = _make_familiar(tmp_path)
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
    async def test_unknown_target_is_rejected(self, tmp_path):
        familiar, _, _, _ = _make_familiar(tmp_path)

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
    async def test_pool_full_evicts_least_recently_used_topic(self, tmp_path):
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, max_resident=2, clock=clock)
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
        assert task.retires_topic is True

    @pytest.mark.asyncio
    async def test_busy_candidate_is_skipped_for_older_leased_topic(self, tmp_path):
        clock = _FakeClock()
        familiar, store, working_set, _ = _make_familiar(tmp_path, max_resident=2, clock=clock)
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
    async def test_admission_failure_preserves_topic_and_propagates(self, tmp_path):
        familiar, store, _, bus = _make_familiar(tmp_path, max_resident=1)
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
    async def test_pool_full_with_all_candidates_busy_raises(self, tmp_path):
        familiar, _, working_set, _ = _make_familiar(tmp_path, max_resident=1)
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


def _fold_every_block_config() -> PerceptionEngineConfig:
    """每个 block 都超过阈值、只保留最近 1 块：第二块写入后即有可折叠前缀。"""
    return PerceptionEngineConfig(fold_token_threshold=1, fold_retain_recent_blocks=1)


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
    async def test_apply_writes_block_without_compacting_over_threshold_topic(self, tmp_path):
        """应用只写入 block；话题超过阈值也不在应用内折叠。"""
        familiar, store, _, _ = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        await _apply_to_topic(familiar, scope, "t1", "a", "b")

        topic = store.get(scope.workspace_identity, "t1")
        assert [block.user_message for block in topic.blocks] == ["a", "b"]
        assert topic.fold_summary == ""
        assert familiar.needs_compact(scope.workspace_identity, "t1") is True

    @pytest.mark.asyncio
    async def test_compact_topic_appends_fold_records_and_keeps_interactions(self, tmp_path):
        """折叠只推进工作视图：交互记录不移除，每次折叠追加一条记录，摘要在上一次之上累积。"""
        familiar, store, _, bus = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        for message in ("a", "b", "c"):
            await _apply_to_topic(familiar, scope, "t1", message)
            await familiar.compact_topic(scope.workspace_identity, "t1")

        topic = store.get(scope.workspace_identity, "t1")
        first, second, third = topic.interactions
        assert [block.user_message for block in topic.interactions] == ["a", "b", "c"]
        assert [fold.folded_block_ids for fold in topic.folds] == [
            (first.block_id,),
            (second.block_id,),
        ]
        assert [block.user_message for block in topic.blocks] == ["c"]
        # 第二次折叠收到第一次的摘要作为 previous_summary，累积成链
        assert topic.fold_summary == "folded:1|folded:1"
        assert topic.total_tokens == third.total_tokens
        assert familiar.needs_compact(scope.workspace_identity, "t1") is False
        # 折叠只压缩上下文，不触发记忆生成，也不驱逐话题
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_writes_recorded_while_summarizing_survive_the_fold(self, tmp_path):
        """记下生成的记忆不取 lease：折叠等待摘要期间记下的，折叠写回之后仍在话题上。"""
        summarizing = asyncio.Event()
        release = asyncio.Event()

        async def summarize(blocks_to_fold, previous_summary=None):
            summarizing.set()
            await release.wait()
            return "summary"

        relay = Mock()
        relay.generate_summary = AsyncMock(side_effect=summarize)
        familiar, store, _, _ = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=relay
        )
        scope = _identity_scope()
        workspace = scope.workspace_identity
        store.create(workspace, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")

        compaction = asyncio.create_task(familiar.compact_topic(workspace, "t1"))
        await asyncio.wait_for(summarizing.wait(), timeout=1)
        await familiar.record_generated_memories(
            "t1", belong_to=workspace, generated=frozenset({"m"}), absorbed_into={}
        )
        release.set()
        await compaction

        topic = store.get(workspace, "t1")
        assert (topic.fold_summary, topic.topic_generated_memory_ids) == ("summary", {"m"})

    @pytest.mark.asyncio
    async def test_failed_compaction_leaves_topic_intact_and_can_compact_again(self, tmp_path):
        """摘要失败时话题原样保留并释放占用，之后的折叠照常完成。"""
        relay = Mock()
        relay.generate_summary = AsyncMock(
            side_effect=[RuntimeError("relay down"), "folded-summary"]
        )
        familiar, store, _, _ = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=relay
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
        assert [block.user_message for block in topic.blocks] == ["b"]
        assert topic.fold_summary == "folded-summary"

    @pytest.mark.asyncio
    async def test_compact_topic_rejects_topic_held_by_another_use_case(self, tmp_path):
        """话题正被占用（例如结算中）时折叠以 busy 拒绝，留待重试，不改动话题。"""
        familiar, store, working_set, _ = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")
        lease = working_set.acquire(scope.workspace_identity, "t1")
        assert lease is not None

        with pytest.raises(TopicBusyError, match="正忙"):
            await familiar.compact_topic(scope.workspace_identity, "t1")

        working_set.release(lease)
        assert store.get(scope.workspace_identity, "t1").fold_summary == ""

    @pytest.mark.asyncio
    async def test_compact_topic_ends_without_reviving_settled_topic(self, tmp_path):
        """折叠开始前话题已被结算删除：直接结束，不把话题写回。"""
        familiar, store, _, bus = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
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
    async def test_blocks_within_retain_count_need_no_compaction(self, tmp_path):
        """超过阈值但块数不多于保留数时没有可折叠前缀：不需要折叠，折叠也不改动话题。"""
        familiar, store, _, _ = _make_familiar(
            tmp_path,
            engine_config=PerceptionEngineConfig(
                fold_token_threshold=1, fold_retain_recent_blocks=5
            ),
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
        assert data.fold_summary == ""
        assert len(data.blocks) == 1


# ========== manual_settle / evict 具名用例 ==========


class TestNamedUseCases:
    @pytest.mark.asyncio
    async def test_manual_settle_admits_task_then_deletes_topic(self, tmp_path):
        events = RecordingRuntimeEventSink()
        familiar, store, _, bus = _make_familiar(tmp_path, events=events)
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
        route, task = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert task.retires_topic is True
        assert _topic_events(events) == [
            ("topic.created", topic_id, {"created_by": "interaction"}),
            (
                "topic.settled",
                topic_id,
                {
                    "reason": TriggerReason.MANUAL_SETTLE.value,
                    "generation_submitted": True,
                    "retired": True,
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_manual_settle_retires_settled_topic_without_generation(self, tmp_path):
        """空闲时已沉淀完成、之后没有新交互的话题，手动结算直接退役，不再生成，日志立即删除。"""
        events = RecordingRuntimeEventSink()
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(
            tmp_path, idle_timeout=30, clock=clock, events=events
        )
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == [topic_id]

        result = await familiar.manual_settle_topic(scope.workspace_identity, topic_id)

        assert result.generation_submitted is False
        assert result.generation_task_id is None
        assert bus.request.await_count == 1  # 只有空闲时的那次沉淀
        assert store.get(scope.workspace_identity, topic_id) is None
        assert not _log_exists(tmp_path, topic_id)  # 沉淀都已完成，日志随退役删除
        assert _topic_events(events)[-1] == (
            "topic.settled",
            topic_id,
            {
                "reason": TriggerReason.MANUAL_SETTLE.value,
                "generation_submitted": False,
                "retired": True,
            },
        )

    @pytest.mark.asyncio
    async def test_manual_settle_busy_topic_raises_busy_error(self, tmp_path):
        familiar, store, working_set, _ = _make_familiar(tmp_path)
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
    async def test_manual_settle_unknown_topic_raises_not_found(self, tmp_path):
        familiar, _, _, _ = _make_familiar(tmp_path)

        with pytest.raises(ResourceNotFoundError):
            await familiar.manual_settle_topic(_identity_scope().workspace_identity, "missing")

    @pytest.mark.asyncio
    async def test_evict_topic_removes_topic_and_log_without_settlement(self, tmp_path):
        events = RecordingRuntimeEventSink()
        familiar, store, working_set, bus = _make_familiar(tmp_path, events=events)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        result = await familiar.evict_topic(scope.workspace_identity, topic_id)

        await familiar.record_settled(
            topic_id, belong_to=scope.workspace_identity, interaction_count=1
        )

        assert result.removed is True
        assert store.get(scope.workspace_identity, topic_id) is None
        assert working_set.select_lru_candidate(scope.workspace_identity) is None
        assert not _log_exists(tmp_path, topic_id)  # 之后到达的沉淀完成没有日志可记
        bus.request.assert_not_awaited()  # evict 不触发结算
        assert [event_type for event_type, _, _ in _topic_events(events)] == [
            "topic.created",
            "topic.deleted",
        ]

    @pytest.mark.asyncio
    async def test_evict_busy_topic_reports_not_removed(self, tmp_path):
        familiar, store, working_set, _ = _make_familiar(tmp_path)
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
    @pytest.mark.parametrize(
        "failure", [WorkQueueCapacityError("memory", 1), WorkQueueStoppedError("stopped")]
    )
    async def test_manual_settle_admission_failure_keeps_topic_intact_and_allows_retry(
        self, tmp_path, failure
    ):
        events = RecordingRuntimeEventSink()
        familiar, store, _, bus = _make_familiar(tmp_path, events=events)
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
        assert [b.user_message for b in after.blocks] == [b.user_message for b in before.blocks]
        assert after.fold_summary == before.fold_summary
        assert "topic.settled" not in [event_type for event_type, _, _ in _topic_events(events)]

        # 修复 admission 后可再次重试并成功结束生命周期
        bus.request = AsyncMock(return_value=None)
        result = await familiar.manual_settle_topic(scope.workspace_identity, topic_id)
        assert result.topic_id == topic_id
        assert store.get(scope.workspace_identity, topic_id) is None


# ========== 空闲维护 ==========


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_scan_idle_settles_timed_out_topic_and_keeps_it_in_pool(self, tmp_path):
        """空闲沉淀之后话题仍在存储与驻留中，访问时间不变；协调器没有建任务时没有可等的完成，
        立即记下覆盖到的交互数。"""
        events = RecordingRuntimeEventSink()
        clock = _FakeClock()
        familiar, store, working_set, bus = _make_familiar(
            tmp_path, idle_timeout=30, clock=clock, events=events
        )
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

        flushed = await familiar.scan_idle_topics_once()

        assert flushed == [stale_id]
        assert store.get(scope.workspace_identity, stale_id).settled_interaction_count == 1
        assert store.get(scope.workspace_identity, fresh_id).settled_interaction_count == 0
        # 访问时间没有被沉淀刷新：话题仍在驻留集合中，也仍是空闲候选。
        assert working_set.list_idle_candidates(30) == [(scope.workspace_identity, stale_id)]
        route, settlement = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert settlement.reason == TriggerReason.IDLE_TIMEOUT
        assert settlement.retires_topic is False
        assert _topic_events(events)[-1] == (
            "topic.settled",
            stale_id,
            {
                "reason": TriggerReason.IDLE_TIMEOUT.value,
                "generation_submitted": True,
                "retired": False,
            },
        )

    @pytest.mark.asyncio
    async def test_writes_recorded_while_awaiting_admission_survive_the_settlement(self, tmp_path):
        """沉淀等待生成队列接纳期间记下的生成的记忆，接纳之后仍在话题上。"""
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, idle_timeout=30, clock=clock)
        scope = _identity_scope()
        workspace = scope.workspace_identity
        topic_id = await familiar.apply_interaction(
            _payload("1"),
            belong_to=workspace,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        admitting = asyncio.Event()
        release = asyncio.Event()

        async def admit(route, payload):
            admitting.set()
            await release.wait()
            return _accept_settlement_task(route, payload)

        bus.request = AsyncMock(side_effect=admit)
        clock.advance(100)

        scan = asyncio.create_task(familiar.scan_idle_topics_once())
        await asyncio.wait_for(admitting.wait(), timeout=1)
        await familiar.record_generated_memories(
            topic_id, belong_to=workspace, generated=frozenset({"m"}), absorbed_into={}
        )
        release.set()

        assert await scan == [topic_id]
        assert store.get(workspace, topic_id).topic_generated_memory_ids == {"m"}

    @pytest.mark.asyncio
    async def test_scan_idle_skips_leased_topic(self, tmp_path):
        clock = _FakeClock()
        familiar, store, working_set, _ = _make_familiar(tmp_path, idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        working_set.acquire(scope.workspace_identity, topic_id)

        flushed = await familiar.scan_idle_topics_once()

        assert flushed == []
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_scan_idle_admission_failure_skips_and_preserves(self, tmp_path):
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        flushed = await familiar.scan_idle_topics_once()

        # idle 维护不向上传播；话题与沉淀进度不变，下一轮重试
        assert flushed == []
        assert store.get(scope.workspace_identity, topic_id).has_unsettled_interactions is True
        bus.request = AsyncMock(return_value=None)
        assert await familiar.scan_idle_topics_once() == [topic_id]

    @pytest.mark.asyncio
    async def test_scan_idle_settles_again_only_after_new_interactions(self, tmp_path):
        """同一段空闲不重复沉淀；话题接收新交互后再次空闲，沉淀覆盖到最新一轮。"""
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, idle_timeout=30, clock=clock)
        scope = _identity_scope()
        topic_id = await familiar.apply_interaction(
            _payload("1"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == [topic_id]

        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == []
        assert bus.request.await_count == 1

        await _apply_to_topic(familiar, scope, topic_id, "2")
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == [topic_id]

        assert bus.request.await_count == 2
        settlement = bus.request.await_args.args[1]
        assert [block.user_message for block in settlement.interactions] == ["1", "2"]
        assert store.get(scope.workspace_identity, topic_id).settled_interaction_count == 2

    @pytest.mark.asyncio
    async def test_scan_idle_retires_empty_topic_without_generation(self, tmp_path):
        """为写入预先建立、从没接收过交互的空话题，空闲时直接退役，不生成。"""
        events = RecordingRuntimeEventSink()
        clock = _FakeClock()
        familiar, store, working_set, bus = _make_familiar(
            tmp_path, idle_timeout=30, clock=clock, events=events
        )
        scope = _identity_scope()
        topic_id = await familiar.prepare_topic("NEW_TOPIC", None, None, scope.workspace_identity)

        clock.advance(100)
        flushed = await familiar.scan_idle_topics_once()

        assert flushed == [topic_id]
        bus.request.assert_not_awaited()
        assert store.get(scope.workspace_identity, topic_id) is None
        assert working_set.select_lru_candidate(scope.workspace_identity) is None
        assert not _log_exists(tmp_path, topic_id)
        assert _topic_events(events)[-1] == (
            "topic.settled",
            topic_id,
            {
                "reason": TriggerReason.IDLE_TIMEOUT.value,
                "generation_submitted": False,
                "retired": True,
            },
        )

    @pytest.mark.asyncio
    async def test_idle_settlement_after_folding_carries_folded_turns_with_working_view(
        self, tmp_path
    ):
        """折叠过的话题沉淀时，任务携带全部交互与折叠记录，生成材料仍是工作视图。"""
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(
            tmp_path,
            idle_timeout=30,
            clock=clock,
            max_resident=2,
            engine_config=PerceptionEngineConfig(
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

        folded = store.get(scope.workspace_identity, topic_id)
        assert folded.fold_summary == "folded:1|folded:1"
        assert [block.user_message for block in folded.blocks] == ["question-2-" * 80]
        assert bus.request.await_count == 0
        bus.request.side_effect = _accept_settlement_task
        clock.advance(100)

        assert await familiar.scan_idle_topics_once() == [topic_id]

        bus.request.assert_awaited_once()
        route, settlement = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT
        assert [block.user_message for block in settlement.interactions] == [
            f"question-{i}-" * 80 for i in range(3)
        ]
        assert len(settlement.folds) == 2
        assert settlement.fold_summary == "folded:1|folded:1"
        assert [block.user_message for block in settlement.blocks] == ["question-2-" * 80]


# ========== 沉淀进度与退役话题的日志 ==========


class TestSettlementProgress:
    @pytest.mark.asyncio
    async def test_accepted_settlement_stays_in_flight_until_generation_completes(self, tmp_path):
        """接纳之后只记在途：日志中的进度不动，在途期间空闲扫描不再沉淀同一段；生成完成后日志
        记下进度、在途清除，此后接收新交互的话题再次空闲时照常沉淀。"""
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, idle_timeout=30, clock=clock)
        bus.request.side_effect = _accept_settlement_task
        scope = _identity_scope()
        workspace = scope.workspace_identity
        topic_id = await familiar.apply_interaction(
            _payload("1"), belong_to=workspace, from_actor=scope.actor_identity
        )
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == [topic_id]
        in_flight = store.get(workspace, topic_id)
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == []

        await familiar.record_settled(topic_id, belong_to=workspace, interaction_count=1)
        reloaded = ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))
        reloaded.load()
        await _apply_to_topic(familiar, scope, topic_id, "2")
        clock.advance(100)

        assert in_flight.settled_interaction_count == 0
        assert reloaded.get(workspace, topic_id).settled_interaction_count == 1
        assert await familiar.scan_idle_topics_once() == [topic_id]
        assert bus.request.await_count == 2

    @pytest.mark.asyncio
    async def test_retired_topic_keeps_its_log_until_the_last_settlement_completes(self, tmp_path):
        """退役随沉淀被接纳时话题离开话题池，日志保留，排在前面的沉淀生成的记忆照样记、照样
        读；这次沉淀完成时记下进度并删除日志。"""
        familiar, store, _, bus = _make_familiar(tmp_path)
        bus.request.side_effect = _accept_settlement_task
        scope = _identity_scope()
        workspace = scope.workspace_identity
        topic_id = await familiar.apply_interaction(
            _payload("1"), belong_to=workspace, from_actor=scope.actor_identity
        )

        await familiar.manual_settle_topic(workspace, topic_id)
        await familiar.record_generated_memories(
            topic_id, belong_to=workspace, generated=frozenset({"m"}), absorbed_into={}
        )
        generated_while_retired = await familiar.get_generated_memories(
            topic_id, belong_to=workspace
        )
        kept = _log_exists(tmp_path, topic_id)
        await familiar.record_settled(topic_id, belong_to=workspace, interaction_count=1)

        assert (store.get(workspace, topic_id), kept) == (None, True)
        assert generated_while_retired == {"m"}
        assert not _log_exists(tmp_path, topic_id)
        assert await familiar.get_generated_memories(topic_id, belong_to=workspace) is None

    @pytest.mark.asyncio
    async def test_retiring_while_a_settlement_is_in_flight_waits_for_it(self, tmp_path):
        """在途的沉淀已经覆盖全部交互时退役不再生成，日志保留到这次在途的沉淀完成。"""
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, idle_timeout=30, clock=clock)
        bus.request.side_effect = _accept_settlement_task
        scope = _identity_scope()
        workspace = scope.workspace_identity
        topic_id = await familiar.apply_interaction(
            _payload("1"), belong_to=workspace, from_actor=scope.actor_identity
        )
        clock.advance(100)
        await familiar.scan_idle_topics_once()

        result = await familiar.manual_settle_topic(workspace, topic_id)
        kept = _log_exists(tmp_path, topic_id)
        await familiar.record_settled(topic_id, belong_to=workspace, interaction_count=1)

        assert (result.generation_submitted, bus.request.await_count) == (False, 1)
        assert (store.get(workspace, topic_id), kept) == (None, True)
        assert not _log_exists(tmp_path, topic_id)


# ========== 感知关闭（engine=None）==========


class TestDisabledPerception:
    @pytest.mark.asyncio
    async def test_disabled_engine_has_no_side_effects(self, tmp_path):
        familiar, store, _, _ = _make_familiar(tmp_path, engine=None)
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
async def test_manual_settle_unknown_failure_keeps_topic_and_propagates(tmp_path, failure):
    """未知接纳结果与取消不改判明确拒绝；保留材料并释放占用，供后续核实。"""
    familiar, store, _, bus = _make_familiar(tmp_path)
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
