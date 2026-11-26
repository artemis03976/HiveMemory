"""PerceptionFamiliar 感知链路集成测试（lease 编排架构）。

真实协作：PerceptionFamiliar + MemoryPerceptionEngine + TopicWorkingSet
+ ShortTermMemoryStore（话题日志写在临时目录）；Relay（LLM 摘要）与
Bus（Generation 队列）是边界之外协作者，使用 Mock，时间使用可控时钟。

测试覆盖:
- receive_interaction: 新建（含提交携带的标题与摘要）/指定话题写入、binding 幂等、busy 拒绝、未知目标
- LRU 驱逐: 池满驱逐最久未访问话题、busy 候选改选、admission 失败传播
- 折叠: 应用不在内部折叠、needs_fold 判断、折叠追加折叠记录且不移除交互记录、摘要累积、
  失败后话题完整且可再次折叠、占用时 busy 拒绝、话题已退役时直接结束、未超保留块数不折叠
- settle_and_retire_topic / delete_topic 具名用例，含 admission 失败可重试、已沉淀的话题退役不生成
- scan_idle_topics_once：空闲只沉淀，话题留在话题池、访问时间不变，没有新交互不再沉淀；
  从没接收过交互的空话题空闲时直接退役；折叠后沉淀携带被折叠的轮次
- 沉淀进度在生成完成时推进：接纳后只记在途，在途期间不重复沉淀同一段，完成后日志记下进度；
  协调器没有建任务时立即记下
- 退役与删除：退役话题的日志保留到最后一次沉淀完成再删除，其间照样记下与交出本话题生成的记忆；
  没有未完成沉淀的退役与手动删除立即删除日志
- 本话题生成的记忆不取 lease：折叠等待摘要、沉淀等待接纳期间记下的生成的记忆在写回之后仍在
- 进展记录：停顿超过阈值且有还没整理的交互时排入写一条，写一条只看旧进展记录与还没整理的
  交互，写下之后没有新交互不再排入；失败后同一位置不重试，有新交互后再写；写一条不取 lease，
  生成期间话题退役时这一条不记
- 感知关闭时无副作用
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.work_queue import WorkQueueCapacityError, WorkQueueStoppedError
from hivememory.config.patchouli import GenerationTranscriptConfig, PerceptionEngineConfig
from hivememory.core.errors import ModelProviderError, ResourceNotFoundError
from hivememory.core.models import WorkspaceAssetRef, system_actor_for_workspace
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TriggerReason
from hivememory.engines.perception.progress_record import ProgressRecordWriter
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
from tests.helpers.llm import DEFAULT_LLM_PARAMS, CallbackLLMService, ScriptedLLMService
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
    pause_seconds=30,
    progress_llm=None,
):
    """构造 (familiar, store, working_set, bus) 测试组合。

    Engine（含 RelayController 与写进展记录的 ProgressRecordWriter）/ Store / WorkingSet / Journal
    为真实对象；模型是边界外的替身，没有给出 ``progress_llm`` 时不写进展记录；时间使用可控时钟。
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
        progress_writer = None
        if progress_llm is not None:
            progress_writer = ProgressRecordWriter(
                progress_llm,
                DEFAULT_LLM_PARAMS,
                timeout_seconds=60,
                transcript_config=GenerationTranscriptConfig(),
            )
        engine = MemoryPerceptionEngine(
            config=engine_config or PerceptionEngineConfig(fold_token_threshold=999999),
            fold_summary_writer=relay,
            progress_writer=progress_writer,
        )
    familiar = PerceptionFamiliar(
        engine=engine,
        store=store,
        working_set=working_set,
        bus=bus,
        config=SimpleNamespace(
            idle_timeout_seconds=idle_timeout, progress_record_pause_seconds=pause_seconds
        ),
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


# ========== receive_interaction：接收一轮交互 ==========


class TestApplyInteraction:
    @pytest.mark.asyncio
    async def test_new_topic_creates_topic_and_tracks_residency(self, tmp_path):
        familiar, store, working_set, bus = _make_familiar(tmp_path)
        scope = _identity_scope()

        topic_id = await familiar.receive_interaction(
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

        topic_id = await familiar.receive_interaction(
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

        first = await familiar.receive_interaction(
            _payload("a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="t-fixed",
            interaction_id="i-1",
            asset_refs=(ref,),
        )
        second = await familiar.receive_interaction(
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
            await familiar.receive_interaction(
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
            await familiar.receive_interaction(
                _payload(),
                belong_to=_identity_scope().workspace_identity,
                from_actor=_identity_scope().actor_identity,
                target_topic_id="missing",
            )


# ========== LRU 驱逐 ==========


class TestLruRetirement:
    @pytest.mark.asyncio
    async def test_pool_full_retires_least_recently_used_topic(self, tmp_path):
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(tmp_path, max_resident=2, clock=clock)
        scope = _identity_scope()
        t1 = await familiar.receive_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)
        t2 = await familiar.receive_interaction(
            _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)

        t3 = await familiar.receive_interaction(
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
        t1 = await familiar.receive_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)
        t2 = await familiar.receive_interaction(
            _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        clock.advance(10)
        lease = working_set.acquire(scope.workspace_identity, t1)
        assert lease is not None

        t3 = await familiar.receive_interaction(
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
        t1 = await familiar.receive_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        bus.request = AsyncMock(side_effect=RuntimeError("admission boom"))

        with pytest.raises(RuntimeError, match="admission boom"):
            await familiar.receive_interaction(
                _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
            )

        # admission 失败：话题内容完整保留，可重试
        assert store.get(scope.workspace_identity, t1) is not None
        assert store.get(scope.workspace_identity, t1).block_count == 1

    @pytest.mark.asyncio
    async def test_pool_full_with_all_candidates_busy_raises(self, tmp_path):
        familiar, _, working_set, _ = _make_familiar(tmp_path, max_resident=1)
        scope = _identity_scope()
        t1 = await familiar.receive_interaction(
            _payload("1"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
        )
        working_set.acquire(scope.workspace_identity, t1)  # 唯一候选正被占用

        with pytest.raises(TopicBusyError, match="全部话题正被占用"):
            await familiar.receive_interaction(
                _payload("2"), belong_to=scope.workspace_identity, from_actor=scope.actor_identity
            )


# ========== 折叠（compact） ==========


def _fold_every_block_config() -> PerceptionEngineConfig:
    """每个 block 都超过阈值、只保留最近 1 块：第二块写入后即有可折叠前缀。"""
    return PerceptionEngineConfig(fold_token_threshold=1, fold_retain_recent_blocks=1)


async def _apply_to_topic(familiar, scope, topic_id: str, *messages: str) -> None:
    for message in messages:
        await familiar.receive_interaction(
            _payload(message),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id=topic_id,
            interaction_id=f"i-{topic_id}-{message}",
        )


class TestFold:
    @pytest.mark.asyncio
    async def test_receive_writes_block_without_folding_over_threshold_topic(self, tmp_path):
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
        assert familiar.needs_fold(scope.workspace_identity, "t1") is True

    @pytest.mark.asyncio
    async def test_fold_topic_appends_fold_records_and_keeps_interactions(self, tmp_path):
        """折叠只推进工作视图：交互记录不移除，每次折叠追加一条记录，摘要在上一次之上累积。"""
        familiar, store, _, bus = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")

        for message in ("a", "b", "c"):
            await _apply_to_topic(familiar, scope, "t1", message)
            await familiar.fold_topic(scope.workspace_identity, "t1")

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
        assert familiar.needs_fold(scope.workspace_identity, "t1") is False
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

        compaction = asyncio.create_task(familiar.fold_topic(workspace, "t1"))
        await asyncio.wait_for(summarizing.wait(), timeout=1)
        await familiar.record_generated_memories(
            "t1", belong_to=workspace, generated=frozenset({"m"}), absorbed_into={}
        )
        release.set()
        await compaction

        topic = store.get(workspace, "t1")
        assert (topic.fold_summary, topic.topic_generated_memory_ids) == ("summary", {"m"})

    @pytest.mark.asyncio
    async def test_failed_fold_leaves_topic_intact_and_can_fold_again(self, tmp_path):
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
            await familiar.fold_topic(scope.workspace_identity, "t1")

        assert store.get(scope.workspace_identity, "t1") == before
        await familiar.fold_topic(scope.workspace_identity, "t1")
        topic = store.get(scope.workspace_identity, "t1")
        assert [block.user_message for block in topic.blocks] == ["b"]
        assert topic.fold_summary == "folded-summary"

    @pytest.mark.asyncio
    async def test_fold_topic_rejects_topic_held_by_another_use_case(self, tmp_path):
        """话题正被占用（例如沉淀中）时折叠以 busy 拒绝，留待重试，不改动话题。"""
        familiar, store, working_set, _ = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")
        lease = working_set.acquire(scope.workspace_identity, "t1")
        assert lease is not None

        with pytest.raises(TopicBusyError, match="正忙"):
            await familiar.fold_topic(scope.workspace_identity, "t1")

        working_set.release(lease)
        assert store.get(scope.workspace_identity, "t1").fold_summary == ""

    @pytest.mark.asyncio
    async def test_fold_topic_ends_without_reviving_retired_topic(self, tmp_path):
        """折叠开始前话题已被沉淀并退役：直接结束，不把话题写回。"""
        familiar, store, _, bus = _make_familiar(
            tmp_path, engine_config=_fold_every_block_config(), relay=_folding_relay()
        )
        scope = _identity_scope()
        store.create(scope.workspace_identity, topic_id="t1")
        await _apply_to_topic(familiar, scope, "t1", "a", "b")
        bus.request.side_effect = _accept_settlement_task
        await familiar.settle_and_retire_topic(scope.workspace_identity, "t1")

        await familiar.fold_topic(scope.workspace_identity, "t1")

        assert store.get(scope.workspace_identity, "t1") is None
        assert familiar.needs_fold(scope.workspace_identity, "t1") is False

    @pytest.mark.asyncio
    async def test_blocks_within_retain_count_need_no_fold(self, tmp_path):
        """超过阈值但块数不多于保留数时没有可折叠前缀：不需要折叠，折叠也不改动话题。"""
        familiar, store, _, _ = _make_familiar(
            tmp_path,
            engine_config=PerceptionEngineConfig(
                fold_token_threshold=1, fold_retain_recent_blocks=5
            ),
        )
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload("q", "a"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )

        assert familiar.needs_fold(scope.workspace_identity, topic_id) is False
        await familiar.fold_topic(scope.workspace_identity, topic_id)
        data = store.get(scope.workspace_identity, topic_id)
        assert data.fold_summary == ""
        assert len(data.blocks) == 1


# ========== manual_settle / evict 具名用例 ==========


class TestNamedUseCases:
    @pytest.mark.asyncio
    async def test_settle_and_retire_admits_task_then_retires_topic(self, tmp_path):
        events = RecordingRuntimeEventSink()
        familiar, store, _, bus = _make_familiar(tmp_path, events=events)
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        accepted = _generation_task("task-1", topic_id)
        bus.request = AsyncMock(return_value=accepted)

        result = await familiar.settle_and_retire_topic(scope.workspace_identity, topic_id)

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
    async def test_settle_and_retire_retires_settled_topic_without_generation(self, tmp_path):
        """空闲时已沉淀完成、之后没有新交互的话题，手动沉淀并退役时直接退役，不再生成，日志立即删除。"""
        events = RecordingRuntimeEventSink()
        clock = _FakeClock()
        familiar, store, _, bus = _make_familiar(
            tmp_path, idle_timeout=30, clock=clock, events=events
        )
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == [topic_id]

        result = await familiar.settle_and_retire_topic(scope.workspace_identity, topic_id)

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
    async def test_settle_and_retire_busy_topic_raises_busy_error(self, tmp_path):
        familiar, store, working_set, _ = _make_familiar(tmp_path)
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        working_set.acquire(scope.workspace_identity, topic_id)

        with pytest.raises(TopicBusyError):
            await familiar.settle_and_retire_topic(scope.workspace_identity, topic_id)

        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    async def test_settle_and_retire_unknown_topic_raises_not_found(self, tmp_path):
        familiar, _, _, _ = _make_familiar(tmp_path)

        with pytest.raises(ResourceNotFoundError):
            await familiar.settle_and_retire_topic(_identity_scope().workspace_identity, "missing")

    @pytest.mark.asyncio
    async def test_delete_topic_removes_topic_and_log_without_settling(self, tmp_path):
        events = RecordingRuntimeEventSink()
        familiar, store, working_set, bus = _make_familiar(tmp_path, events=events)
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        result = await familiar.delete_topic(scope.workspace_identity, topic_id)

        await familiar.record_settled(topic_id, belong_to=scope.workspace_identity, settled_until=1)

        assert result.removed is True
        assert store.get(scope.workspace_identity, topic_id) is None
        assert working_set.select_lru_candidate(scope.workspace_identity) is None
        assert not _log_exists(tmp_path, topic_id)  # 之后到达的沉淀完成没有日志可记
        bus.request.assert_not_awaited()  # 删除不沉淀
        assert [event_type for event_type, _, _ in _topic_events(events)] == [
            "topic.created",
            "topic.deleted",
        ]

    @pytest.mark.asyncio
    async def test_delete_busy_topic_reports_not_removed(self, tmp_path):
        familiar, store, working_set, _ = _make_familiar(tmp_path)
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        working_set.acquire(scope.workspace_identity, topic_id)

        result = await familiar.delete_topic(scope.workspace_identity, topic_id)

        assert result.removed is False
        assert store.get(scope.workspace_identity, topic_id) is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure", [WorkQueueCapacityError("memory", 1), WorkQueueStoppedError("stopped")]
    )
    async def test_settle_and_retire_admission_failure_keeps_topic_intact_and_allows_retry(
        self, tmp_path, failure
    ):
        events = RecordingRuntimeEventSink()
        familiar, store, _, bus = _make_familiar(tmp_path, events=events)
        scope = _identity_scope()
        topic_id = await familiar.receive_interaction(
            _payload("question", "answer"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id="NEW_TOPIC",
        )
        before = store.get(scope.workspace_identity, topic_id)
        bus.request = AsyncMock(side_effect=failure)

        with pytest.raises(TopicSettleAdmissionError, match="可重试"):
            await familiar.settle_and_retire_topic(scope.workspace_identity, topic_id)

        after = store.get(scope.workspace_identity, topic_id)
        assert after is not None
        assert [b.user_message for b in after.blocks] == [b.user_message for b in before.blocks]
        assert after.fold_summary == before.fold_summary
        assert "topic.settled" not in [event_type for event_type, _, _ in _topic_events(events)]

        # 修复 admission 后可再次重试并成功结束生命周期
        bus.request = AsyncMock(return_value=None)
        result = await familiar.settle_and_retire_topic(scope.workspace_identity, topic_id)
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
        stale_id = await familiar.receive_interaction(
            _payload("1"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        fresh_id = await familiar.receive_interaction(
            _payload("2"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-2",
        )

        flushed = await familiar.scan_idle_topics_once()

        assert flushed == [stale_id]
        assert store.get(scope.workspace_identity, stale_id).settled_until == 1
        assert store.get(scope.workspace_identity, fresh_id).settled_until == 0
        # 访问时间没有被沉淀刷新：话题仍在驻留集合中，也仍是空闲候选。
        assert working_set.list_inactive(30) == [(scope.workspace_identity, stale_id)]
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
        topic_id = await familiar.receive_interaction(
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
        topic_id = await familiar.receive_interaction(
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
        topic_id = await familiar.receive_interaction(
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
        topic_id = await familiar.receive_interaction(
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
        assert store.get(scope.workspace_identity, topic_id).settled_until == 2

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
            topic_id = await familiar.receive_interaction(
                _payload(f"question-{i}-" * 80, f"answer-{i}"),
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
                target_topic_id=topic_id,
            )
            await familiar.fold_topic(scope.workspace_identity, topic_id)

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
        topic_id = await familiar.receive_interaction(
            _payload("1"), belong_to=workspace, from_actor=scope.actor_identity
        )
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == [topic_id]
        in_flight = store.get(workspace, topic_id)
        clock.advance(100)
        assert await familiar.scan_idle_topics_once() == []

        await familiar.record_settled(topic_id, belong_to=workspace, settled_until=1)
        reloaded = ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))
        reloaded.load()
        await _apply_to_topic(familiar, scope, topic_id, "2")
        clock.advance(100)

        assert in_flight.settled_until == 0
        assert reloaded.get(workspace, topic_id).settled_until == 1
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
        topic_id = await familiar.receive_interaction(
            _payload("1"), belong_to=workspace, from_actor=scope.actor_identity
        )

        await familiar.settle_and_retire_topic(workspace, topic_id)
        await familiar.record_generated_memories(
            topic_id, belong_to=workspace, generated=frozenset({"m"}), absorbed_into={}
        )
        generated_while_retired = await familiar.get_generated_memories(
            topic_id, belong_to=workspace
        )
        kept = _log_exists(tmp_path, topic_id)
        await familiar.record_settled(topic_id, belong_to=workspace, settled_until=1)

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
        topic_id = await familiar.receive_interaction(
            _payload("1"), belong_to=workspace, from_actor=scope.actor_identity
        )
        clock.advance(100)
        await familiar.scan_idle_topics_once()

        result = await familiar.settle_and_retire_topic(workspace, topic_id)
        kept = _log_exists(tmp_path, topic_id)
        await familiar.record_settled(topic_id, belong_to=workspace, settled_until=1)

        assert (result.generation_submitted, bus.request.await_count) == (False, 1)
        assert (store.get(workspace, topic_id), kept) == (None, True)
        assert not _log_exists(tmp_path, topic_id)


# ========== 感知关闭（engine=None）==========


class _ScheduledEntries:
    """停顿扫描排入维护项的替身：记下排入的话题，总是排入成功。"""

    def __init__(self) -> None:
        self.topics: list[str] = []

    async def __call__(self, belong_to, topic_id) -> bool:
        self.topics.append(topic_id)
        return True


class TestProgressRecord:
    @pytest.mark.asyncio
    async def test_paused_topic_gets_entries_over_its_unrecorded_turns(self, tmp_path):
        """停顿且有还没整理的交互时排入；写一条只看旧进展记录与还没整理的交互，写下之后没有新交互
        就不再排入。没停顿的话题与空话题不排入。"""
        clock = _FakeClock()
        llm = ScriptedLLMService('{"entry": "讨论了发布顺序"}', '{"entry": "定下先发 staging"}')
        familiar, store, _, _ = _make_familiar(
            tmp_path, clock=clock, pause_seconds=30, progress_llm=llm
        )
        scope = _identity_scope()
        belong_to = scope.workspace_identity

        async def apply(message, topic_id="NEW_TOPIC"):
            return await familiar.receive_interaction(
                _payload(message),
                belong_to=belong_to,
                from_actor=scope.actor_identity,
                target_topic_id=topic_id,
                interaction_id=f"i-{message}",
            )

        paused = await apply("q1")
        await apply("q2", paused)
        await familiar.prepare_topic("NEW_TOPIC", None, None, belong_to)  # 空话题
        clock.advance(100)
        await apply("fresh")
        first_scan = _ScheduledEntries()
        assert await familiar.scan_paused_topics_once(first_scan) == [paused]
        await familiar.write_progress_entry(belong_to, paused)
        clock.advance(100)
        after_entry = _ScheduledEntries()
        await familiar.scan_paused_topics_once(after_entry)
        await apply("q3", paused)
        clock.advance(100)
        await familiar.scan_paused_topics_once(_ScheduledEntries())
        await familiar.write_progress_entry(belong_to, paused)

        topic = store.get(belong_to, paused)
        assert first_scan.topics == [paused]
        assert paused not in after_entry.topics
        assert [(entry.text, entry.covered_until) for entry in topic.progress_record] == [
            ("讨论了发布顺序", 2),
            ("定下先发 staging", 3),
        ]
        second_input = llm.calls[1].messages[1]["content"]
        assert second_input.startswith("[旧进展记录]\n- 讨论了发布顺序\n\n[新交互记录原文]\n")
        assert "[Turn 3]\n[User]: q3" in second_input
        assert "q1" not in second_input and "q2" not in second_input

    @pytest.mark.asyncio
    async def test_failed_entry_is_not_retried_until_the_topic_has_new_turns(self, tmp_path):
        """生成失败照常抛出、话题不变；同一位置不再排入，有了新交互后再写，覆盖到新的位置。"""
        clock = _FakeClock()
        llm = ScriptedLLMService(
            ModelProviderError("provider down"),
            ModelProviderError("provider down"),
            '{"entry": "讨论了发布顺序"}',
        )
        familiar, store, _, _ = _make_familiar(
            tmp_path, clock=clock, pause_seconds=30, progress_llm=llm
        )
        scope = _identity_scope()
        belong_to = scope.workspace_identity
        topic_id = await familiar.receive_interaction(
            _payload("q1"),
            belong_to=belong_to,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )
        clock.advance(100)
        assert await familiar.scan_paused_topics_once(_ScheduledEntries()) == [topic_id]

        with pytest.raises(ModelProviderError):
            await familiar.write_progress_entry(belong_to, topic_id)
        clock.advance(100)
        retried_at_same_position = await familiar.scan_paused_topics_once(_ScheduledEntries())
        await familiar.receive_interaction(
            _payload("q2"),
            belong_to=belong_to,
            from_actor=scope.actor_identity,
            target_topic_id=topic_id,
            interaction_id="i-2",
        )
        clock.advance(100)
        after_new_turn = await familiar.scan_paused_topics_once(_ScheduledEntries())
        await familiar.write_progress_entry(belong_to, topic_id)

        assert retried_at_same_position == []
        assert after_new_turn == [topic_id]
        assert [
            entry.covered_until for entry in store.get(belong_to, topic_id).progress_record
        ] == [2]

    @pytest.mark.asyncio
    async def test_topic_retired_while_writing_keeps_no_entry(self, tmp_path):
        """写一条不取 lease：生成期间手动沉淀并退役照常让话题退役，生成完成后这一条不记到退役话题的日志。"""
        generating = asyncio.Event()
        release = asyncio.Event()

        async def slow_entry(messages, params):
            generating.set()
            await release.wait()
            return '{"entry": "讨论了发布顺序"}'

        bus = Mock()
        familiar, store, _, bus = _make_familiar(
            tmp_path, bus=bus, progress_llm=CallbackLLMService(complete=slow_entry)
        )
        bus.request = AsyncMock(side_effect=_accept_settlement_task)
        scope = _identity_scope()
        belong_to = scope.workspace_identity
        topic_id = await familiar.receive_interaction(
            _payload("q1"),
            belong_to=belong_to,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        writing = asyncio.create_task(familiar.write_progress_entry(belong_to, topic_id))
        await asyncio.wait_for(generating.wait(), timeout=2)
        await familiar.settle_and_retire_topic(belong_to, topic_id)
        release.set()
        await asyncio.wait_for(writing, timeout=2)

        [(retired, _)] = (
            ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics")).load().retired
        )
        assert store.get(belong_to, topic_id) is None
        assert retired.topic_id == topic_id
        assert retired.progress_record == ()


class TestDisabledPerception:
    @pytest.mark.asyncio
    async def test_disabled_engine_has_no_side_effects(self, tmp_path):
        familiar, store, _, _ = _make_familiar(tmp_path, engine=None)
        scope = _identity_scope()

        topic_id = await familiar.receive_interaction(
            _payload(),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id="i-1",
        )

        assert topic_id == "NEW_TOPIC"
        assert store.list_all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [KeyError("handler bug"), asyncio.CancelledError()])
async def test_settle_and_retire_unknown_failure_keeps_topic_and_propagates(tmp_path, failure):
    """未知接纳结果与取消不改判明确拒绝；保留材料并释放占用，供后续核实。"""
    familiar, store, _, bus = _make_familiar(tmp_path)
    scope = _identity_scope()
    topic_id = await familiar.receive_interaction(
        _payload("question", "answer"),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        target_topic_id="NEW_TOPIC",
    )
    before = store.get(scope.workspace_identity, topic_id)
    bus.request = AsyncMock(side_effect=failure)

    with pytest.raises(type(failure)) as caught:
        await familiar.settle_and_retire_topic(scope.workspace_identity, topic_id)

    assert caught.value is failure
    assert store.get(scope.workspace_identity, topic_id) == before
    # 释放占用后仍可由拥有者操作话题；失败不能留下永久 busy 状态。
    result = await familiar.delete_topic(scope.workspace_identity, topic_id)
    assert result.removed is True
    assert store.get(scope.workspace_identity, topic_id) is None
