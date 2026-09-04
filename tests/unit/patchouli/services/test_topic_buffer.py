"""TopicBufferService 领域服务单元测试。

测试覆盖:
- 触发策略: 七种 TriggerReason 的矩阵映射、settle => evict 不变量、TriggerPlan 校验
- 状态机: IDLE/PROCESSING/SETTLING 的合法与非法转换、busy 隔离
- apply_interaction: block 追加、binding 幂等、元数据写回
- compact: 预约恢复、retain 约束、无可折叠前缀 no-op、Relay 失败恢复
- settle 协议: 所有 settle 来源共用 begin -> admission -> complete/abort；
  admission 失败保留内容并恢复 IDLE，无材料正常移除
- Topic Pool: LRU 候选选择、idle/shutdown 扫描、Workspace 隔离
"""

from unittest.mock import Mock

import pytest

from hivememory.core.models import (
    BufferState,
    Identity,
    LogicalBlock,
    TurnRecord,
    WorkspaceAssetRef,
)
from hivememory.engines.perception.models import TriggerReason
from hivememory.patchouli.errors import TopicBusyError
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.topic_buffer import (
    TRIGGER_PLANS,
    SettlementOutcome,
    SettlementStatus,
    TopicBufferService,
    TriggerPlan,
    resolve_trigger_plan,
)
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1", workspace_id="main_workspace"):
    return make_identity_scope(user_id=user_id, workspace_id=workspace_id)


def _block(text="q", *, tokens=10, worth_saving=None, user_id="u1") -> LogicalBlock:
    turn = TurnRecord(
        identity=Identity(user_id=user_id, agent_id="a1"),
        user_query=text,
        assistant_final_text="a",
    )
    return LogicalBlock(turn=turn, total_tokens=tokens, worth_saving=worth_saving)


def _make_service(*, relay=None, store=None) -> tuple[TopicBufferService, ShortTermMemoryStore]:
    store = store or ShortTermMemoryStore()
    relay = relay or Mock()
    service = TopicBufferService(store=store, relay_controller=relay)
    return service, store


def _fill_topic(store, scope, topic_id, blocks, *, state=BufferState.IDLE):
    """把话题填充为给定内容并写回（测试夹具）。"""
    topic = store.get(scope, topic_id, touch=False)
    store.put(
        topic.model_copy(
            update={
                "blocks": tuple(blocks),
                "total_tokens": sum(b.total_tokens for b in blocks),
                "state": state,
            }
        )
    )


# ========== 触发策略：唯一决策矩阵 ==========


class TestTriggerPlan:
    def test_matrix_covers_exactly_seven_reasons(self):
        assert set(TRIGGER_PLANS) == set(TriggerReason)

    @pytest.mark.parametrize(
        "reason,plan",
        [
            (TriggerReason.TOKEN_OVERFLOW, TriggerPlan(compact=True)),
            (TriggerReason.IDLE_TIMEOUT, TriggerPlan(settle=True, evict=True)),
            (TriggerReason.LRU_EVICTION, TriggerPlan(settle=True, evict=True)),
            (TriggerReason.SHUTDOWN, TriggerPlan(settle=True, evict=True)),
            (TriggerReason.MANUAL_SETTLE, TriggerPlan(settle=True, evict=True)),
            (TriggerReason.MANUAL_COMPACT, TriggerPlan(compact=True)),
            (TriggerReason.MANUAL_DELETE, TriggerPlan(evict=True)),
        ],
    )
    def test_matrix_columns(self, reason, plan):
        assert TRIGGER_PLANS[reason] == plan

    def test_resolve_trigger_plan_returns_same_plan_instance(self):
        for reason, plan in TRIGGER_PLANS.items():
            assert resolve_trigger_plan(reason) is plan

    def test_resolve_unknown_reason_raises(self):
        with pytest.raises(ValueError, match="未知的触发原因"):
            resolve_trigger_plan("not-a-reason")

    def test_settle_requires_evict(self):
        with pytest.raises(ValueError, match="settle=True requires evict=True"):
            TriggerPlan(settle=True)

    def test_empty_plan_rejected(self):
        with pytest.raises(ValueError, match="at least one action"):
            TriggerPlan()

    def test_evict_only_is_legal(self):
        plan = TriggerPlan(evict=True)
        assert plan.evict and not plan.settle and not plan.compact


# ========== 状态预约与 Interaction 写入 ==========


class TestReservationAndApplyInteraction:
    def test_reserve_requires_idle_topic(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)

        assert service.reserve_processing(scope, topic.topic_id) is True
        assert (
            store.get(scope, topic.topic_id, touch=False).state
            is BufferState.PROCESSING
        )
        # 已预约的话题不能重复预约。
        assert service.reserve_processing(scope, topic.topic_id) is False

    def test_reserve_missing_topic_returns_false(self):
        service, _ = _make_service()
        assert service.reserve_processing(_identity_scope(), "ghost") is False

    def test_release_processing_restores_idle(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        service.reserve_processing(scope, topic.topic_id)

        service.release_processing(scope, topic.topic_id)

        assert store.get(scope, topic.topic_id, touch=False).state is BufferState.IDLE

    def test_apply_interaction_requires_processing(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)

        with pytest.raises(TopicBusyError, match="PROCESSING"):
            service.apply_interaction(scope, topic.topic_id, _block())

    def test_apply_interaction_appends_block_and_updates_tokens(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        service.reserve_processing(scope, topic.topic_id)

        updated = service.apply_interaction(scope, topic.topic_id, _block("q1", tokens=7))

        assert [b.turn.user_query for b in updated.blocks] == ["q1"]
        assert updated.total_tokens == 7
        assert store.get(scope, topic.topic_id, touch=False).total_tokens == 7

    def test_apply_interaction_binding_is_idempotent_per_asset(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        service.reserve_processing(scope, topic.topic_id)
        ref = WorkspaceAssetRef(token="asset-token")
        block = _block()

        service.apply_interaction(
            scope, topic.topic_id, block,
            interaction_id="i1",
            asset_id_and_refs=(("asset-1", ref),),
        )
        service.apply_interaction(
            scope, topic.topic_id, block,
            interaction_id="i2",
            asset_id_and_refs=(("asset-1", ref), ("asset-2", ref)),
        )

        bindings = store.get(scope, topic.topic_id, touch=False).bindings
        assert [b.asset_id for b in bindings] == ["asset-1", "asset-2"]
        # 首次使用事实不被重复提交改写。
        assert bindings[0].first_bound_interaction_id == "i1"

    def test_apply_interaction_binding_requires_interaction_id(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        service.reserve_processing(scope, topic.topic_id)

        with pytest.raises(ValueError, match="interaction_id"):
            service.apply_interaction(
                scope, topic.topic_id, _block(),
                asset_id_and_refs=(("asset-1", WorkspaceAssetRef(token="t")),),
            )

    def test_apply_interaction_writeback_is_snapshot_isolated(self):
        """两次读取返回相互独立的快照，调用方持有引用不影响 Store 内事实。"""
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        service.reserve_processing(scope, topic.topic_id)

        first = service.apply_interaction(scope, topic.topic_id, _block())
        second = store.get(scope, topic.topic_id, touch=False)

        assert first is not second
        assert second.block_count == 1
        assert second.total_tokens == first.total_tokens


# ========== Compact ==========


class TestCompact:
    def test_manual_compact_requires_idle(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        service.reserve_processing(scope, topic.topic_id)

        with pytest.raises(TopicBusyError, match="manual compact"):
            service.handle_trigger(
                scope, topic.topic_id, TriggerReason.MANUAL_COMPACT,
                retain_recent_blocks=1,
            )

    def test_compact_rejects_missing_retain_argument(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)

        with pytest.raises(ValueError, match="requires retain_recent_blocks"):
            service.handle_trigger(
                scope, topic.topic_id, TriggerReason.MANUAL_COMPACT
            )

    @pytest.mark.parametrize("retain", [0, -1])
    def test_compact_rejects_non_positive_retain(self, retain):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)

        with pytest.raises(ValueError, match="retain_recent_blocks must be >= 1"):
            service.handle_trigger(
                scope, topic.topic_id, TriggerReason.MANUAL_COMPACT,
                retain_recent_blocks=retain,
            )

    def test_compact_is_noop_when_no_foldable_prefix(self):
        relay = Mock()
        service, store = _make_service(relay=relay)
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block("q1"), _block("q2")])

        execution = service.handle_trigger(
            scope, topic.topic_id, TriggerReason.MANUAL_COMPACT,
            retain_recent_blocks=2,
        )

        assert execution.compacted is False
        relay.generate_summary.assert_not_called()

    def test_manual_compact_writes_summary_and_restores_idle(self):
        relay = Mock()
        relay.generate_summary.return_value = "folded summary"
        service, store = _make_service(relay=relay)
        scope = _identity_scope()
        topic = store.create(scope)
        blocks = [_block(f"q{i}", tokens=20) for i in range(5)]
        _fill_topic(store, scope, topic.topic_id, blocks)

        execution = service.handle_trigger(
            scope, topic.topic_id, TriggerReason.MANUAL_COMPACT,
            retain_recent_blocks=2,
        )

        assert execution.compacted is True
        folded = relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [b.turn.user_query for b in folded] == ["q0", "q1", "q2"]
        final = store.get(scope, topic.topic_id, touch=False)
        assert final.state is BufferState.IDLE
        assert final.state_summary == "folded summary"
        assert [b.turn.user_query for b in final.blocks] == ["q3", "q4"]
        assert final.total_tokens == 40

    def test_token_overflow_compact_keeps_processing_reservation(self):
        """TOKEN_OVERFLOW 复用调用方的 PROCESSING 预约，不自行释放。"""
        relay = Mock()
        relay.generate_summary.return_value = "folded"
        service, store = _make_service(relay=relay)
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(
            store, scope, topic.topic_id,
            [_block("q0"), _block("q1")],
            state=BufferState.PROCESSING,
        )

        execution = service.handle_trigger(
            scope, topic.topic_id, TriggerReason.TOKEN_OVERFLOW,
            retain_recent_blocks=1,
        )

        assert execution.compacted is True
        final = store.get(scope, topic.topic_id, touch=False)
        assert final.state is BufferState.PROCESSING
        assert [b.turn.user_query for b in final.blocks] == ["q1"]

    def test_manual_compact_failure_restores_idle(self):
        relay = Mock()
        relay.generate_summary.side_effect = RuntimeError("summary boom")
        service, store = _make_service(relay=relay)
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block("q0"), _block("q1")])

        with pytest.raises(RuntimeError, match="summary boom"):
            service.handle_trigger(
                scope, topic.topic_id, TriggerReason.MANUAL_COMPACT,
                retain_recent_blocks=1,
            )

        final = store.get(scope, topic.topic_id, touch=False)
        assert final.state is BufferState.IDLE
        assert final.state_summary == ""


# ========== 统一 settle 协议 ==========


class TestSettlementProtocol:
    @pytest.mark.parametrize(
        "reason",
        [
            TriggerReason.IDLE_TIMEOUT,
            TriggerReason.LRU_EVICTION,
            TriggerReason.SHUTDOWN,
            TriggerReason.MANUAL_SETTLE,
        ],
    )
    def test_all_settle_sources_share_same_protocol(self, reason):
        """所有 settle=True 来源共用 begin -> admission -> complete 时序。"""
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block("q1")])

        # handle_trigger 是唯一计划执行入口，对 settle 只完成 begin。
        execution = service.handle_trigger(scope, topic.topic_id, reason)
        assert execution.settlement is not None
        assert execution.settlement.task is not None
        assert store.get(scope, topic.topic_id, touch=False).state is BufferState.SETTLING

        # 模拟 queue 接纳后的 complete。
        outcome = service.complete_settlement(
            scope, topic.topic_id, generation_task_id="task-1", reason=reason
        )
        assert outcome.status is SettlementStatus.ACCEPTED
        assert outcome.removed is True
        assert outcome.generation_task_id == "task-1"
        assert store.get(scope, topic.topic_id, touch=False) is None

    @pytest.mark.parametrize(
        "reason",
        [
            TriggerReason.IDLE_TIMEOUT,
            TriggerReason.LRU_EVICTION,
            TriggerReason.SHUTDOWN,
            TriggerReason.MANUAL_SETTLE,
        ],
    )
    def test_admission_failure_restores_idle_for_all_sources(self, reason):
        """明确 admission 失败时，Topic 对所有触发来源均保留并恢复 IDLE。"""
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block("q1")])

        reservation = service.begin_settlement(scope, topic.topic_id, reason)
        outcome = service.abort_settlement(scope, topic.topic_id, reason=reason)

        assert reservation.task is not None
        assert outcome.status is SettlementStatus.REJECTED
        assert outcome.removed is False
        final = store.get(scope, topic.topic_id, touch=False)
        assert final.state is BufferState.IDLE
        assert [b.turn.user_query for b in final.blocks] == ["q1"]

    def test_settlement_is_rejected_for_busy_topic(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block()], state=BufferState.PROCESSING)

        with pytest.raises(TopicBusyError):
            service.begin_settlement(scope, topic.topic_id, TriggerReason.IDLE_TIMEOUT)

    def test_begin_missing_topic_returns_none(self):
        service, _ = _make_service()
        assert (
            service.begin_settlement(_identity_scope(), "ghost", TriggerReason.IDLE_TIMEOUT)
            is None
        )

    def test_complete_without_task_is_no_material_success(self):
        """没有可生成材料是正常成功：Topic 同样结束生命周期。"""
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)

        reservation = service.begin_settlement(scope, topic.topic_id, TriggerReason.IDLE_TIMEOUT)
        outcome = service.complete_settlement(scope, topic.topic_id, reason=TriggerReason.IDLE_TIMEOUT)

        assert reservation.task is None
        assert outcome.status is SettlementStatus.NO_MATERIAL
        assert outcome.removed is True
        assert store.get(scope, topic.topic_id, touch=False) is None

    def test_summary_only_topic_is_no_material(self):
        """summary-only Topic 没有 worth-saving block：按矩阵视为无材料正常完成。"""
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        summary_topic = store.get(scope, topic.topic_id, touch=False)
        store.put(summary_topic.model_copy(update={"state_summary": "已折叠历史内容"}))

        reservation = service.begin_settlement(
            scope, topic.topic_id, TriggerReason.IDLE_TIMEOUT
        )
        outcome = service.complete_settlement(scope, topic.topic_id, reason=TriggerReason.IDLE_TIMEOUT)

        # from_topic_data 契约：无 worth-saving block 即 no-material。
        assert reservation.task is None
        assert outcome.status is SettlementStatus.NO_MATERIAL
        assert outcome.removed is True
        assert store.get(scope, topic.topic_id, touch=False) is None

    def test_complete_and_abort_require_settling_state(self):
        """complete/abort 只作用于仍处于 SETTLING 状态的话题。"""
        service, store = _make_service()
        scope = _identity_scope()

        missing = service.complete_settlement(scope, "ghost")
        assert missing.status is SettlementStatus.NOT_FOUND
        assert missing.removed is False

        topic = store.create(scope)
        aborted_outcome = service.abort_settlement(scope, topic.topic_id)
        assert aborted_outcome.status is SettlementStatus.NOT_FOUND
        # abort 未取得预约，话题仍为 IDLE。
        assert store.get(scope, topic.topic_id, touch=False).state is BufferState.IDLE

    def test_worth_saving_filtered_blocks_produce_no_material(self):
        """全部 block 均为 worth_saving=False 时无可生成材料。"""
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block("q", worth_saving=False)])

        reservation = service.begin_settlement(scope, topic.topic_id, TriggerReason.MANUAL_SETTLE)

        assert reservation.task is None

    def test_worth_saving_none_blocks_are_kept(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block("kept", worth_saving=None)])

        reservation = service.begin_settlement(scope, topic.topic_id, TriggerReason.MANUAL_SETTLE)

        assert reservation.task is not None
        assert [b.turn.user_query for b in reservation.task.blocks] == ["kept"]

    def test_settlement_outcome_invariants(self):
        with pytest.raises(ValueError, match="ACCEPTED"):
            SettlementOutcome(
                topic_id="t", status=SettlementStatus.ACCEPTED, removed=True,
                generation_task_id=None,
            )
        with pytest.raises(ValueError, match="removed=True"):
            SettlementOutcome(
                topic_id="t", status=SettlementStatus.REJECTED, removed=True,
            )


# ========== Topic Pool ==========


class TestTopicPool:
    def test_select_lru_candidate_skips_busy_and_excluded(self):
        service, store = _make_service()
        scope = _identity_scope()
        old = store.create(scope)
        busy = store.create(scope)
        newest = store.create(scope)
        _fill_topic(store, scope, busy.topic_id, [_block()], state=BufferState.PROCESSING)

        # 构造访问顺序: old < newest。
        old_topic = store.get(scope, old.topic_id, touch=False)
        store.put(old_topic.model_copy(update={"last_accessed_at": 100.0}))
        new_topic = store.get(scope, newest.topic_id, touch=False)
        store.put(new_topic.model_copy(update={"last_accessed_at": 200.0}))

        candidate = service.select_lru_candidate(scope)
        assert candidate == old.topic_id
        # busy 与已尝试候选被排除后返回剩余最久未访问者。
        candidate = service.select_lru_candidate(
            scope, exclude_ids={old.topic_id}
        )
        assert candidate == newest.topic_id
        # 全部排除后无候选。
        candidate = service.select_lru_candidate(
            scope, exclude_ids={old.topic_id, newest.topic_id}
        )
        assert candidate is None

    def test_select_lru_candidate_returns_none_without_idle_topics(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        _fill_topic(store, scope, topic.topic_id, [_block()], state=BufferState.PROCESSING)

        assert service.select_lru_candidate(scope) is None

    def test_list_idle_candidates_filters_state_and_timeout(self):
        service, store = _make_service()
        scope = _identity_scope()
        stale = store.create(scope)
        fresh = store.create(scope)
        processing = store.create(scope)
        stale_topic = store.get(scope, stale.topic_id, touch=False)
        store.put(stale_topic.model_copy(update={"last_update": 1.0}))
        proc_topic = store.get(scope, processing.topic_id, touch=False)
        store.put(
            proc_topic.model_copy(update={"last_update": 1.0, "state": BufferState.PROCESSING})
        )

        candidates = service.list_idle_candidates(idle_timeout_seconds=60)

        topic_ids = {c.topic_id for c in candidates}
        # stale 为 IDLE 且超时；fresh 刚创建；processing 虽超时但被状态过滤。
        assert stale.topic_id in topic_ids
        assert fresh.topic_id not in topic_ids
        assert processing.topic_id not in topic_ids

    def test_list_shutdown_candidates_include_all_states(self):
        service, store = _make_service()
        scope = _identity_scope()
        idle = store.create(scope)
        processing = store.create(scope)
        _fill_topic(store, scope, processing.topic_id, [_block()], state=BufferState.PROCESSING)

        candidates = service.list_shutdown_candidates()

        assert {c.topic_id for c in candidates} == {idle.topic_id, processing.topic_id}
        assert all(c.block_count >= 0 for c in candidates)

    def test_candidate_scope_keeps_workspace_isolation(self):
        """两个 Workspace 的同名/不同名 Topic 不串扰。"""
        service, store = _make_service()
        scope_a = _identity_scope(user_id="u1", workspace_id="ws-a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws-b")

        topic_a = service.create_topic(scope_a, topic_title="A")
        topic_b = service.create_topic(scope_b, topic_title="B")

        assert topic_a.topic_id != topic_b.topic_id  # 全局唯一
        assert service.get_topic(scope_a, topic_b.topic_id) is None
        assert service.get_topic(scope_b, topic_a.topic_id) is None
        assert service.count_topics(scope_a) == 1
        assert service.count_topics(scope_b) == 1

    def test_discard_if_empty_keeps_summary_only_topic(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)
        summary_topic = store.get(scope, topic.topic_id, touch=False)
        store.put(summary_topic.model_copy(update={"state_summary": "历史内容"}))

        assert service.discard_if_empty(scope, topic.topic_id) is False
        assert store.get(scope, topic.topic_id, touch=False) is not None

    def test_discard_if_empty_removes_truly_empty_topic(self):
        service, store = _make_service()
        scope = _identity_scope()
        topic = store.create(scope)

        assert service.discard_if_empty(scope, topic.topic_id) is True
        assert store.get(scope, topic.topic_id, touch=False) is None

    def test_ensure_topic_rejects_unknown_target(self):
        service, _ = _make_service()
        scope = _identity_scope()

        with pytest.raises(KeyError, match="does not exist"):
            service.ensure_topic(scope, "ghost-topic")
