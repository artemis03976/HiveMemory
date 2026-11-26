"""TopicWorkingSet 单元测试：驻留话题的进程内记录与占用表。

测试覆盖:
- LRU 顺序: touch 刷新排序、候选按最久未访问选择、Workspace 隔离
- 占用权: acquire 互斥（含同 Workspace 不同执行者）、不要求驻留、过期令牌释放被忽略
- 候选查询: 停下多久的阈值过滤、占用过滤、显式 now
- 容量: needs_eviction 的上限判断与 Workspace 隔离
- 在途沉淀与写进展记录失败的位置: 只记最远的在途、刷新访问不改位置、只记在驻留话题上
- 移除: 驻留记录整条删除（访问时间与两个位置一起消失），不清理 lease
"""

import pytest

from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1", workspace_id="main_workspace", agent_id="test_agent"):
    return make_identity_scope(user_id=user_id, workspace_id=workspace_id, agent_id=agent_id)


class _FakeClock:
    """可控单调时钟：由测试手动推进，替代真实时间。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_working_set(max_resident=5):
    """返回 (工作集, 可控时钟)，时间相关行为不依赖真实时钟。"""
    clock = _FakeClock()
    return TopicWorkingSet(max_resident=max_resident, clock=clock), clock


# ========== LRU 顺序 ==========


class TestLruOrder:
    def test_select_returns_oldest_touched_topic(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        clock.advance(10)
        ws.touch(scope.workspace_identity, "topic-2")

        assert ws.select_lru_candidate(scope.workspace_identity) == "topic-1"

    def test_retouch_moves_topic_to_lru_tail(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        clock.advance(10)
        ws.touch(scope.workspace_identity, "topic-2")
        clock.advance(10)
        ws.touch(scope.workspace_identity, "topic-1")  # 重新访问后 topic-1 不再是最旧

        assert ws.select_lru_candidate(scope.workspace_identity) == "topic-2"

    def test_select_ignores_other_workspaces(self):
        ws, _ = _make_working_set()
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a.workspace_identity, "topic-a1")
        ws.touch(scope_b.workspace_identity, "topic-b1")

        assert ws.select_lru_candidate(scope_a.workspace_identity) == "topic-a1"
        assert ws.select_lru_candidate(scope_b.workspace_identity) == "topic-b1"
        # 无任何驻留话题的 Workspace 没有候选
        scope_c = _identity_scope(user_id="u3", workspace_id="ws_c")
        assert ws.select_lru_candidate(scope_c.workspace_identity) is None

    def test_select_returns_none_for_empty_working_set(self):
        ws, _ = _make_working_set()

        assert ws.select_lru_candidate(_identity_scope().workspace_identity) is None

    def test_select_respects_exclude(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        ws.touch(scope.workspace_identity, "topic-2")

        assert ws.select_lru_candidate(scope.workspace_identity, exclude={"topic-1"}) == "topic-2"
        assert (
            ws.select_lru_candidate(scope.workspace_identity, exclude={"topic-1", "topic-2"})
            is None
        )


# ========== 占用权（lease） ==========


class TestLease:
    def test_acquire_is_exclusive_until_release(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        lease = ws.acquire(scope.workspace_identity, "topic-1")
        assert lease is not None
        assert ws.acquire(scope.workspace_identity, "topic-1") is None  # 占用期间互斥

        ws.release(lease)
        assert ws.acquire(scope.workspace_identity, "topic-1") is not None  # 释放后可再次获取

    def test_acquire_is_exclusive_across_actor_scopes_in_same_workspace(self):
        ws, _ = _make_working_set()
        scope_agent_a = _identity_scope(user_id="u1", agent_id="agent-a")
        scope_agent_b = _identity_scope(user_id="u1", agent_id="agent-b")

        assert ws.acquire(scope_agent_a.workspace_identity, "topic-1") is not None
        # 同一 Workspace 内不同执行者作用域指向同一个话题，必须互斥
        assert ws.acquire(scope_agent_b.workspace_identity, "topic-1") is None

    def test_acquire_allows_non_resident_topic(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        # 新创建的话题在被 touch 前也必须可以占用（receive_interaction 主线路径）
        assert ws.acquire(scope.workspace_identity, "brand-new-topic") is not None

    def test_select_skips_leased_topic_and_recovers_after_release(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        ws.touch(scope.workspace_identity, "topic-2")

        lease = ws.acquire(scope.workspace_identity, "topic-1")
        assert (
            ws.select_lru_candidate(scope.workspace_identity) == "topic-2"
        )  # topic-1 被占用，跳过

        ws.release(lease)
        assert ws.select_lru_candidate(scope.workspace_identity) == "topic-1"  # 释放后恢复候选资格

    def test_release_ignores_stale_token(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        first = ws.acquire(scope.workspace_identity, "topic-1")
        ws.release(first)
        second = ws.acquire(scope.workspace_identity, "topic-1")
        ws.release(first)  # 重复释放过期令牌，不得清掉后来者的租约

        assert second is not None
        assert ws.acquire(scope.workspace_identity, "topic-1") is None


# ========== 容量判断 ==========


class TestCapacity:
    def test_needs_eviction_when_workspace_reaches_capacity(self):
        ws, _ = _make_working_set(max_resident=2)
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        assert ws.needs_eviction(scope.workspace_identity) is False

        ws.touch(scope.workspace_identity, "topic-2")
        assert ws.needs_eviction(scope.workspace_identity) is True

    def test_needs_eviction_is_isolated_per_workspace(self):
        ws, _ = _make_working_set(max_resident=2)
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a.workspace_identity, "topic-a1")
        ws.touch(scope_a.workspace_identity, "topic-a2")

        assert ws.needs_eviction(scope_a.workspace_identity) is True
        assert ws.needs_eviction(scope_b.workspace_identity) is False  # 其他 Workspace 不受影响

    def test_init_rejects_invalid_capacity(self):
        with pytest.raises(ValueError, match="max_resident"):
            TopicWorkingSet(max_resident=0)


# ========== 停下多久的候选查询 ==========


class TestCandidateQueries:
    def test_inactive_candidates_only_include_timeout_topics(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        clock.advance(150)
        ws.touch(scope.workspace_identity, "topic-2")

        candidates = ws.list_inactive(timeout_seconds=100)
        assert [topic_id for _, topic_id in candidates] == ["topic-1"]
        # 返回的坐标只携带归属，后台维护无需保存访问者。
        assert candidates[0][0].workspace_id == "main_workspace"

        clock.advance(100)  # topic-2 恰好达到阈值，还不算停下够久（严格大于）
        assert [topic_id for _, topic_id in ws.list_inactive(timeout_seconds=100)] == ["topic-1"]

    def test_inactive_candidates_exclude_leased_topics(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope.workspace_identity, "topic-1")
        clock.advance(50)
        ws.touch(scope.workspace_identity, "topic-2")
        clock.advance(200)  # 两个话题均已超时

        ws.acquire(scope.workspace_identity, "topic-1")

        assert [topic_id for _, topic_id in ws.list_inactive(timeout_seconds=100)] == ["topic-2"]

    def test_inactive_candidates_honor_explicit_now(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        touched_at = clock.now
        ws.touch(scope.workspace_identity, "topic-1")
        clock.advance(200)

        # 显式 now 早于真实流逝时间时不得误判为停下够久
        assert ws.list_inactive(timeout_seconds=100, now=touched_at + 50) == []
        assert [
            topic_id for _, topic_id in ws.list_inactive(timeout_seconds=100, now=touched_at + 150)
        ] == ["topic-1"]

    def test_inactive_candidates_cover_all_workspaces(self):
        ws, clock = _make_working_set()
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a.workspace_identity, "topic-a1")
        ws.touch(scope_b.workspace_identity, "topic-b1")
        clock.advance(500)

        candidates = ws.list_inactive(timeout_seconds=100)
        assert {(belong_to.workspace_id, topic_id) for belong_to, topic_id in candidates} == {
            ("ws_a", "topic-a1"),
            ("ws_b", "topic-b1"),
        }

    def test_candidates_keep_workspace_after_different_actors_touch(self):
        """不同执行者访问同一话题时，维护候选仍只携带归属。"""
        ws, clock = _make_working_set()
        scope_agent_a = _identity_scope(user_id="u1", agent_id="agent-a")
        scope_agent_b = _identity_scope(user_id="u1", agent_id="agent-b")

        ws.touch(scope_agent_a.workspace_identity, "topic-1")
        ws.touch(scope_agent_b.workspace_identity, "topic-1")
        clock.advance(10)

        assert ws.list_inactive(timeout_seconds=5) == [
            (scope_agent_a.workspace_identity, "topic-1")
        ]

    def test_inactive_candidates_on_empty_working_set(self):
        ws, _ = _make_working_set()

        assert ws.list_inactive(timeout_seconds=0) == []


# ========== 在途沉淀与写进展记录失败的位置 ==========


class TestPositions:
    def test_earlier_completion_keeps_the_furthest_settling_position(self):
        """同一话题先后接纳两次沉淀：较早的完成时更远的仍在途，更远的完成后才清除。"""
        ws, _ = _make_working_set()
        belong_to = _identity_scope().workspace_identity
        ws.touch(belong_to, "topic-1")
        ws.mark_settling(belong_to, "topic-1", 2)
        ws.mark_settling(belong_to, "topic-1", 5)

        ws.clear_settling(belong_to, "topic-1", settled_until=2)
        still_settling = ws.settling_until(belong_to, "topic-1")
        ws.clear_settling(belong_to, "topic-1", settled_until=5)

        assert (still_settling, ws.settling_until(belong_to, "topic-1")) == (5, 0)

    def test_touch_refreshes_access_but_keeps_both_positions(self):
        """再次访问只刷新访问时间：在途沉淀与写进展记录失败的位置照旧。"""
        ws, clock = _make_working_set()
        belong_to = _identity_scope().workspace_identity
        ws.touch(belong_to, "topic-1")
        ws.mark_settling(belong_to, "topic-1", 3)
        ws.mark_progress_failed(belong_to, "topic-1", 2)
        clock.advance(200)

        ws.touch(belong_to, "topic-1")

        assert ws.list_inactive(timeout_seconds=100) == []
        assert (
            ws.settling_until(belong_to, "topic-1"),
            ws.progress_failed_until(belong_to, "topic-1"),
        ) == (3, 2)

    def test_positions_are_only_kept_for_resident_topics(self):
        """不驻留的话题不会再被沉淀或写进展记录：两个位置都不记，读出 0。"""
        ws, _ = _make_working_set()
        belong_to = _identity_scope().workspace_identity

        ws.mark_settling(belong_to, "retired", 3)
        ws.mark_progress_failed(belong_to, "retired", 2)

        assert (
            ws.settling_until(belong_to, "retired"),
            ws.progress_failed_until(belong_to, "retired"),
        ) == (0, 0)


# ========== 驻留移除 ==========


class TestRemoval:
    def test_remove_drops_the_whole_resident_record_but_keeps_lease(self):
        """退役或删除后，访问时间、在途沉淀与写进展记录失败的位置一起消失；lease 仍由持有者释放。"""
        ws, _ = _make_working_set(max_resident=1)
        belong_to = _identity_scope().workspace_identity
        ws.touch(belong_to, "topic-1")
        ws.mark_settling(belong_to, "topic-1", 3)
        ws.mark_progress_failed(belong_to, "topic-1", 2)
        ws.acquire(belong_to, "topic-1")

        ws.remove(belong_to, "topic-1")

        assert ws.needs_eviction(belong_to) is False  # 驻留记录已删除
        assert (
            ws.settling_until(belong_to, "topic-1"),
            ws.progress_failed_until(belong_to, "topic-1"),
        ) == (0, 0)
        assert ws.acquire(belong_to, "topic-1") is None  # lease 不随 remove 清理

    def test_remove_unknown_topic_is_silent(self):
        ws, _ = _make_working_set(max_resident=1)
        scope = _identity_scope()
        ws.touch(scope.workspace_identity, "topic-1")

        ws.remove(scope.workspace_identity, "missing")

        assert ws.select_lru_candidate(scope.workspace_identity) == "topic-1"
