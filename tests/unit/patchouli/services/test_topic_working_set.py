"""TopicWorkingSet 驻留工作集单元测试。

测试覆盖:
- LRU 顺序: touch 刷新排序、候选按最久未访问选择、Workspace 隔离
- 占用权: acquire 互斥（含同 Workspace 不同执行者）、过期令牌释放被忽略
- 候选查询: idle 超时过滤、占用过滤、显式 now、shutdown 全量返回
- 容量: needs_eviction 的上限判断与 Workspace 隔离
- 移除: remove 仅清除驻留、不清理 lease
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

        ws.touch(scope, "topic-1")
        clock.advance(10)
        ws.touch(scope, "topic-2")

        assert ws.select_lru_candidate(scope) == "topic-1"

    def test_retouch_moves_topic_to_lru_tail(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        clock.advance(10)
        ws.touch(scope, "topic-2")
        clock.advance(10)
        ws.touch(scope, "topic-1")  # 重新访问后 topic-1 不再是最旧

        assert ws.select_lru_candidate(scope) == "topic-2"

    def test_select_ignores_other_workspaces(self):
        ws, _ = _make_working_set()
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a, "topic-a1")
        ws.touch(scope_b, "topic-b1")

        assert ws.select_lru_candidate(scope_a) == "topic-a1"
        assert ws.select_lru_candidate(scope_b) == "topic-b1"
        # 无任何驻留话题的 Workspace 没有候选
        scope_c = _identity_scope(user_id="u3", workspace_id="ws_c")
        assert ws.select_lru_candidate(scope_c) is None

    def test_select_returns_none_for_empty_working_set(self):
        ws, _ = _make_working_set()

        assert ws.select_lru_candidate(_identity_scope()) is None

    def test_select_respects_exclude(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        ws.touch(scope, "topic-2")

        assert ws.select_lru_candidate(scope, exclude={"topic-1"}) == "topic-2"
        assert ws.select_lru_candidate(scope, exclude={"topic-1", "topic-2"}) is None


# ========== 占用权（lease） ==========


class TestLease:
    def test_acquire_is_exclusive_until_release(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        lease = ws.acquire(scope, "topic-1")
        assert lease is not None
        assert ws.acquire(scope, "topic-1") is None  # 占用期间互斥

        ws.release(lease)
        assert ws.acquire(scope, "topic-1") is not None  # 释放后可再次获取

    def test_acquire_is_exclusive_across_actor_scopes_in_same_workspace(self):
        ws, _ = _make_working_set()
        scope_agent_a = _identity_scope(user_id="u1", agent_id="agent-a")
        scope_agent_b = _identity_scope(user_id="u1", agent_id="agent-b")

        assert ws.acquire(scope_agent_a, "topic-1") is not None
        # 同一 Workspace 内不同执行者作用域指向同一个话题，必须互斥
        assert ws.acquire(scope_agent_b, "topic-1") is None

    def test_acquire_allows_non_resident_topic(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        # 新创建的话题在被 touch 前也必须可以占用（apply_interaction 主线路径）
        assert ws.acquire(scope, "brand-new-topic") is not None

    def test_select_skips_leased_topic_and_recovers_after_release(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        ws.touch(scope, "topic-2")

        lease = ws.acquire(scope, "topic-1")
        assert ws.select_lru_candidate(scope) == "topic-2"  # topic-1 被占用，跳过

        ws.release(lease)
        assert ws.select_lru_candidate(scope) == "topic-1"  # 释放后恢复候选资格

    def test_release_ignores_stale_token(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        first = ws.acquire(scope, "topic-1")
        ws.release(first)
        second = ws.acquire(scope, "topic-1")
        ws.release(first)  # 重复释放过期令牌，不得清掉后来者的租约

        assert second is not None
        assert ws.acquire(scope, "topic-1") is None


# ========== 容量判断 ==========


class TestCapacity:
    def test_needs_eviction_when_workspace_reaches_capacity(self):
        ws, _ = _make_working_set(max_resident=2)
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        assert ws.needs_eviction(scope) is False

        ws.touch(scope, "topic-2")
        assert ws.needs_eviction(scope) is True

    def test_needs_eviction_is_isolated_per_workspace(self):
        ws, _ = _make_working_set(max_resident=2)
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a, "topic-a1")
        ws.touch(scope_a, "topic-a2")

        assert ws.needs_eviction(scope_a) is True
        assert ws.needs_eviction(scope_b) is False  # 其他 Workspace 不受影响

    def test_init_rejects_invalid_capacity(self):
        with pytest.raises(ValueError, match="max_resident"):
            TopicWorkingSet(max_resident=0)


# ========== idle / shutdown 候选查询 ==========


class TestCandidateQueries:
    def test_idle_candidates_only_include_timeout_topics(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        clock.advance(150)
        ws.touch(scope, "topic-2")

        candidates = ws.list_idle_candidates(timeout_seconds=100)
        assert [topic_id for _, topic_id in candidates] == ["topic-1"]
        # 返回的 scope 保持最后访问时冻结的执行作用域
        assert candidates[0][0].workspace_identity.workspace_id == "main_workspace"

        clock.advance(100)  # topic-2 恰好达到超时阈值，仍未空闲（与 is_idle 的严格大于一致）
        assert [topic_id for _, topic_id in ws.list_idle_candidates(timeout_seconds=100)] == [
            "topic-1"
        ]

    def test_idle_candidates_exclude_leased_topics(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        clock.advance(50)
        ws.touch(scope, "topic-2")
        clock.advance(200)  # 两个话题均已超时

        ws.acquire(scope, "topic-1")

        assert [topic_id for _, topic_id in ws.list_idle_candidates(timeout_seconds=100)] == [
            "topic-2"
        ]

    def test_idle_candidates_honor_explicit_now(self):
        ws, clock = _make_working_set()
        scope = _identity_scope()

        touched_at = clock.now
        ws.touch(scope, "topic-1")
        clock.advance(200)

        # 显式 now 早于真实流逝时间时不得误判为空闲
        assert ws.list_idle_candidates(timeout_seconds=100, now=touched_at + 50) == []
        assert [
            topic_id
            for _, topic_id in ws.list_idle_candidates(timeout_seconds=100, now=touched_at + 150)
        ] == ["topic-1"]

    def test_idle_candidates_cover_all_workspaces(self):
        ws, clock = _make_working_set()
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a, "topic-a1")
        ws.touch(scope_b, "topic-b1")
        clock.advance(500)

        candidates = ws.list_idle_candidates(timeout_seconds=100)
        assert {
            (scope.workspace_identity.workspace_id, topic_id) for scope, topic_id in candidates
        } == {
            ("ws_a", "topic-a1"),
            ("ws_b", "topic-b1"),
        }

    def test_candidates_reflect_last_touch_scope(self):
        ws, clock = _make_working_set()
        scope_agent_a = _identity_scope(user_id="u1", agent_id="agent-a")
        scope_agent_b = _identity_scope(user_id="u1", agent_id="agent-b")

        ws.touch(scope_agent_a, "topic-1")
        ws.touch(scope_agent_b, "topic-1")  # 最后访问的执行作用域生效
        clock.advance(10)

        ((idle_scope, _),) = ws.list_idle_candidates(timeout_seconds=5)
        assert idle_scope.actor_identity.agent_id == "agent-b"
        ((shutdown_scope, _),) = ws.list_shutdown_candidates()
        assert shutdown_scope.actor_identity.agent_id == "agent-b"

    def test_idle_candidates_on_empty_working_set(self):
        ws, _ = _make_working_set()

        assert ws.list_idle_candidates(timeout_seconds=0) == []

    def test_shutdown_candidates_include_all_resident_topics(self):
        ws, _ = _make_working_set()
        scope_a = _identity_scope(user_id="u1", workspace_id="ws_a")
        scope_b = _identity_scope(user_id="u2", workspace_id="ws_b")

        ws.touch(scope_a, "topic-a1")
        ws.touch(scope_b, "topic-b1")
        ws.acquire(scope_a, "topic-a1")  # 被占用的话题同样在 shutdown 清理范围内

        candidates = ws.list_shutdown_candidates()
        assert {
            (scope.workspace_identity.workspace_id, topic_id) for scope, topic_id in candidates
        } == {
            ("ws_a", "topic-a1"),
            ("ws_b", "topic-b1"),
        }


# ========== 驻留移除 ==========


class TestRemoval:
    def test_remove_drops_residency_but_keeps_lease(self):
        ws, _ = _make_working_set()
        scope = _identity_scope()

        ws.touch(scope, "topic-1")
        ws.acquire(scope, "topic-1")
        ws.remove(scope, "topic-1")

        assert ws.list_shutdown_candidates() == []  # 驻留已移除
        assert ws.acquire(scope, "topic-1") is None  # lease 不随 remove 清理

    def test_remove_unknown_topic_is_silent(self):
        ws, _ = _make_working_set()

        ws.remove(_identity_scope(), "missing")

        assert ws.list_shutdown_candidates() == []
