"""
PendingAtom 统一状态体系单元测试 (Commit 1)。

覆盖：
- PendingAtomStatus / PendingAtomResolution 枚举属性
- 状态机合法/非法迁移
- PendingAtomSnapshot 不变量校验
- map_legacy_status 兼容映射
- PendingAtomCache.snapshot() 派生路径（未结算 / SETTLED 各类 resolution）
"""

from __future__ import annotations

import pytest

from hivememory.alice.runtime.cache import PendingAtomCache
from hivememory.alice.runtime.pending_atom_state import (
    PendingAtomResolution,
    PendingAtomSnapshot,
    PendingAtomStatus,
    allowed_transitions,
    is_legal_transition,
    map_legacy_status,
)
from hivememory.core.models import Identity
from hivememory.engines.generation.models import PendingAtomSettlement

# ---------------------------------------------------------------------------
# Enum 属性
# ---------------------------------------------------------------------------


class TestPendingAtomStatus:
    def test_in_flight_states(self):
        assert PendingAtomStatus.PENDING.is_in_flight
        assert PendingAtomStatus.MATERIALIZING.is_in_flight
        assert not PendingAtomStatus.SETTLED.is_in_flight

    def test_terminal_states(self):
        assert PendingAtomStatus.SETTLED.is_terminal
        assert PendingAtomStatus.FAILED.is_terminal
        assert PendingAtomStatus.EXPIRED.is_terminal
        assert PendingAtomStatus.CANCELLED.is_terminal

    def test_in_flight_and_terminal_are_mutually_exclusive(self):
        for status in PendingAtomStatus:
            assert status.is_in_flight ^ status.is_terminal, (
                f"{status} must be either in-flight or terminal, not both/neither"
            )


class TestPendingAtomResolution:
    @pytest.mark.parametrize(
        "resolution",
        [
            PendingAtomResolution.CREATED,
            PendingAtomResolution.MERGED,
            PendingAtomResolution.UPDATED,
            PendingAtomResolution.TOUCHED,
        ],
    )
    def test_canonical_resolutions(self, resolution):
        assert resolution.has_canonical

    def test_discarded_has_no_canonical(self):
        assert not PendingAtomResolution.DISCARDED.has_canonical


# ---------------------------------------------------------------------------
# 状态机迁移
# ---------------------------------------------------------------------------


class TestStateMachine:
    @pytest.mark.parametrize(
        "src,dst",
        [
            (PendingAtomStatus.PENDING, PendingAtomStatus.MATERIALIZING),
            (PendingAtomStatus.PENDING, PendingAtomStatus.EXPIRED),
            (PendingAtomStatus.PENDING, PendingAtomStatus.CANCELLED),
            (PendingAtomStatus.MATERIALIZING, PendingAtomStatus.SETTLED),
            (PendingAtomStatus.MATERIALIZING, PendingAtomStatus.FAILED),
            (PendingAtomStatus.MATERIALIZING, PendingAtomStatus.CANCELLED),
        ],
    )
    def test_legal_transitions(self, src, dst):
        assert is_legal_transition(src, dst)

    @pytest.mark.parametrize(
        "src,dst",
        [
            # PENDING 不能直接 SETTLED / FAILED（必须经过 MATERIALIZING）
            (PendingAtomStatus.PENDING, PendingAtomStatus.SETTLED),
            (PendingAtomStatus.PENDING, PendingAtomStatus.FAILED),
            # MATERIALIZING 不能 EXPIRED（正在跑的不能被超时清扫）
            (PendingAtomStatus.MATERIALIZING, PendingAtomStatus.EXPIRED),
            # MATERIALIZING 不能回到 PENDING
            (PendingAtomStatus.MATERIALIZING, PendingAtomStatus.PENDING),
        ],
    )
    def test_illegal_transitions(self, src, dst):
        assert not is_legal_transition(src, dst)

    @pytest.mark.parametrize(
        "terminal",
        [
            PendingAtomStatus.SETTLED,
            PendingAtomStatus.FAILED,
            PendingAtomStatus.EXPIRED,
            PendingAtomStatus.CANCELLED,
        ],
    )
    def test_terminal_has_no_outgoing_transitions(self, terminal):
        assert allowed_transitions(terminal) == frozenset()
        for any_status in PendingAtomStatus:
            assert not is_legal_transition(terminal, any_status)


# ---------------------------------------------------------------------------
# Snapshot 不变量
# ---------------------------------------------------------------------------


class TestSnapshotInvariants:
    def test_pending_without_resolution_ok(self):
        snap = PendingAtomSnapshot(
            pending_alias="draft_x_0001",
            status=PendingAtomStatus.PENDING,
        )
        assert snap.resolution is None
        assert snap.canonical_uuid is None

    def test_settled_requires_resolution(self):
        with pytest.raises(ValueError, match="requires a resolution"):
            PendingAtomSnapshot(
                pending_alias="draft_x_0001",
                status=PendingAtomStatus.SETTLED,
                resolution=None,
            )

    def test_non_settled_must_not_carry_resolution(self):
        with pytest.raises(ValueError, match="must not carry a resolution"):
            PendingAtomSnapshot(
                pending_alias="draft_x_0001",
                status=PendingAtomStatus.PENDING,
                resolution=PendingAtomResolution.CREATED,
            )

    def test_canonical_resolution_requires_canonical_uuid(self):
        with pytest.raises(ValueError, match="requires canonical_uuid"):
            PendingAtomSnapshot(
                pending_alias="draft_x_0001",
                status=PendingAtomStatus.SETTLED,
                resolution=PendingAtomResolution.CREATED,
                canonical_uuid=None,
            )

    def test_discarded_must_not_carry_canonical(self):
        with pytest.raises(ValueError, match="DISCARDED"):
            PendingAtomSnapshot(
                pending_alias="draft_x_0001",
                status=PendingAtomStatus.SETTLED,
                resolution=PendingAtomResolution.DISCARDED,
                canonical_uuid="uuid-123",
            )

    def test_settled_created_with_canonical_ok(self):
        snap = PendingAtomSnapshot(
            pending_alias="draft_x_0001",
            status=PendingAtomStatus.SETTLED,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_x",
            canonical_uuid="uuid-123",
        )
        assert snap.resolution == PendingAtomResolution.CREATED
        assert snap.canonical_uuid == "uuid-123"


# ---------------------------------------------------------------------------
# Legacy 映射
# ---------------------------------------------------------------------------


class TestLegacyMapping:
    @pytest.mark.parametrize(
        "legacy,expected_status,expected_resolution",
        [
            ("pending", PendingAtomStatus.PENDING, None),
            ("revision", PendingAtomStatus.PENDING, None),
            ("committed", PendingAtomStatus.SETTLED, PendingAtomResolution.CREATED),
            ("merged", PendingAtomStatus.SETTLED, PendingAtomResolution.MERGED),
            ("updated", PendingAtomStatus.SETTLED, PendingAtomResolution.UPDATED),
            ("touched", PendingAtomStatus.SETTLED, PendingAtomResolution.TOUCHED),
            ("discarded", PendingAtomStatus.SETTLED, PendingAtomResolution.DISCARDED),
            ("failed", PendingAtomStatus.FAILED, None),
        ],
    )
    def test_known_legacy_values(self, legacy, expected_status, expected_resolution):
        status, resolution = map_legacy_status(legacy)
        assert status == expected_status
        assert resolution == expected_resolution

    def test_unknown_legacy_value_raises(self):
        with pytest.raises(ValueError, match="Unknown legacy"):
            map_legacy_status("nonexistent_status")


# ---------------------------------------------------------------------------
# PendingAtomCache.snapshot() 派生
# ---------------------------------------------------------------------------


@pytest.fixture
def identity():
    return Identity(user_id="u1", agent_id="a1")


@pytest.fixture
def cache():
    return PendingAtomCache()


def _make_settlement(
    pending_alias: str,
    intent_id: str,
    status: str,
    *,
    canonical_alias: str | None = None,
    canonical_uuid: str | None = None,
) -> PendingAtomSettlement:
    return PendingAtomSettlement(
        pending_alias=pending_alias,
        intent_id=intent_id,
        status=status,
        duplicate_decision=None,
        canonical_alias=canonical_alias,
        canonical_uuid=canonical_uuid,
        message="",
    )


class TestPendingAtomCacheSnapshot:
    def test_snapshot_unknown_alias_returns_none(self, cache):
        assert cache.snapshot("nonexistent") is None

    def test_fresh_write_snapshot_is_pending(self, cache, identity):
        atom = cache.register_write(
            content="hello",
            title="Hello",
            reason=None,
            identity=identity,
        )
        snap = cache.snapshot(atom.pending_alias)
        assert snap is not None
        assert snap.status == PendingAtomStatus.PENDING
        assert snap.resolution is None
        assert snap.canonical_uuid is None

    def test_fresh_update_snapshot_is_pending(self, cache, identity):
        atom = cache.register_update(
            base_alias="fact_x",
            base_uuid="uuid-base",
            instruction="patch it",
            content="new content",
            identity=identity,
        )
        snap = cache.snapshot(atom.pending_alias)
        assert snap is not None
        # 旧 status=REVISION 在新体系下都是 PENDING
        assert snap.status == PendingAtomStatus.PENDING
        assert snap.resolution is None

    def test_committed_settlement_yields_settled_created(self, cache, identity):
        atom = cache.register_write(
            content="hello", title="Hello", reason=None, identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "COMMITTED",
            canonical_alias="fact_hello",
            canonical_uuid="uuid-1",
        ))
        snap = cache.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.CREATED
        assert snap.canonical_alias == "fact_hello"
        assert snap.canonical_uuid == "uuid-1"

    def test_merged_settlement_yields_settled_merged(self, cache, identity):
        atom = cache.register_write(
            content="dup", title="Dup", reason=None, identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "MERGED",
            canonical_alias="fact_dup",
            canonical_uuid="uuid-2",
        ))
        snap = cache.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.MERGED
        assert snap.canonical_uuid == "uuid-2"

    def test_touched_settlement_yields_settled_touched(self, cache, identity):
        atom = cache.register_write(
            content="touch", title="Touch", reason=None, identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "TOUCHED",
            canonical_alias="fact_touch",
            canonical_uuid="uuid-3",
        ))
        snap = cache.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.TOUCHED

    def test_updated_settlement_yields_settled_updated(self, cache, identity):
        atom = cache.register_update(
            base_alias="fact_x",
            base_uuid="uuid-base",
            instruction="patch",
            content=None,
            identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "UPDATED",
            canonical_alias="fact_x",
            canonical_uuid="uuid-base",
        ))
        snap = cache.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.UPDATED
        assert snap.canonical_uuid == "uuid-base"

    def test_discarded_settlement_strips_canonical(self, cache, identity):
        """DISCARDED 不应该携带 canonical_uuid，即便 settlement 端误传也要被剔除。"""
        atom = cache.register_write(
            content="lowq", title="LowQ", reason=None, identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "DISCARDED",
        ))
        snap = cache.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.DISCARDED
        assert snap.canonical_alias is None
        assert snap.canonical_uuid is None

    def test_failed_settlement_yields_failed_status(self, cache, identity):
        atom = cache.register_write(
            content="boom", title="Boom", reason=None, identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "FAILED",
        ))
        snap = cache.snapshot(atom.pending_alias)
        # FAILED 没有 resolution，走 legacy 派生路径
        assert snap.status == PendingAtomStatus.FAILED
        assert snap.resolution is None

    def test_clear_resets_resolution_index(self, cache, identity):
        atom = cache.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        cache.apply_settlement(_make_settlement(
            atom.pending_alias, atom.intent_id, "COMMITTED",
            canonical_alias="fact_x",
            canonical_uuid="uuid-x",
        ))
        assert cache.snapshot(atom.pending_alias).resolution is not None

        cache.clear()
        assert cache.snapshot(atom.pending_alias) is None
