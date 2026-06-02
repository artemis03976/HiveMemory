"""
PendingAtom 统一状态体系单元测试 (Commit 1)。

覆盖：
- PendingAtomStatus / PendingAtomResolution 枚举属性
- 状态机合法/非法迁移
- PendingAtomSnapshot 不变量校验
- PendingAtomRuntime.snapshot() 派生路径（未结算 / SETTLED 各类 resolution）
"""

from __future__ import annotations

import pytest

from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.core.models import (
    Identity,
    InvalidStateTransition,
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomSnapshot,
    PendingAtomStatus,
)
from hivememory.core.models.pending import (
    allowed_transitions,
    is_legal_transition,
)

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
# PendingAtomRuntime.snapshot() 派生
# ---------------------------------------------------------------------------


@pytest.fixture
def identity():
    return Identity(user_id="u1", agent_id="a1")


@pytest.fixture
def runtime():
    return PendingAtomRuntime()


def _make_settlement(
    pending_alias: str,
    intent_id: str,
    resolution: PendingAtomResolution,
    *,
    canonical_alias: str | None = None,
    canonical_uuid: str | None = None,
) -> PendingAtomSettlement:
    return PendingAtomSettlement(
        pending_alias=pending_alias,
        intent_id=intent_id,
        resolution=resolution,
        duplicate_decision=None,
        canonical_alias=canonical_alias,
        canonical_uuid=canonical_uuid,
        message="",
    )


class TestPendingAtomRuntimeSnapshot:
    def test_snapshot_unknown_alias_returns_none(self, runtime):
        assert runtime.snapshot("nonexistent") is None

    def test_fresh_write_snapshot_is_pending(self, runtime, identity):
        atom = runtime.register_write(
            content="hello",
            title="Hello",
            reason=None,
            identity=identity,
        )
        snap = runtime.snapshot(atom.pending_alias)
        assert snap is not None
        assert snap.status == PendingAtomStatus.PENDING
        assert snap.resolution is None
        assert snap.canonical_uuid is None

    def test_fresh_update_snapshot_is_pending(self, runtime, identity):
        atom = runtime.register_update(
            base_alias="fact_x",
            base_uuid="uuid-base",
            instruction="patch it",
            content="new content",
            identity=identity,
        )
        snap = runtime.snapshot(atom.pending_alias)
        assert snap is not None
        # 旧 status=REVISION 在新体系下都是 PENDING
        assert snap.status == PendingAtomStatus.PENDING
        assert snap.resolution is None

    def test_committed_settlement_yields_settled_created(self, runtime, identity):
        atom = runtime.register_write(
            content="hello", title="Hello", reason=None, identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(_make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.CREATED,
            canonical_alias="fact_hello",
            canonical_uuid="uuid-1",
        ))
        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.CREATED
        assert snap.canonical_alias == "fact_hello"
        assert snap.canonical_uuid == "uuid-1"

    def test_settle_from_pending_is_illegal(self, runtime, identity):
        atom = runtime.register_write(
            content="hello", title="Hello", reason=None, identity=identity,
        )
        settlement = _make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.CREATED,
            canonical_alias="fact_hello",
            canonical_uuid="uuid-1",
        )

        with pytest.raises(InvalidStateTransition, match="pending -> settled"):
            runtime.settle(settlement)

        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.PENDING
        assert snap.resolution is None

    def test_settle_with_mismatched_intent_is_ignored(self, runtime, identity):
        atom = runtime.register_write(
            content="hello", title="Hello", reason=None, identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])

        runtime.settle(_make_settlement(
            atom.pending_alias,
            "intent_other",
            PendingAtomResolution.CREATED,
            canonical_alias="fact_hello",
            canonical_uuid="uuid-1",
        ))

        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.MATERIALIZING
        assert snap.resolution is None
        assert atom.settlement is None

    def test_merged_settlement_yields_settled_merged(self, runtime, identity):
        atom = runtime.register_write(
            content="dup", title="Dup", reason=None, identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(_make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.MERGED,
            canonical_alias="fact_dup",
            canonical_uuid="uuid-2",
        ))
        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.MERGED
        assert snap.canonical_uuid == "uuid-2"

    def test_touched_settlement_yields_settled_touched(self, runtime, identity):
        atom = runtime.register_write(
            content="touch", title="Touch", reason=None, identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(_make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.TOUCHED,
            canonical_alias="fact_touch",
            canonical_uuid="uuid-3",
        ))
        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.TOUCHED

    def test_updated_settlement_yields_settled_updated(self, runtime, identity):
        atom = runtime.register_update(
            base_alias="fact_x",
            base_uuid="uuid-base",
            instruction="patch",
            content=None,
            identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(_make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.UPDATED,
            canonical_alias="fact_x",
            canonical_uuid="uuid-base",
        ))
        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.UPDATED
        assert snap.canonical_uuid == "uuid-base"

    def test_discarded_settlement_strips_canonical(self, runtime, identity):
        """DISCARDED 不应该携带 canonical_uuid，即便 settlement 端误传也要被剔除。"""
        atom = runtime.register_write(
            content="lowq", title="LowQ", reason=None, identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(_make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.DISCARDED,
        ))
        snap = runtime.snapshot(atom.pending_alias)
        assert snap.status == PendingAtomStatus.SETTLED
        assert snap.resolution == PendingAtomResolution.DISCARDED
        assert snap.canonical_alias is None
        assert snap.canonical_uuid is None

    def test_clear_removes_settlement_source(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(_make_settlement(
            atom.pending_alias, atom.intent_id, PendingAtomResolution.CREATED,
            canonical_alias="fact_x",
            canonical_uuid="uuid-x",
        ))
        assert runtime.snapshot(atom.pending_alias).resolution is not None

        runtime.clear()
        assert runtime.snapshot(atom.pending_alias) is None


class TestPendingAtomRuntimeCommands:
    def test_start_materializing(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )

        runtime.start_materializing(atom.pending_alias)

        assert atom.status == PendingAtomStatus.MATERIALIZING

    def test_start_materializing_rejects_terminal_atom(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        runtime.expire(atom.pending_alias)

        with pytest.raises(InvalidStateTransition, match="expired -> materializing"):
            runtime.start_materializing(atom.pending_alias)

    def test_mark_failed_materializing_atom(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        runtime.start_materializing(atom.pending_alias)

        runtime.mark_failed(atom.pending_alias)

        assert atom.status == PendingAtomStatus.FAILED

    def test_mark_failed_keeps_idempotent_skip_for_pending(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )

        runtime.mark_failed(atom.pending_alias)

        assert atom.status == PendingAtomStatus.PENDING

    def test_cancel_pending_atom(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )

        runtime.cancel(atom.pending_alias)

        assert atom.status == PendingAtomStatus.CANCELLED

    def test_cancel_materializing_atom(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        runtime.start_materializing(atom.pending_alias)

        runtime.cancel(atom.pending_alias)

        assert atom.status == PendingAtomStatus.CANCELLED

    def test_expire_pending_atom(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )

        runtime.expire(atom.pending_alias)

        assert atom.status == PendingAtomStatus.EXPIRED

    def test_expire_materializing_atom_is_illegal(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        runtime.start_materializing(atom.pending_alias)

        with pytest.raises(InvalidStateTransition, match="materializing -> expired"):
            runtime.expire(atom.pending_alias)

    def test_duplicate_settle_is_illegal(self, runtime, identity):
        atom = runtime.register_write(
            content="x", title="X", reason=None, identity=identity,
        )
        settlement = _make_settlement(
            atom.pending_alias,
            atom.intent_id,
            PendingAtomResolution.CREATED,
            canonical_alias="fact_x",
            canonical_uuid="uuid-x",
        )
        runtime.claim_for_materialization([atom.pending_alias])
        runtime.settle(settlement)

        with pytest.raises(InvalidStateTransition, match="settled -> settled"):
            runtime.settle(settlement)
