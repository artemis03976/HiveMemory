from __future__ import annotations

import pytest

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope


def _frame(run_id: str, frame_id: str, policy: FrameExecutionPolicy) -> ExecutionFrame:
    return FrameFactory().create(
        FrameSpec(
            runtime_scope=RuntimeScope(run_id=run_id, frame_id=frame_id),
            profile=OMNI_DOLL_PROFILE,
            identity=Identity(user_id="u"),
            messages=[],
            topic_id=None,
            execution_policy=policy,
        )
    )


def test_frame_factory_is_stateless_and_session_owns_registry() -> None:
    policy = FrameExecutionPolicy.from_profile(OMNI_DOLL_PROFILE, max_iterations=7)
    session = RunSession(agent_run_id="run-a")
    frame = _frame("run-a", "frame-a", policy)

    session.register_frame(frame)

    assert session.frames == {"frame-a": frame}
    assert frame.execution_policy.max_iterations == 7
    assert FrameFactory().scope(run_id="run-a").frame_id != frame.runtime_scope.frame_id


def test_session_rejects_frame_id_collision_and_unregistered_call() -> None:
    session = RunSession(agent_run_id="run-a")
    registered = _frame("run-a", "frame-a", FrameExecutionPolicy())
    collision = _frame("run-a", "frame-a", FrameExecutionPolicy())
    unregistered = _frame("run-a", "frame-b", FrameExecutionPolicy())
    session.register_frame(registered)

    with pytest.raises(RuntimeError, match="already exists"):
        session.register_frame(collision)
    with pytest.raises(ValueError, match="not registered"):
        session.register_call(unregistered, "action-b")

    session.register_frame(registered)
    assert session.frames == {"frame-a": registered}


def test_callee_policy_removes_call_without_reading_frame_depth() -> None:
    policy = FrameExecutionPolicy.from_profile(
        OMNI_DOLL_PROFILE,
        max_iterations=3,
        denied_verbs={"CALL"},
    )

    assert policy.allows("READ")
    assert not policy.allows("CALL")


def test_interleaved_sessions_keep_records_isolated() -> None:
    session_a = RunSession(agent_run_id="run-a")
    session_b = RunSession(agent_run_id="run-b")
    frame_a = _frame("run-a", "frame-a", FrameExecutionPolicy())
    frame_b = _frame("run-b", "frame-b", FrameExecutionPolicy())
    session_a.register_frame(frame_a)
    session_b.register_frame(frame_b)

    record = session_b.register_call(frame_b, "action-b")

    assert session_a.call_records == {}
    assert session_b.call_records[("frame-b", "action-b")] is record


def test_session_registers_root_and_callee_with_explicit_run_local_relationship() -> None:
    session = RunSession(agent_run_id="run-a")
    root = _frame("run-a", "frame-root", FrameExecutionPolicy())
    callee = _frame("run-a", "frame-callee", FrameExecutionPolicy())

    session.register_root_frame(root)
    record = session.register_call(root, "action-a")
    record.begin_resolution()
    session.register_callee_frame(callee, record)

    assert session.root_frame_id == "frame-root"
    assert record.callee_frame_id == "frame-callee"
    assert session.call_for_callee("frame-callee") is record
    assert session.frames == {
        "frame-root": root,
        "frame-callee": callee,
    }
    assert not hasattr(session, "frame_statuses")
    assert not hasattr(session, "active_frame_id")


def test_session_rejects_duplicate_root_callee_binding_and_cross_run_frame() -> None:
    session = RunSession(agent_run_id="run-a")
    root = _frame("run-a", "frame-root", FrameExecutionPolicy())
    other_root = _frame("run-a", "frame-other", FrameExecutionPolicy())
    cross_run = _frame("run-b", "frame-cross", FrameExecutionPolicy())
    callee = _frame("run-a", "frame-callee", FrameExecutionPolicy())
    session.register_root_frame(root)

    with pytest.raises(RuntimeError, match="already has a root"):
        session.register_root_frame(other_root)
    with pytest.raises(ValueError, match="run_id does not match"):
        session.register_frame(cross_run)

    record = session.register_call(root, "action-a")
    record.begin_resolution()
    session.register_callee_frame(callee, record)
    with pytest.raises(RuntimeError, match="already exists"):
        session.register_callee_frame(callee, record)


def test_call_record_callee_binding_requires_resolving_and_is_exactly_once() -> None:
    session = RunSession(agent_run_id="run-a")
    root = _frame("run-a", "frame-root", FrameExecutionPolicy())
    callee = _frame("run-a", "frame-callee", FrameExecutionPolicy())
    session.register_root_frame(root)
    record = session.register_call(root, "action-a")

    with pytest.raises(RuntimeError, match="Cannot bind callee"):
        session.register_callee_frame(callee, record)

    record.begin_resolution()
    session.register_callee_frame(callee, record)
    with pytest.raises(RuntimeError):
        record.bind_callee("frame-other")

    record.cancel()
    record.cancel()
    assert record.status.value == "cancelled"
