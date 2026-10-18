from __future__ import annotations

import pytest

from hivememory.agent_runtime.frame import ExecutionFrame, FrameExecutionPolicy
from hivememory.alice.orchestration.frames import FrameFactory, FrameSpec
from hivememory.alice.orchestration.session import RunSession
from hivememory.core.models import OMNI_DOLL_PROFILE
from tests.helpers.call import StubOperationSubmitter
from tests.helpers.workspace import make_runtime_scope


def _frame(run_id: str, frame_id: str, policy: FrameExecutionPolicy) -> ExecutionFrame:
    return FrameFactory().create(
        FrameSpec(
            submit_operation=StubOperationSubmitter(),
            runtime_scope=make_runtime_scope(run_id=run_id, frame_id=frame_id),
            profile=OMNI_DOLL_PROFILE,
            messages=[],
            topic_id=None,
            execution_policy=policy,
        )
    )


def test_session_registers_frame_under_frame_id() -> None:
    policy = FrameExecutionPolicy(max_iterations=7, permitted_verbs=frozenset())
    session = RunSession(agent_run_id="run-a", process_id="process-1")
    frame = _frame("run-a", "frame-a", policy)

    session.register_root_frame(frame)

    assert session.frames == {"frame-a": frame}


def test_session_rejects_frame_id_collision_and_unregistered_call() -> None:
    session = RunSession(agent_run_id="run-a", process_id="process-1")
    registered = _frame("run-a", "frame-a", FrameExecutionPolicy(permitted_verbs=frozenset()))
    collision = _frame("run-a", "frame-a", FrameExecutionPolicy(permitted_verbs=frozenset()))
    unregistered = _frame("run-a", "frame-b", FrameExecutionPolicy(permitted_verbs=frozenset()))
    session.register_root_frame(registered)

    with pytest.raises(RuntimeError, match="already exists"):
        session.register_callee_frame(
            collision, session.register_call(registered, "action-collision")
        )
    with pytest.raises(ValueError, match="not registered"):
        session.register_call(unregistered, "action-b")

    assert session.frames == {"frame-a": registered}


def test_interleaved_sessions_keep_records_isolated() -> None:
    session_a = RunSession(agent_run_id="run-a", process_id="process-1")
    session_b = RunSession(agent_run_id="run-b", process_id="process-1")
    frame_a = _frame("run-a", "frame-a", FrameExecutionPolicy(permitted_verbs=frozenset()))
    frame_b = _frame("run-b", "frame-b", FrameExecutionPolicy(permitted_verbs=frozenset()))
    session_a.register_root_frame(frame_a)
    session_b.register_root_frame(frame_b)

    record = session_b.register_call(frame_b, "action-b")

    assert session_a.call_records == {}
    assert session_b.call_records[("frame-b", "action-b")] is record


def test_session_registers_root_and_callee_with_explicit_run_local_relationship() -> None:
    session = RunSession(agent_run_id="run-a", process_id="process-1")
    root = _frame("run-a", "frame-root", FrameExecutionPolicy(permitted_verbs=frozenset()))
    callee = _frame("run-a", "frame-callee", FrameExecutionPolicy(permitted_verbs=frozenset()))

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


def test_session_rejects_duplicate_root_callee_binding_and_cross_run_frame() -> None:
    session = RunSession(agent_run_id="run-a", process_id="process-1")
    root = _frame("run-a", "frame-root", FrameExecutionPolicy(permitted_verbs=frozenset()))
    other_root = _frame("run-a", "frame-other", FrameExecutionPolicy(permitted_verbs=frozenset()))
    cross_run = _frame("run-b", "frame-cross", FrameExecutionPolicy(permitted_verbs=frozenset()))
    callee = _frame("run-a", "frame-callee", FrameExecutionPolicy(permitted_verbs=frozenset()))
    session.register_root_frame(root)

    with pytest.raises(RuntimeError, match="already has a root"):
        session.register_root_frame(other_root)
    with pytest.raises(ValueError, match="run_id does not match"):
        session.register_callee_frame(cross_run, session.register_call(root, "action-cross"))

    record = session.register_call(root, "action-a")
    record.begin_resolution()
    session.register_callee_frame(callee, record)
    with pytest.raises(RuntimeError, match="already exists"):
        session.register_callee_frame(callee, record)
