from __future__ import annotations

import asyncio

import pytest

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.alice.runtime.agent.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.runtime.agent.run_session import RunSession
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


@pytest.mark.asyncio
async def test_interleaved_sessions_keep_cancel_and_records_isolated() -> None:
    session_a = RunSession(agent_run_id="run-a")
    session_b = RunSession(agent_run_id="run-b")
    frame_a = _frame("run-a", "frame-a", FrameExecutionPolicy())
    frame_b = _frame("run-b", "frame-b", FrameExecutionPolicy())
    session_a.register_frame(frame_a)
    session_b.register_frame(frame_b)

    async def cancel_one() -> None:
        await asyncio.sleep(0)
        session_a.cancel_event.set()

    task = asyncio.create_task(cancel_one())
    await asyncio.sleep(0)
    record = session_b.register_call(frame_b, "action-b")
    await task

    assert session_a.cancel_event.is_set()
    assert not session_b.cancel_event.is_set()
    assert session_a.call_records == {}
    assert session_b.call_records[("frame-b", "action-b")] is record
