from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope


def test_create_frame_uses_only_run_and_frame_coordinates() -> None:
    frame = ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "system", "content": "test"}],
        topic_id="topic-123",
        identity=Identity(user_id="u1"),
    )

    assert not frame.is_transient()
    assert frame.harvested_aliases == []
    assert frame.runtime_scope.run_id == "run-1"
    assert not hasattr(frame.runtime_scope, "depth")
    assert not hasattr(frame.runtime_scope, "parent_frame_id")
    assert not hasattr(frame, "is_main_frame")
    assert not hasattr(frame, "is_sub_frame")


def test_transient_frame_is_a_normal_frame_without_parent_metadata() -> None:
    frame = ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-2"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "write tests"}],
        topic_id=None,
        identity=Identity(user_id="u1"),
    )

    assert frame.is_transient()
    assert frame.runtime_scope.run_id == "run-1"
    assert not hasattr(frame.runtime_scope, "parent_frame_id")


def test_harvest_alias_deduplicates() -> None:
    frame = ExecutionFrame(
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id=None,
        identity=Identity(user_id="u1"),
    )
    frame.add_harvested_alias("mem-code-1")
    frame.add_harvested_alias("mem-code-2")
    frame.add_harvested_alias("mem-code-1")

    assert frame.harvested_aliases == ["mem-code-1", "mem-code-2"]
