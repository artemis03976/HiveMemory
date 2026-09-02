from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.core.models import OMNI_DOLL_PROFILE
from tests.helpers.workspace import make_runtime_scope


def test_frame_with_topic_is_not_transient() -> None:
    frame = ExecutionFrame(
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "system", "content": "test"}],
        topic_id="topic-123",
    )

    assert not frame.is_transient()


def test_transient_frame_is_a_normal_frame_without_parent_metadata() -> None:
    frame = ExecutionFrame(
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-2"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "write tests"}],
        topic_id=None,
    )

    assert frame.is_transient()


def test_harvest_alias_deduplicates() -> None:
    frame = ExecutionFrame(
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id=None,
    )
    frame.add_harvested_alias("mem-code-1")
    frame.add_harvested_alias("mem-code-2")
    frame.add_harvested_alias("mem-code-1")

    assert frame.harvested_aliases == ["mem-code-1", "mem-code-2"]
