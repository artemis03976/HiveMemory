from __future__ import annotations

from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import KoakumaConfig


def test_frame_factory_creates_root_frame_from_spec() -> None:
    factory = FrameFactory()
    frame = factory.create(
        FrameSpec(
            runtime_scope=factory.scope(run_id="run-1", frame_id="frame-1"),
            profile=OMNI_DOLL_PROFILE,
            identity=Identity(user_id="u1"),
            messages=[{"role": "system", "content": "hello"}],
            topic_id="topic-1",
            execution_policy=FrameExecutionPolicy.from_profile(OMNI_DOLL_PROFILE),
        )
    )

    assert frame.runtime_scope.run_id == "run-1"
    assert frame.runtime_scope.frame_id == "frame-1"
    assert frame.topic_id == "topic-1"
    assert frame.working_history == [{"role": "system", "content": "hello"}]
    assert frame.progress.turn_events == []
    assert frame.progress.sequence == 0


def test_frame_factory_creates_transient_frame_with_same_run_id() -> None:
    factory = FrameFactory()
    frame = factory.create(
        FrameSpec(
            runtime_scope=factory.scope(run_id="run-1", frame_id="frame-2"),
            profile=OMNI_DOLL_PROFILE,
            identity=Identity(user_id="u1"),
            messages=[{"role": "user", "content": "task"}],
            topic_id=None,
            execution_policy=FrameExecutionPolicy.from_profile(
                OMNI_DOLL_PROFILE,
                denied_verbs={"CALL"},
            ),
        )
    )

    assert frame.is_transient()
    assert frame.runtime_scope.run_id == "run-1"
    assert frame.execution_policy.allows("READ")
    assert not frame.execution_policy.allows("CALL")
    assert [event.kind for event in frame.progress.turn_events] == ["user_message"]
    assert frame.progress.turn_events[0].content == "task"
    assert frame.progress.turn_events[0].sequence == 0
    assert frame.progress.sequence == 1


def test_frame_factory_records_only_latest_user_message() -> None:
    factory = FrameFactory()
    messages = [
        {"role": "system", "content": "constraints"},
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current"},
    ]

    frame = factory.create(
        FrameSpec(
            runtime_scope=factory.scope(run_id="run-1", frame_id="frame-3"),
            profile=OMNI_DOLL_PROFILE,
            identity=Identity(user_id="u1"),
            messages=messages,
            topic_id="topic-1",
            execution_policy=FrameExecutionPolicy.from_profile(OMNI_DOLL_PROFILE),
        )
    )

    assert frame.working_history == messages
    assert len(frame.progress.turn_events) == 1
    assert frame.progress.turn_events[0].content == "current"
    assert frame.progress.turn_events[0].sequence == 0
    assert frame.progress.sequence == 1


def test_frame_factory_does_not_record_empty_latest_user_message() -> None:
    factory = FrameFactory()
    frame = factory.create(
        FrameSpec(
            runtime_scope=factory.scope(run_id="run-1", frame_id="frame-4"),
            profile=OMNI_DOLL_PROFILE,
            identity=Identity(user_id="u1"),
            messages=[
                {"role": "user", "content": "previous"},
                {"role": "user", "content": ""},
            ],
            topic_id="topic-1",
            execution_policy=FrameExecutionPolicy.from_profile(OMNI_DOLL_PROFILE),
        )
    )

    assert frame.progress.turn_events == []
    assert frame.progress.sequence == 0


def test_sub_agent_prompt_disables_call() -> None:
    assembler = AgentPromptAssembler(KoakumaConfig())

    messages = assembler.build_sub_agent_messages(
        profile=OMNI_DOLL_PROFILE,
        task="Write unit tests",
    )

    assert "CALL" not in messages[0]["content"]
    assert "READ" in messages[0]["content"]
