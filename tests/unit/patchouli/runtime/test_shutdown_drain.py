from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.runtime.shutdown_drain import (
    build_shutdown_generation_summary,
    shutdown_drain_completed_severity,
    shutdown_drain_completed_status,
    summarize_shutdown_drain_failure,
    summarize_shutdown_drain_perception,
    summarize_shutdown_drain_result,
)


def test_summarize_shutdown_drain_result_uses_counts_only():
    perception = TopicShutdownFlushReport(
        settled_topic_ids=("topic-a", "topic-b"),
        resident_block_count=3,
    )
    generation = build_shutdown_generation_summary(
        [
            MemoryGenerationTask(
                task_id="memory-task-timeout",
                topic_id="topic-1",
                label="timeout",
                source=MemoryGenerationSource.WRITE,
                status=MemoryGenerationTaskStatus.RUNNING,
            )
        ]
    )

    summary = summarize_shutdown_drain_result(
        {
            "reentrant": False,
            "success": False,
            "observer_payloads_submitted": 0,
            "perception": perception,
            "generation": generation,
            "generation_cancelled_after_timeout": 1,
        }
    )

    assert summary["perception"] == {
        "settled_topic_count": 2,
        "resident_block_count": 3,
    }
    assert summary["generation"]["running"] == 1
    assert summary["generation"]["timed_out"] == 1
    assert summary["generation_cancelled_after_timeout"] == 1
    assert shutdown_drain_completed_status({"generation": generation}) == (
        "completed_with_timeout"
    )
    assert shutdown_drain_completed_severity({"generation": generation}) == "warning"


def test_summarize_shutdown_drain_perception_accepts_reentrant_dict():
    summary = summarize_shutdown_drain_perception(
        {
            "settled_topic_ids": [],
            "resident_block_count": 0,
        }
    )

    assert summary == {
        "settled_topic_count": 0,
        "resident_block_count": 0,
    }


def test_summarize_shutdown_drain_failure_is_minimal():
    summary = summarize_shutdown_drain_failure(RuntimeError("boom"))

    assert summary == {
        "reentrant": False,
        "perception": None,
        "generation": None,
    }
