from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationTaskStatus,
    MemoryGenerationTaskWaitResult,
    MemoryGenerationTaskWaitSummary,
)
from hivememory.patchouli.runtime.shutdown_drain import (
    shutdown_drain_completed_severity,
    shutdown_drain_completed_status,
    summarize_shutdown_drain_failure,
    summarize_shutdown_drain_perception,
    summarize_shutdown_drain_result,
)
from hivememory.patchouli.services.perception import ShutdownFlushResult


def test_summarize_shutdown_drain_result_uses_counts_only():
    perception = ShutdownFlushResult(
        success=True,
        trigger_reason="shutdown",
        flushed_topics=["topic-a", "topic-b"],
        skipped_topics=["topic-c"],
        archived_blocks=3,
    )
    generation = MemoryGenerationTaskWaitSummary(
        requested=1,
        found=1,
        missing=0,
        completed=0,
        failed=0,
        cancelled=0,
        pending=0,
        running=1,
        timed_out=1,
        results=(
            MemoryGenerationTaskWaitResult(
                task_id="memory-task-timeout",
                found=True,
                timed_out=True,
                status=MemoryGenerationTaskStatus.RUNNING,
            ),
        ),
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
        "success": True,
        "trigger_reason": "shutdown",
        "flushed_topic_count": 2,
        "skipped_topic_count": 1,
        "archived_blocks": 3,
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
            "success": True,
            "trigger_reason": "shutdown",
            "flushed_topics": [],
            "skipped_topics": [],
            "archived_blocks": 0,
        }
    )

    assert summary == {
        "success": True,
        "trigger_reason": "shutdown",
        "flushed_topic_count": 0,
        "skipped_topic_count": 0,
        "archived_blocks": 0,
    }


def test_summarize_shutdown_drain_failure_is_minimal():
    summary = summarize_shutdown_drain_failure(RuntimeError("boom"))

    assert summary == {
        "reentrant": False,
        "observer_payloads_submitted": 0,
        "perception": None,
        "generation": None,
    }
