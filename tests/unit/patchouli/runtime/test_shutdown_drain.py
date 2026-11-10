from hivememory.patchouli.control.memory_generation.saved_work import SavedWorkReport
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.runtime.shutdown_drain import (
    shutdown_drain_completed_severity,
    shutdown_drain_completed_status,
    summarize_shutdown_drain_failure,
    summarize_shutdown_drain_perception,
    summarize_shutdown_drain_result,
)


def test_shutdown_drain_reports_saved_work_and_flags_unwritten_file_as_error():
    perception = TopicShutdownFlushReport(
        settled_topic_ids=("topic-a", "topic-b"),
        generation_skipped_topic_ids=("topic-b",),
        resident_block_count=3,
    )
    saved = SavedWorkReport(
        saved_by_source={"SETTLE": 1, "WRITE": 2}, waited_for_commit=1, file_written=False
    )
    result = {"reentrant": False, "success": False, "perception": perception, "generation": saved}

    summary = summarize_shutdown_drain_result(result)

    assert summary["perception"] == {
        "settled_topic_count": 2,
        "generation_skipped_topic_count": 1,
        "resident_block_count": 3,
    }
    assert summary["generation"] == {
        "saved": 3,
        "saved_by_source": {"SETTLE": 1, "WRITE": 2},
        "waited_for_commit": 1,
        "file_written": False,
    }
    assert shutdown_drain_completed_status(result) == "completed_without_saved_work"
    assert shutdown_drain_completed_severity(result) == "error"


def test_summarize_shutdown_drain_perception_accepts_reentrant_dict():
    summary = summarize_shutdown_drain_perception(
        {
            "settled_topic_ids": [],
            "generation_skipped_topic_ids": [],
            "resident_block_count": 0,
        }
    )

    assert summary == {
        "settled_topic_count": 0,
        "generation_skipped_topic_count": 0,
        "resident_block_count": 0,
    }


def test_summarize_shutdown_drain_failure_is_minimal():
    summary = summarize_shutdown_drain_failure(RuntimeError("boom"))

    assert summary == {
        "reentrant": False,
        "perception": None,
        "generation": None,
    }
