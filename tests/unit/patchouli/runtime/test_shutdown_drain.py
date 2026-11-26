from hivememory.patchouli.control.memory_generation.saved_work import SavedWorkReport
from hivememory.patchouli.runtime.models import RestoredTopicPoolReport
from hivememory.patchouli.runtime.shutdown_drain import (
    shutdown_drain_completed_severity,
    shutdown_drain_completed_status,
    summarize_shutdown_drain_failure,
    summarize_shutdown_drain_result,
    summarize_topic_pool_restore,
    topic_pool_restore_completed_severity,
    topic_pool_restore_completed_status,
)


def test_shutdown_drain_reports_saved_work_and_flags_unwritten_file_as_error():
    saved = SavedWorkReport(saved_by_source={"WRITE": 2}, waited_for_commit=1, file_written=False)
    result = {"reentrant": False, "success": False, "generation": saved}

    summary = summarize_shutdown_drain_result(result)

    assert summary == {
        "reentrant": False,
        "success": False,
        "generation": {
            "saved": 2,
            "saved_by_source": {"WRITE": 2},
            "waited_for_commit": 1,
            "file_written": False,
        },
    }
    assert shutdown_drain_completed_status(result) == "completed_without_saved_work"
    assert shutdown_drain_completed_severity(result) == "error"


def test_summarize_shutdown_drain_failure_is_minimal():
    summary = summarize_shutdown_drain_failure(RuntimeError("boom"))

    assert summary == {"reentrant": False, "generation": None}


def test_topic_pool_restore_flags_rejected_logs_as_error():
    report = RestoredTopicPoolReport(restored=2, retired=1, rejected=1)

    assert summarize_topic_pool_restore(report) == {"restored": 2, "retired": 1, "rejected": 1}
    assert topic_pool_restore_completed_status(report) == "completed_with_rejections"
    assert topic_pool_restore_completed_severity(report) == "error"
