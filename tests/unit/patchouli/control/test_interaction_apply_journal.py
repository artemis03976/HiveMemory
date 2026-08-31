"""Interaction apply journal 的状态转换与保留边界测试。"""

import pytest

from hivememory.engines.perception.models import TopicMaterializeTask
from tests.helpers.memory import make_memory_identity_scope
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
    InteractionApplyStage,
)


def test_journal_records_all_apply_stages() -> None:
    journal = InMemoryInteractionApplyJournal()
    settlement = TopicMaterializeTask(
        topic_id="topic-1",
        identity_scope=make_memory_identity_scope(),
    )

    journal.record_interaction_applied("interaction-1", "topic-1", "digest-1")
    record = journal.get("interaction-1")
    assert record is not None
    assert record.stage is InteractionApplyStage.INTERACTION_APPLIED
    assert record.input_digest == "digest-1"

    journal.record_local_completed("interaction-1", "topic-1", settlement)
    record = journal.get("interaction-1")
    assert record is not None
    assert record.stage is InteractionApplyStage.LOCAL_COMPLETED
    assert record.settlement_to_submit is settlement
    assert record.input_digest == "digest-1"

    journal.complete("interaction-1", "topic-1")
    record = journal.get("interaction-1")
    assert record is not None
    assert record.stage is InteractionApplyStage.COMPLETED
    assert record.settlement_to_submit is None


def test_journal_rejects_same_interaction_for_another_topic() -> None:
    journal = InMemoryInteractionApplyJournal()
    journal.record_interaction_applied("interaction-1", "topic-1", "digest-1")

    with pytest.raises(ValueError, match="already applied to topic 'topic-1'"):
        journal.record_interaction_applied("interaction-1", "topic-2", "digest-1")


def test_journal_rejects_same_interaction_with_different_digest() -> None:
    journal = InMemoryInteractionApplyJournal()
    journal.record_interaction_applied("interaction-1", "topic-1", "digest-1")

    with pytest.raises(ValueError, match="different input digest"):
        journal.record_interaction_applied("interaction-1", "topic-1", "digest-2")


def test_journal_rejects_blank_digest() -> None:
    journal = InMemoryInteractionApplyJournal()
    with pytest.raises(ValueError, match="input_digest"):
        journal.record_interaction_applied("interaction-1", "topic-1", "  ")


def test_journal_cannot_complete_before_local_obligations() -> None:
    journal = InMemoryInteractionApplyJournal()
    journal.record_interaction_applied("interaction-1", "topic-1", "digest-1")

    with pytest.raises(RuntimeError, match="local obligations are not completed"):
        journal.complete("interaction-1", "topic-1")


def test_journal_requires_an_applied_interaction_for_later_stages() -> None:
    journal = InMemoryInteractionApplyJournal()

    with pytest.raises(KeyError, match="has not been applied"):
        journal.record_local_completed("missing", "topic-1", None)

    with pytest.raises(KeyError, match="has not been applied"):
        journal.complete("missing", "topic-1")


def test_journal_has_a_bounded_idempotency_window() -> None:
    journal = InMemoryInteractionApplyJournal(max_entries=2)

    for index in range(3):
        interaction_id = f"interaction-{index}"
        topic_id = f"topic-{index}"
        journal.record_interaction_applied(interaction_id, topic_id, f"digest-{index}")
        journal.record_local_completed(interaction_id, topic_id, None)
        journal.complete(interaction_id, topic_id)

    assert len(journal) == 2
    assert journal.get("interaction-0") is None
    assert journal.get("interaction-1") is not None
    assert journal.get("interaction-2") is not None
