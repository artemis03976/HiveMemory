"""Local Work Queue Runtime Q0 契约测试。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hivememory.system.runtime.work_queue import (
    TERMINAL_WORK_STATES,
    FailureAction,
    FailureDecision,
    QueuePolicy,
    WorkItem,
    WorkReceipt,
    WorkRecord,
    WorkState,
    can_transition_work_state,
    decode_canonical_json,
    encode_canonical_json,
)


def test_work_state_transitions_are_closed_after_terminal_state() -> None:
    assert can_transition_work_state(WorkState.QUEUED, WorkState.RUNNING)
    assert can_transition_work_state(WorkState.RUNNING, WorkState.RETRY_WAIT)
    assert can_transition_work_state(WorkState.RETRY_WAIT, WorkState.QUEUED)
    assert can_transition_work_state(WorkState.RUNNING, WorkState.SUCCEEDED)

    for state in TERMINAL_WORK_STATES:
        assert not any(can_transition_work_state(state, target) for target in WorkState)


def test_work_item_is_an_immutable_bytes_envelope() -> None:
    source_payload = {"topic_id": "topic-1", "events": ["created"]}
    item = WorkItem(
        work_id="work-1",
        lane="memory_generation",
        kind="patchouli.memory_generation.v1",
        schema_version=1,
        payload=encode_canonical_json(source_payload),
        idempotency_key="intent-1",
    )

    with pytest.raises(FrozenInstanceError):
        item.lane = "runtime_job"  # type: ignore[misc]
    source_payload["events"].append("changed")  # type: ignore[union-attr]
    assert decode_canonical_json(item.payload) == {
        "events": ["created"],
        "topic_id": "topic-1",
    }
    with pytest.raises(ValueError, match="lane"):
        WorkItem(
            work_id="work-2",
            lane=" ",
            kind="example.v1",
            schema_version=1,
            payload=encode_canonical_json({}),
        )
    with pytest.raises(ValueError, match="schema_version"):
        WorkItem(
            work_id="work-3",
            lane="example",
            kind="example.v1",
            schema_version=0,
            payload=encode_canonical_json({}),
        )
    with pytest.raises(TypeError, match="schema_version"):
        WorkItem(
            work_id="work-3b",
            lane="example",
            kind="example.v1",
            schema_version=True,  # type: ignore[arg-type]
            payload=encode_canonical_json({}),
        )

    with pytest.raises(TypeError, match="payload must be bytes"):
        WorkItem(
            work_id="work-4",
            lane="example",
            kind="example.v1",
            schema_version=1,
            payload={},  # type: ignore[arg-type]
        )
    non_canonical_item = WorkItem(
        work_id="work-5",
        lane="example",
        kind="example.v1",
        schema_version=1,
        payload=b'{"topic_id": "topic-1"}',
    )
    assert decode_canonical_json(non_canonical_item.payload) == {"topic_id": "topic-1"}


def test_work_record_keeps_runtime_state_outside_work_item() -> None:
    now = datetime.now(UTC)
    item = WorkItem(
        work_id="work-1",
        lane="interaction_submission",
        kind="patchouli.interaction_submission.v1",
        schema_version=1,
        payload=encode_canonical_json({"interaction_id": "interaction-1"}),
    )
    record = WorkRecord(
        item=item,
        state=WorkState.QUEUED,
        attempt_count=0,
        enqueued_at=now,
        available_at=now,
    )

    assert record.work_id == item.work_id
    assert record.lane == item.lane
    assert not hasattr(item, "attempt_count")


def test_receipt_only_records_runtime_acceptance_snapshot() -> None:
    receipt = WorkReceipt(
        work_id="work-1",
        lane="runtime_job",
        state=WorkState.QUEUED,
        enqueued_at=datetime.now(UTC),
    )

    assert receipt.state == WorkState.QUEUED
    assert not hasattr(receipt, "result")
    assert not hasattr(receipt, "outcome")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_attempts": -1}, "max_attempts"),
        ({"terminal_retention": -1}, "terminal_retention"),
    ],
)
def test_queue_policy_rejects_invalid_limits(changes: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "capacity": 10,
        "max_concurrency": 2,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        QueuePolicy(**values)  # type: ignore[arg-type]


def test_zero_max_attempts_leaves_retry_limit_to_business_decision() -> None:
    policy = QueuePolicy(capacity=10, max_concurrency=2, max_attempts=0)

    assert policy.max_attempts == 0


def test_failure_decision_only_allows_retry_delay_for_retry() -> None:
    decision = FailureDecision(
        action=FailureAction.RETRY,
        retry_after_seconds=1.5,
        reason="transient",
    )

    assert decision.retry_after_seconds == 1.5
    with pytest.raises(ValueError, match="only valid for retry"):
        FailureDecision(action=FailureAction.FAIL, retry_after_seconds=1)


def test_public_contract_has_no_business_or_server_imports() -> None:
    package_root = (
        Path(__file__).parents[5] / "src" / "hivememory" / "system" / "runtime" / "work_queue"
    )
    forbidden_prefixes = (
        "hivememory.patchouli",
        "hivememory.alice",
        "hivememory.server",
    )

    imported_modules: set[str] = set()
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

    assert not {module for module in imported_modules if module.startswith(forbidden_prefixes)}
