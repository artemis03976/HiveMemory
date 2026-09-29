"""任务进程 stop 控制契约测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hivememory.core.errors import WorkspaceDomainError
from hivememory.workspace.process.table import (
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
    ProcessTable,
)
from hivememory.workspace.process.task_process import _run_interruptible
from tests.helpers.workspace import make_identity_scope


def _run(process_id: str) -> ProcessRecord:
    return ProcessRecord(
        identity_scope=make_identity_scope(),
        process_id=process_id,
    )


@pytest.mark.asyncio
async def test_gateway_stop_cancels_bound_task_immediately() -> None:
    run = _run("process-1")
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    run.bind_phase(ProcessPhase.GATEWAY, task)

    result = run.request_stop()

    assert result.accepted is True
    assert result.reason == "user_requested"
    assert run.outcome is ProcessOutcome.STOP_REQUESTED
    assert run.phase is ProcessPhase.GATEWAY
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_prepare_stop_records_request_without_cancelling_prepare_task() -> None:
    run = _run("process-2")
    run.enter_phase(ProcessPhase.PREPARE)
    blocker = asyncio.Event()
    prepare_task = asyncio.create_task(blocker.wait())

    result = run.request_stop("during_prepare")

    assert result.accepted is True
    assert run.outcome is ProcessOutcome.STOP_REQUESTED
    assert run.stop_reason == "during_prepare"
    assert prepare_task.cancelled() is False
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task


def test_finalize_and_terminal_stop_are_rejected() -> None:
    finalizing = _run("process-3")
    finalizing.enter_phase(ProcessPhase.FINALIZE)
    result = finalizing.request_stop()
    assert result.accepted is False
    assert result.reason == "already_finalizing"
    assert finalizing.outcome is ProcessOutcome.RUNNING

    terminal = _run("process-4")
    terminal.phase = ProcessPhase.TERMINAL
    terminal.outcome = ProcessOutcome.COMPLETED
    result = terminal.request_stop()
    assert result.accepted is False
    assert result.reason == "already_terminal"
    assert terminal.outcome is ProcessOutcome.COMPLETED


def test_repeated_stop_keeps_first_reason_and_does_not_cancel_again() -> None:
    run = _run("process-5")
    task = MagicMock()
    task.done.return_value = False
    run.bind_phase(ProcessPhase.ALICE, task)

    first = run.request_stop("first_reason")
    second = run.request_stop("second_reason")

    assert first.accepted is True
    assert second.accepted is True
    assert second.reason == "first_reason"
    task.cancel.assert_called_once_with()


def test_registry_not_found_and_terminal_results_are_stable() -> None:
    process_table = ProcessTable()

    identity_scope = make_identity_scope()
    missing = process_table.cancel("missing-process", identity_scope)
    assert missing.cancelled is False
    assert missing.status == "not_found"

    run = _run("process-6")
    run.phase = ProcessPhase.TERMINAL
    run.outcome = ProcessOutcome.FAILED
    process_table.register(run)

    terminal = process_table.cancel(run.process_id, run.identity_scope)
    assert terminal.cancelled is False
    assert terminal.reason == "already_terminal"
    assert run.outcome is ProcessOutcome.FAILED


def test_stop_after_bound_task_finished_is_accepted_without_second_cancel() -> None:
    run = _run("process-7")
    task = MagicMock()
    task.done.return_value = True
    run.bind_phase(ProcessPhase.GATEWAY, task)

    result = run.request_stop("late_stop")

    assert result.accepted is True
    assert result.reason == "late_stop"
    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_owner_task_cancellation_is_not_translated_to_chat_run_cancelled() -> None:
    run = _run("process-8")
    blocker = asyncio.Event()

    async def operation():
        await blocker.wait()

    task = asyncio.create_task(_run_interruptible(run, ProcessPhase.GATEWAY, operation))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.outcome is ProcessOutcome.RUNNING


def test_registry_hides_run_from_different_workspace_control_plane() -> None:
    """防止仅凭 process_id 跨 Workspace 查询或取消另一条进程记录。"""
    process_table = ProcessTable()
    owner_context = make_identity_scope(
        user_id="u1",
        workspace_id="main_workspace",
        interaction_id="interaction-main",
    )
    other_context = make_identity_scope(
        user_id="u1",
        workspace_id="isolation_workspace",
        interaction_id="interaction-isolation",
    )
    run = ProcessRecord(
        identity_scope=owner_context,
        process_id="shared-process-id",
    )
    process_table.register(run)

    assert process_table.get(run.process_id, other_context) is None
    assert process_table.status(run.process_id, other_context) is None
    rejected = process_table.cancel(run.process_id, other_context)
    assert rejected.status == "not_found"
    assert rejected.cancelled is False
    assert run.outcome is ProcessOutcome.RUNNING


def test_registry_rejects_process_id_collision_without_overwriting_owner() -> None:
    """防止重复 process_id 覆盖既有 scope 并把控制权转给后注册者。"""
    process_table = ProcessTable()
    original = _run("collision")
    replacement = ProcessRecord(
        identity_scope=make_identity_scope(
            workspace_id="isolation_workspace",
        ),
        process_id="collision",
    )
    process_table.register(original)

    with pytest.raises(WorkspaceDomainError, match="拒绝覆盖"):
        process_table.register(replacement)

    assert process_table.get("collision", original.identity_scope) is original
    assert process_table.get("collision", replacement.identity_scope) is None
