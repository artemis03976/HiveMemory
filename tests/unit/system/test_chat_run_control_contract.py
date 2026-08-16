"""Chat Run 取消控制契约测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hivememory.system.application.chat_service import _run_interruptible
from hivememory.system.runtime.control import (
    ChatGenerationRun,
    ChatGenerationRunRegistry,
    ChatRunOutcome,
    ChatRunPhase,
)


@pytest.mark.asyncio
async def test_gateway_stop_cancels_bound_task_immediately() -> None:
    run = ChatGenerationRun(generation_id="generation-1")
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    run.bind_phase(ChatRunPhase.GATEWAY, task)

    result = run.request_stop()

    assert result.accepted is True
    assert result.reason == "user_requested"
    assert run.outcome is ChatRunOutcome.STOP_REQUESTED
    assert run.phase is ChatRunPhase.GATEWAY
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_prepare_stop_records_request_without_cancelling_prepare_task() -> None:
    run = ChatGenerationRun(generation_id="generation-2")
    run.enter_phase(ChatRunPhase.PREPARE)
    blocker = asyncio.Event()
    prepare_task = asyncio.create_task(blocker.wait())

    result = run.request_stop("during_prepare")

    assert result.accepted is True
    assert run.outcome is ChatRunOutcome.STOP_REQUESTED
    assert run.stop_reason == "during_prepare"
    assert prepare_task.cancelled() is False
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task


def test_finalize_and_terminal_stop_are_rejected() -> None:
    finalizing = ChatGenerationRun(generation_id="generation-3")
    finalizing.enter_phase(ChatRunPhase.FINALIZE)
    result = finalizing.request_stop()
    assert result.accepted is False
    assert result.reason == "already_finalizing"
    assert finalizing.outcome is ChatRunOutcome.RUNNING

    terminal = ChatGenerationRun(generation_id="generation-4")
    terminal.phase = ChatRunPhase.TERMINAL
    terminal.outcome = ChatRunOutcome.COMPLETED
    result = terminal.request_stop()
    assert result.accepted is False
    assert result.reason == "already_terminal"
    assert terminal.outcome is ChatRunOutcome.COMPLETED


def test_repeated_stop_keeps_first_reason_and_does_not_cancel_again() -> None:
    run = ChatGenerationRun(generation_id="generation-5")
    task = MagicMock()
    task.done.return_value = False
    run.bind_phase(ChatRunPhase.ALICE, task)

    first = run.request_stop("first_reason")
    second = run.request_stop("second_reason")

    assert first.accepted is True
    assert second.accepted is True
    assert second.reason == "first_reason"
    task.cancel.assert_called_once_with()


def test_registry_not_found_and_terminal_results_are_stable() -> None:
    registry = ChatGenerationRunRegistry()

    missing = registry.cancel("missing-generation")
    assert missing.cancelled is False
    assert missing.status == "not_found"

    run = ChatGenerationRun(generation_id="generation-6")
    run.phase = ChatRunPhase.TERMINAL
    run.outcome = ChatRunOutcome.FAILED
    registry.register(run)

    terminal = registry.cancel(run.generation_id)
    assert terminal.cancelled is False
    assert terminal.reason == "already_terminal"
    assert run.outcome is ChatRunOutcome.FAILED


def test_stop_after_bound_task_finished_is_accepted_without_second_cancel() -> None:
    run = ChatGenerationRun(generation_id="generation-7")
    task = MagicMock()
    task.done.return_value = True
    run.bind_phase(ChatRunPhase.GATEWAY, task)

    result = run.request_stop("late_stop")

    assert result.accepted is True
    assert result.reason == "late_stop"
    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_owner_task_cancellation_is_not_translated_to_chat_run_cancelled() -> None:
    run = ChatGenerationRun(generation_id="generation-8")
    blocker = asyncio.Event()

    async def operation():
        await blocker.wait()

    task = asyncio.create_task(_run_interruptible(run, ChatRunPhase.GATEWAY, operation))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.outcome is ChatRunOutcome.RUNNING
